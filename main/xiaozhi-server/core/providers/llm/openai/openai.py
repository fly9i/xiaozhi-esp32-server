import httpx
import openai
from openai.types import CompletionUsage
from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.llm.base import LLMProviderBase
from urllib.parse import urlparse

TAG = __name__
logger = setup_logging()

# 需要禁用思考模式的平台域名及其对应参数（默认关闭思考模式）
THINKING_DISABLED_DOMAINS = {
    "aliyuncs.com": {"enable_thinking": False},
    "bigmodel.cn": {"thinking": {"type": "disabled"}},
    "moonshot.cn": {"thinking": {"type": "disabled"}},
    "volces.com": {"thinking": {"type": "disabled"}},
    "xiaomimimo.com": {"thinking": {"type": "disabled"}},
}


def _normalize_openai_base_url(base_url: str) -> str:
    if not base_url:
        return base_url
    base_url = base_url.rstrip("/")
    chat_completions_path = "/chat/completions"
    if base_url.endswith(chat_completions_path):
        return base_url[: -len(chat_completions_path)]
    return base_url


def _merge_dict(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key")
        if "base_url" in config:
            self.base_url = _normalize_openai_base_url(config.get("base_url"))
        else:
            self.base_url = _normalize_openai_base_url(config.get("url"))
        self.extra_body = (
            config.get("extra_body") if isinstance(config.get("extra_body"), dict) else {}
        )

        timeout_config = config.get("timeout")
        if isinstance(timeout_config, dict):
            # 细粒度超时配置
            custom_timeout = httpx.Timeout(
                pool=timeout_config.get("pool", 2.0),
                connect=timeout_config.get("connect", 3.0),
                write=timeout_config.get("write", 5.0),
                read=timeout_config.get("read", 60.0),
            )
        elif isinstance(timeout_config, (int, float)) and timeout_config > 0:
            # 兼容旧的单一超时配置（整数或浮点数）
            custom_timeout = httpx.Timeout(timeout_config)
        else:
            # 未配置或配置无效，使用默认值
            custom_timeout = httpx.Timeout(300)

        param_defaults = {
            "max_tokens": int,
            "temperature": lambda x: round(float(x), 1),
            "top_p": lambda x: round(float(x), 1),
            "frequency_penalty": lambda x: round(float(x), 1),
            "presence_penalty": lambda x: round(float(x), 1),
        }

        for param, converter in param_defaults.items():
            value = config.get(param)
            try:
                setattr(
                    self,
                    param,
                    converter(value) if value not in (None, "") else None,
                )
            except (ValueError, TypeError):
                setattr(self, param, None)

        logger.debug(
            f"意图识别参数初始化: {self.temperature}, {self.max_tokens}, {self.top_p}, {self.frequency_penalty}"
        )

        model_key_msg = check_model_key("LLM", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=custom_timeout
        )

    @staticmethod
    def normalize_dialogue(dialogue):
        """修复消息格式，并将 system 消息合并到最前面。"""
        system_contents = []
        normalized_dialogue = []
        for msg in dialogue:
            if "role" in msg and "content" not in msg:
                msg["content"] = ""
            if msg.get("role") == "system":
                content = msg.get("content")
                if content:
                    system_contents.append(str(content))
            else:
                normalized_dialogue.append(msg)

        if system_contents:
            normalized_dialogue.insert(
                0,
                {"role": "system", "content": "\n\n".join(system_contents)},
            )
        return normalized_dialogue

    def _apply_extra_body(self, request_params: dict):
        """应用平台默认和配置指定的额外请求参数。"""
        extra_body = {}
        parsed_url = urlparse(self.base_url)
        domain = parsed_url.netloc
        for disabled_domain, params in THINKING_DISABLED_DOMAINS.items():
            if disabled_domain in domain:
                _merge_dict(extra_body, params)
                logger.bind(tag=TAG).info(f"为域名 {domain} 禁用思考模式，参数: {params}")
                break
        _merge_dict(extra_body, self.extra_body)
        if extra_body:
            request_params["extra_body"] = extra_body

    def response(self, session_id, dialogue, **kwargs):
        dialogue = self.normalize_dialogue(dialogue)

        request_params = {
            "model": self.model_name,
            "messages": dialogue,
            "stream": True,
        }

        # 添加可选参数,只有当参数不为None时才添加
        optional_params = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
        }

        for key, value in optional_params.items():
            if value is not None:
                request_params[key] = value

        self._apply_extra_body(request_params)

        responses = self.client.chat.completions.create(**request_params)

        is_active = True
        _think_buffer = ""
        try:            
            for chunk in responses:
                try:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    content = getattr(delta, "content", "") if delta else ""
                except IndexError:
                    content = ""
                if content:
                    _think_buffer += content
                    while _think_buffer:
                        if not is_active:
                            end = _think_buffer.find("</think>")
                            if end == -1:
                                keep = max(0, len(_think_buffer) - 8)
                                _think_buffer = _think_buffer[keep:]
                                break
                            _think_buffer = _think_buffer[end + 8:]
                            is_active = True
                        else:
                            start = _think_buffer.find("<think>")
                            if start == -1:
                                safe = max(0, len(_think_buffer) - 7)
                                if safe:
                                    yield _think_buffer[:safe]
                                    _think_buffer = _think_buffer[safe:]
                                break
                            if start > 0:
                                yield _think_buffer[:start]
                            _think_buffer = _think_buffer[start + 7:]
                            is_active = False
        finally:
            if is_active and _think_buffer:
                yield _think_buffer
            responses.close()

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        dialogue = self.normalize_dialogue(dialogue)

        request_params = {
            "model": self.model_name,
            "messages": dialogue,
            "stream": True,
            "tools": functions,
        }

        optional_params = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
        }

        for key, value in optional_params.items():
            if value is not None:
                request_params[key] = value

        self._apply_extra_body(request_params)

        stream = self.client.chat.completions.create(**request_params)

        try:
            for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", "")
                    tool_calls = getattr(delta, "tool_calls", None)
                    yield content, tool_calls
                elif isinstance(getattr(chunk, "usage", None), CompletionUsage):
                    usage_info = getattr(chunk, "usage", None)
                    logger.bind(tag=TAG).info(
                        f"Token 消耗：输入 {getattr(usage_info, 'prompt_tokens', '未知')}，"
                        f"输出 {getattr(usage_info, 'completion_tokens', '未知')}，"
                        f"共计 {getattr(usage_info, 'total_tokens', '未知')}"
                    )
        finally:
            stream.close()
