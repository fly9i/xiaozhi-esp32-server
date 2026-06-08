"""通用定时动作序列执行器。

按时间顺序依次执行一串「任意已注册工具」的调用，并控制每个步骤执行后保持多久，
用于满足「每隔 N 秒做一件事 / 依次轮流展示多个效果 / 先…保持几秒…再…」这类带节奏的多步操作。

设计要点：
- 步骤里的工具可以是任意已注册工具（灯带效果、屏幕背景色、表情颜色、播放音乐等），不局限于灯带；
- 注册函数本身是同步的：起一个后台协程在 conn.loop 上跑，立刻返回，不阻塞对话；
- 可被打断：每步前、保持期间都检查 conn.client_abort，用户说话/打断即提前结束；
- 上限保护：限制步数、单步时长和总时长，避免把连接长时间占住；
- 黑名单：禁止序列里再调自身，以及退出/重启类工具。
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any

from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

# 健壮性上限
MAX_STEPS = 20           # 最多步数
MAX_HOLD_SEC = 30.0      # 单步最长保持时长（秒）
MAX_TOTAL_SEC = 120.0    # 整个序列最长总时长（秒）
_SLEEP_CHUNK_SEC = 0.2   # 保持期间的分片粒度，用于及时响应打断

# 禁止编入序列的工具（防递归 / 防误触发危险动作）
_BLOCKED_TOOLS = {"run_timed_sequence", "handle_exit_intent"}
_BLOCKED_KEYWORDS = ("reboot", "exit", "shutdown", "restart")

run_timed_sequence_function_desc = {
    "type": "function",
    "function": {
        "name": "run_timed_sequence",
        "description": (
            "按时间顺序依次执行一串设备动作，并控制每个动作执行后保持多久。"
            "当用户要求“每隔 N 秒做一件事”、“依次/轮流展示多个效果”、“先…保持几秒…再…”这类"
            "带时间节奏的多步操作时调用。每个步骤可以是任意一个已注册的工具，"
            "例如灯带效果(self_led_strip_*)、屏幕背景色、表情颜色、播放音乐等。"
            "不需要节奏、只是一次性切换一个效果时，直接调用对应工具即可，不要用本工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "要依次执行的步骤列表，严格按数组顺序执行",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "string",
                                "description": "该步骤要调用的工具名，如 self_led_strip_breath、self_screen_set_background_color",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "传给该工具的参数对象，没有参数时传 {}",
                            },
                            "hold_sec": {
                                "type": "number",
                                "description": "执行该步骤后保持多少秒再进入下一步；0 表示立刻进入下一步",
                            },
                        },
                        "required": ["tool"],
                    },
                }
            },
            "required": ["steps"],
        },
    },
}


def _normalize_steps(
    conn: "ConnectionHandler", steps: list
) -> list[dict[str, Any]]:
    """校验并裁剪步骤：过滤非法/受限/未注册工具，规整参数与时长，施加总时长上限。"""
    normalized: list[dict[str, Any]] = []
    total = 0.0
    for raw in steps:
        if len(normalized) >= MAX_STEPS:
            logger.bind(tag=TAG).warning(f"序列步数超过上限 {MAX_STEPS}，截断")
            break
        if not isinstance(raw, dict):
            continue

        tool = raw.get("tool") or raw.get("name")
        if not isinstance(tool, str) or not tool.strip():
            continue
        tool = tool.strip()

        if tool in _BLOCKED_TOOLS or any(k in tool.lower() for k in _BLOCKED_KEYWORDS):
            logger.bind(tag=TAG).warning(f"跳过受限工具: {tool}")
            continue
        if not conn.func_handler or not conn.func_handler.has_tool(tool):
            logger.bind(tag=TAG).warning(f"跳过未注册工具: {tool}")
            continue

        args = raw.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}

        try:
            hold = float(raw.get("hold_sec", 0))
        except (TypeError, ValueError):
            hold = 0.0
        hold = max(0.0, min(hold, MAX_HOLD_SEC))
        # 总时长保护：超出后把剩余步骤的保持时长压到 0
        if total + hold > MAX_TOTAL_SEC:
            hold = max(0.0, MAX_TOTAL_SEC - total)
        total += hold

        normalized.append({"tool": tool, "arguments": args, "hold_sec": hold})

    return normalized


def _cancel_running_sequence(conn: "ConnectionHandler") -> None:
    """取消上一个仍在运行的序列，保证同一时间只有一个序列在跑。"""
    task = getattr(conn, "_timed_sequence_task", None)
    if task is not None and not task.done():
        task.cancel()
    conn._timed_sequence_task = None


async def _run_sequence(
    conn: "ConnectionHandler", steps: list[dict[str, Any]]
) -> None:
    """后台协程：逐步派发工具调用，并在步骤之间按 hold_sec 等待。"""
    try:
        last_index = len(steps) - 1
        for idx, step in enumerate(steps):
            if getattr(conn, "client_abort", False):
                logger.bind(tag=TAG).info("定时序列被打断，提前结束")
                break

            tool = step["tool"]
            args = step["arguments"]
            try:
                result = await conn.func_handler.tool_manager.execute_tool(tool, args)
                logger.bind(tag=TAG).info(
                    f"定时序列步骤 {idx + 1}/{len(steps)}: {tool} -> "
                    f"{getattr(getattr(result, 'action', None), 'name', None)}"
                )
            except Exception as e:
                logger.bind(tag=TAG).error(f"定时序列步骤 {tool} 执行失败: {e}")

            # 最后一步无需再等待
            if idx >= last_index:
                break

            slept = 0.0
            hold = step["hold_sec"]
            while slept < hold:
                if getattr(conn, "client_abort", False):
                    logger.bind(tag=TAG).info("定时序列被打断，提前结束")
                    return
                chunk = min(_SLEEP_CHUNK_SEC, hold - slept)
                await asyncio.sleep(chunk)
                slept += chunk
    except asyncio.CancelledError:
        logger.bind(tag=TAG).info("定时序列任务被取消")
        raise
    finally:
        if getattr(conn, "_timed_sequence_task", None) is asyncio.current_task():
            conn._timed_sequence_task = None


@register_function(
    "run_timed_sequence", run_timed_sequence_function_desc, ToolType.SYSTEM_CTL
)
def run_timed_sequence(
    conn: "ConnectionHandler", steps: list = None, **kwargs
) -> ActionResponse:
    """按时间节奏依次执行一串工具调用（后台运行，立即返回）。"""
    if not isinstance(steps, list) or not steps:
        return ActionResponse(
            action=Action.RESPONSE,
            result="序列为空",
            response="你想让我依次做哪些动作呢？",
        )

    if conn.loop is None or not conn.loop.is_running():
        logger.bind(tag=TAG).error("事件循环未运行，无法启动定时序列")
        return ActionResponse(
            action=Action.ERROR,
            result="事件循环不可用",
            response="现在没法执行这个序列，请稍后再试。",
        )

    normalized = _normalize_steps(conn, steps)
    if not normalized:
        return ActionResponse(
            action=Action.RESPONSE,
            result="无有效步骤",
            response="这些动作我都没找到对应的工具，换个说法再试试？",
        )

    # 同一时间只允许一个序列运行，先取消上一个
    _cancel_running_sequence(conn)

    task = conn.loop.create_task(_run_sequence(conn, normalized))
    conn._timed_sequence_task = task

    total = round(sum(step["hold_sec"] for step in normalized))
    logger.bind(tag=TAG).info(
        f"启动定时序列: {len(normalized)} 步, 预计约 {total} 秒"
    )
    return ActionResponse(
        action=Action.RESPONSE,
        result=f"已开始按节奏执行 {len(normalized)} 个步骤，预计约 {total} 秒",
        response="好的，开始啦～",
    )
