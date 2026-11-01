import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.core.message.components import Image

from astrbot_plugin_fastbrain_proactive.image_captions import (
    CAPTIONS,
    MAX_CAPTION_BYTES,
    ImageCaptionBridge,
    caption_text,
)
from test_plugin import event


def core():
    group = GroupChatContext(None, SimpleNamespace())
    group.cfg = lambda e: {
        "image_caption": True,
        "image_caption_provider_id": "vision",
        "image_caption_prompt": "describe",
        "group_message_max_cnt": 10,
    }
    group.get_image_caption = AsyncMock(
        side_effect=lambda url, *args: "description " + url
    )
    return group


@pytest.mark.asyncio
async def test_core_results_reused_once_and_unload_restores():
    group = core()
    original = group.get_image_caption
    bridge = ImageCaptionBridge(lambda e: True)
    bridge.attach(group)
    bridge.attach(group)
    e = event("", parts=[Image(file="first"), Image(file="second")])
    await group.handle_message(e)
    capture = e.get_extra(CAPTIONS)
    assert capture["captions"] == ["description first", "description second"]
    assert original.await_count == 2
    assert all(
        s in group.raw_records[e.unified_msg_origin][-1] for s in capture["captions"]
    )
    assert "description first" in caption_text(capture)
    bridge.close()
    assert group.get_image_caption is original
    assert "_format_message" not in vars(group)


@pytest.mark.asyncio
async def test_parallel_groups_never_exchange_captions():
    group = core()

    async def describe(url, *args):
        await asyncio.sleep(0.02 if url == "slow" else 0)
        return url

    group.get_image_caption = describe
    bridge = ImageCaptionBridge(lambda e: True)
    bridge.attach(group)
    a = event(group="a", parts=[Image(file="slow")])
    b = event(group="b", parts=[Image(file="fast")])
    await asyncio.gather(group.handle_message(a), group.handle_message(b))
    assert a.get_extra(CAPTIONS)["captions"] == ["slow"]
    assert b.get_extra(CAPTIONS)["captions"] == ["fast"]
    bridge.close()


@pytest.mark.asyncio
async def test_failure_disabled_and_foreign_group_do_not_invent_descriptions():
    group = core()
    group.get_image_caption = AsyncMock(side_effect=[RuntimeError("offline"), "ok"])
    bridge = ImageCaptionBridge(lambda e: e.get_group_id() == "g")
    bridge.attach(group)
    a = event(parts=[Image(file="bad"), Image(file="good")])
    await group.handle_message(a)
    assert a.get_extra(CAPTIONS)["captions"] == ["ok"]
    assert a.get_extra(CAPTIONS)["failed"] == 1
    b = event(group="foreign", parts=[])
    await group.handle_message(b)
    assert b.get_extra(CAPTIONS) is None
    c = event(parts=[Image(file="disabled")])
    await group._format_message(c, {"image_caption": False})
    assert c.get_extra(CAPTIONS) is None
    bridge.close()


@pytest.mark.asyncio
async def test_caption_budget_keeps_full_core_result_and_marks_truncation():
    group = core()
    full = "图片文字" * 3000
    original = AsyncMock(return_value=full)
    group.get_image_caption = original
    bridge = ImageCaptionBridge(lambda e: True)
    bridge.attach(group)
    e = event(parts=[Image(file="a"), Image(file="b")])
    await group.handle_message(e)
    capture = e.get_extra(CAPTIONS)
    assert original.await_count == 2
    assert capture["bytes"] <= MAX_CAPTION_BYTES
    assert capture["truncated"]
    assert "已截短" in caption_text(capture)
    assert full in group.raw_records[e.unified_msg_origin][-1]
    bridge.close()


@pytest.mark.asyncio
async def test_cancelled_capture_does_not_leak_to_next_event():
    group = core()
    group.get_image_caption = AsyncMock(side_effect=[asyncio.CancelledError(), "next"])
    bridge = ImageCaptionBridge(lambda e: True)
    bridge.attach(group)
    with pytest.raises(asyncio.CancelledError):
        await group.handle_message(event(parts=[Image(file="cancel")]))
    e = event(parts=[Image(file="next")])
    await group.handle_message(e)
    assert e.get_extra(CAPTIONS)["captions"] == ["next"]
    assert bridge.current.get() is None
    bridge.close()
