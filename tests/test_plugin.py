import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from astrbot.api.provider import ProviderRequest
from astrbot.core.message.components import At, AtAll, Image, Plain, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata

from astrbot_plugin_fastbrain_proactive.client import (
    DecisionError,
    SystemOneClient,
    probability,
    retry_delay,
)
from astrbot_plugin_fastbrain_proactive.engine import (
    Rooms,
    Settings,
    build_payload,
    compact,
)
from astrbot_plugin_fastbrain_proactive.main import (
    DECISION,
    FastBrainProactive,
    SessionServiceManager,
)


@pytest.mark.asyncio
async def test_core_caption_reaches_decision_history_dataset_and_reply(
    plugin, tmp_path
):
    from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
    from astrbot_plugin_fastbrain_proactive.dataset import Dataset
    from astrbot_plugin_fastbrain_proactive.image_captions import CAPTIONS

    group = GroupChatContext(None, plugin.context)
    group.cfg = lambda e: {
        "image_caption": True,
        "image_caption_provider_id": "vision",
        "image_caption_prompt": "describe",
        "group_message_max_cnt": 10,
    }
    vision = AsyncMock(return_value="白板上写着：睦头，帮忙整理这份计划")
    group.get_image_caption = vision
    plugin.context.get_all_stars = lambda: [
        SimpleNamespace(
            name="astrbot", star_cls=SimpleNamespace(group_chat_context=group)
        )
    ]
    plugin.dataset = Dataset(tmp_path, 1)
    e = event("", parts=[Image(file="local-test")])
    await plugin.connect_image_captions(e)
    await group.handle_message(e)
    assert e.get_extra(CAPTIONS)["captions"]
    await plugin.collect(e)
    await plugin.judge(e)
    state = json.loads(plugin.client.decide.call_args.args[0]["state"])
    assert "白板" in state["target"]["text"]
    assert e.get_extra(DECISION) and e.is_at_or_wake_command
    vision.assert_awaited_once()
    req = ProviderRequest(prompt="image")
    await plugin.prepare_reply(e, req)
    assert "白板" in req.extra_user_content_parts[0].text
    rows = [
        json.loads(line)
        for file in (tmp_path / "data").glob("*.jsonl")
        for line in file.read_text().splitlines()
    ]
    inp = json.loads(rows[0]["input"])
    assert inp["media_content_available"] is True
    assert inp["target"][7] == 0 and "白板" in inp["target"][3]
    next_event = event("嗯", mid="next")
    await plugin.collect(next_event)
    await plugin.judge(next_event)
    state = json.loads(plugin.client.decide.call_args.args[0]["state"])
    assert "白板" in state["history"][0]["text"]
    await plugin.terminate()
    assert group.get_image_caption is vision


def event(text="官网写的是按年收费", mid="1", sender="user", group="g", parts=None):
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.self_id = "bot"
    message.group_id = group
    message.message_id = mid
    message.sender = MessageMember(sender, "User")
    message.message = parts if parts is not None else [Plain(text)]
    message.message_str = text
    message.raw_message = {}
    return AstrMessageEvent(
        text, message, PlatformMetadata("aiocqhttp", "test", "qq"), group
    )


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setattr(
        SessionServiceManager,
        "should_process_llm_request",
        AsyncMock(return_value=True),
    )
    config = {
        "provider_settings": {"enable": True, "wake_prefix": ""},
        "agent_runner": {"runner_type": "local"},
    }
    p = FastBrainProactive(
        SimpleNamespace(get_config=lambda umo: config),
        {"mode": "reply", "allowed_groups": ["qq:g"], "check_interval_seconds": 0},
    )
    p.client.decide = AsyncMock(return_value=0.9)
    p.clock = lambda: 1000.0
    return p


@pytest.mark.asyncio
async def test_positive_uses_native_session_and_temporary_user_content(plugin):
    e = event()
    original_umo = e.unified_msg_origin
    await plugin.collect(e)
    await plugin.judge(e)
    assert e.is_at_or_wake_command
    req = ProviderRequest(
        prompt="original",
        system_prompt="stable system",
        contexts=[{"role": "user", "content": "old"}],
    )
    await plugin.prepare_reply(e, req)
    await plugin.prepare_reply(e, req)
    assert e.unified_msg_origin == original_umo
    assert req.prompt == "original" and req.system_prompt == "stable system"
    assert req.contexts == [{"role": "user", "content": "old"}]
    assert len(req.extra_user_content_parts) == 1
    assert req.extra_user_content_parts[0].model_dump_for_context()["_no_save"] is True
    assert "官网写的是按年收费" in req.extra_user_content_parts[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "false",
        "observe",
        "unlisted",
        "passive",
        "image",
        "self",
        "disabled",
        "provider_prefix",
        "other_runner",
        "stopped",
        "sent",
        "veto",
        "result",
        "provider_request",
    ],
)
async def test_non_proactive_paths_do_not_wake(plugin, case):
    e = event()
    if case == "false":
        plugin.client.decide.return_value = 0.1
    if case == "observe":
        object.__setattr__(plugin.cfg, "mode", "observe")
    if case == "unlisted":
        e = event(group="elsewhere")
    if case == "passive":
        e.is_at_or_wake_command = True
    if case == "image":
        e = event("", parts=[Image(file="https://invalid/image")])
    if case == "self":
        e = event(sender="bot")
    if case == "disabled":
        plugin.context.get_config("")["provider_settings"]["enable"] = False
    if case == "provider_prefix":
        plugin.context.get_config("")["provider_settings"]["wake_prefix"] = "chat"
    if case == "other_runner":
        plugin.context.get_config("")["agent_runner"]["runner_type"] = "dify"
    if case == "stopped":
        e.stop_event()
    if case == "sent":
        e._has_send_oper = True
    if case == "veto":
        e.call_llm = True
    if case == "result":
        e.set_result(e.plain_result("handled"))
    if case == "provider_request":
        e.set_extra("provider_request", ProviderRequest())
    await plugin.collect(e)
    await plugin.judge(e)
    assert not e.get_extra(DECISION)
    assert e.is_at_or_wake_command == (case == "passive")
    assert not e.is_stopped() or case == "stopped"
    if case not in {"false", "observe"}:
        plugin.client.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_disabled_and_duplicate_do_not_infer(plugin, monkeypatch):
    monkeypatch.setattr(
        SessionServiceManager,
        "should_process_llm_request",
        AsyncMock(return_value=False),
    )
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    plugin.client.decide.assert_not_awaited()
    monkeypatch.setattr(
        SessionServiceManager,
        "should_process_llm_request",
        AsyncMock(return_value=True),
    )
    duplicate = event()
    await plugin.collect(duplicate)
    await plugin.judge(duplicate)
    plugin.client.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_burst_queues_each_snapshot_and_reserves_only_one_reply(plugin):
    entered, release = asyncio.Event(), asyncio.Event()
    bodies = []

    async def slow(payload):
        bodies.append(json.loads(payload["state"]))
        entered.set()
        await release.wait()
        return 0.9

    plugin.client.decide.side_effect = slow
    first = event("第一个邀请", mid="1")
    await plugin.collect(first)
    task = asyncio.create_task(plugin.judge(first))
    await entered.wait()
    second = event("第二个邀请", mid="2")
    await plugin.collect(second)
    task2 = asyncio.create_task(plugin.judge(second))
    await asyncio.sleep(0)
    assert not task2.done()
    release.set()
    await asyncio.gather(task, task2)
    assert first.is_at_or_wake_command and not second.is_at_or_wake_command
    assert plugin.client.decide.await_count == 2
    assert not plugin.decision_slot.locked()
    assert [x["target"]["text"] for x in bodies] == ["第一个邀请", "第二个邀请"]
    assert not bodies[0]["history"]
    assert bodies[1]["history"][-1]["text"] == "第一个邀请"
    assert plugin.rooms.items[plugin.key(first)].last["status"] == "reply_pending"


@pytest.mark.asyncio
async def test_new_message_before_generation_does_not_stop_request(plugin):
    first = event()
    await plugin.collect(first)
    await plugin.judge(first)
    await plugin.collect(
        event(mid="2", parts=[Reply(id="q", sender_id="other"), Plain("新消息")])
    )
    req = ProviderRequest()
    await plugin.prepare_reply(first, req)
    assert not first.is_stopped()
    assert len(req.extra_user_content_parts) == 1
    assert "新消息" not in req.extra_user_content_parts[0].text


@pytest.mark.asyncio
async def test_timeout_and_cancellation_release_global_slot(plugin):
    plugin.client.decide.side_effect = DecisionError("timeout")
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    assert not e.is_at_or_wake_command and not plugin.decision_slot.locked()
    assert plugin.blocked_until == 1060
    another = event(mid="2")
    await plugin.collect(another)
    await plugin.judge(another)
    assert plugin.client.decide.await_count == 1
    plugin.clock = lambda: 1100
    plugin.client.decide.side_effect = asyncio.CancelledError
    e = event(mid="3")
    await plugin.collect(e)
    with pytest.raises(asyncio.CancelledError):
        await plugin.judge(e)
    assert not plugin.decision_slot.locked()


@pytest.mark.asyncio
async def test_cooldown_pending_and_group_isolation(plugin):
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    e2 = event(mid="2")
    await plugin.collect(e2)
    await plugin.judge(e2)
    plugin.clock = lambda: 1150
    e3 = event(mid="3")
    await plugin.collect(e3)
    await plugin.judge(e3)
    assert (
        plugin.client.decide.await_count == 3
    )  # Pending reply blocks speech, not judging.
    assert not e2.get_extra(DECISION) and not e3.get_extra(DECISION)
    object.__setattr__(plugin.cfg, "groups", frozenset({"qq:g", "qq:g2"}))
    e4 = event(group="g2")
    await plugin.collect(e4)
    await plugin.judge(e4)
    assert plugin.client.decide.await_count == 4


@pytest.mark.asyncio
async def test_sent_reply_recorded_once(plugin):
    e = event()
    await plugin.collect(e)
    await plugin.remember_answer(e, None, SimpleNamespace(completion_text="bot answer"))
    room = plugin.rooms.items[plugin.key(e)]
    assert len(room.messages) == 1
    e._has_send_oper = True
    await plugin.after_sent(e)
    await plugin.after_sent(e)
    assert len(room.messages) == 2 and room.messages[-1][1] == "bot"


def test_budget_prunes_oldest_but_never_target_or_persona():
    cfg = Settings.read({"max_messages": 100, "max_request_bytes": 3072})
    rooms = Rooms(cfg, lambda: 1000)
    room = rooms.get(("p", "b", "g"))
    for i in range(50):
        room.add("u", f"{i}:" + "聊天内容" * 30, str(i), 1000, cfg)
    payload, state = build_payload(cfg, room, "b")
    assert len(compact(payload).encode()) <= 3072
    assert json.loads(state)["target"]["text"].startswith("49:")
    assert len(json.loads(state)["history"]) < 49
    object.__setattr__(cfg, "persona", "人格" * 2000)
    with pytest.raises(ValueError, match="input_budget"):
        build_payload(cfg, room, "b")


def test_ttl_lru_and_message_bounds():
    now = [1000]
    cfg = Settings.read({"max_groups": 2, "max_messages": 2, "context_ttl_seconds": 10})
    rooms = Rooms(cfg, lambda: now[0])
    a = rooms.get(("p", "b", "a"))
    for i in range(300):
        a.add("u", "a", str(i), now[0], cfg)
    assert len(a.messages) == 2 and len(a.seen) == 256
    rooms.get(("p", "b", "b"))
    rooms.get(("p", "b", "c"))
    assert len(rooms.items) == 2 and ("p", "b", "a") not in rooms.items
    now[0] += 11
    rooms.get(("p", "b", "d"))
    assert len(rooms.items) == 1


@pytest.mark.parametrize(
    "p,q",
    [
        (True, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        (-1, 2),
        (1.1, -0.1),
        (0.7, 0.7),
        ("0.1", 0.9),
        (None, 1),
    ],
)
def test_reject_invalid_probabilities(p, q):
    with pytest.raises(DecisionError):
        probability(
            {
                "model": "FastBrain-9B",
                "answers": {"should_reply": {"probabilities": {"True": p, "False": q}}},
            },
            "FastBrain-9B",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["ok", "unauthorized", "redirect", "large", "bad_json", "wrong_model", "timeout"],
)
@pytest.mark.parametrize("priority_enabled", [False, True])
async def test_real_http_client(scenario, unused_tcp_port, priority_enabled):
    received = []

    async def serve(request):
        received.append((request.headers.get("Authorization"), await request.json()))
        if scenario == "unauthorized":
            return web.Response(
                status=429, headers={"Retry-After": "120"}, text="sensitive error"
            )
        if scenario == "redirect":
            return web.Response(status=302, headers={"Location": "/elsewhere"})
        if scenario == "large":
            return web.Response(text="x" * 70000)
        if scenario == "bad_json":
            return web.Response(text="bad json")
        if scenario == "timeout":
            await asyncio.sleep(0.08)
        return web.json_response(
            {
                "model": "wrong" if scenario == "wrong_model" else "FastBrain-9B",
                "answers": {
                    "should_reply": {"probabilities": {"True": 0.3, "False": 0.7}}
                },
            }
        )

    app = web.Application()
    app.router.add_post("/systemone", serve)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    client = SystemOneClient(
        f"http://127.0.0.1:{unused_tcp_port}/systemone",
        "dummy-test-key",
        0.01 if scenario == "timeout" else 2,
        priority_enabled=priority_enabled,
    )
    try:
        if scenario == "ok":
            assert await client.decide({"model": "FastBrain-9B"}) == 0.3
        else:
            with pytest.raises(DecisionError) as exc:
                await client.decide({"model": "FastBrain-9B"})
            assert "sensitive" not in str(exc.value)
            if scenario == "unauthorized":
                assert exc.value.retry_after == 120
        assert len(received) == 1 and received[0][0] == "Bearer dummy-test-key"
    finally:
        await client.close()
        await runner.cleanup()


def test_retry_after_and_setting_validation():
    assert retry_delay("3600") == 3600
    assert retry_delay("garbage") == 60
    assert retry_delay("Wed, 01 Jan 2031 00:00:00 GMT") > 60
    with pytest.raises(ValueError):
        Settings.read({"threshold": float("nan")})
    with pytest.raises(ValueError):
        Settings.read({"endpoint": "https://user:password@host/path"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,expected",
    [
        ("positive", 1),
        ("named_invitation", 1),
        ("slash_plain", 1),
        ("bang_plain", 1),
        ("fullbang_plain", 1),
        ("human_at_positive", 1),
        ("human_quote_positive", 1),
        ("human_at_then_followup", 1),
        ("human_quote_then_followup", 1),
        ("negative", 0),
        ("passive", 1),
        ("whitelist_denied", 0),
        ("session_disabled", 0),
        ("plugin_disabled", 0),
        ("observe", 0),
        ("admin_command", 0),
        ("member_command", 0),
    ],
)
async def test_sdk_waking_access_and_process_pipeline(
    plugin, monkeypatch, case, expected
):
    """Real SDK dispatch/access stages; only the final generation is a spy."""
    import inspect
    from astrbot.core.pipeline.context import PipelineContext
    from astrbot.core.pipeline.context_utils import call_event_hook
    from astrbot.core.pipeline.process_stage.stage import ProcessStage
    from astrbot.core.pipeline.process_stage.method.star_request import (
        StarRequestSubStage,
    )
    from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
    from astrbot.core.pipeline.whitelist_check.stage import WhitelistCheckStage
    from astrbot.core.pipeline.session_status_check.stage import SessionStatusCheckStage
    from astrbot.core.star.star import star_map
    from astrbot.core.star.star_handler import EventType, star_handlers_registry
    from astrbot.core.star.session_plugin_manager import SessionPluginManager

    module = type(plugin).__module__
    metadata = star_map[module]
    monkeypatch.setattr(metadata, "name", "astrbot_plugin_fastbrain_proactive")
    monkeypatch.setattr(metadata, "star_cls", plugin)
    for handler in star_handlers_registry.get_handlers_by_event_type(
        EventType.AdapterMessageEvent
    ) + star_handlers_registry.get_handlers_by_event_type(EventType.OnLLMRequestEvent):
        if handler.handler_module_path == module:
            monkeypatch.setattr(
                handler, "handler", getattr(plugin, handler.handler_name)
            )

    async def handlers_for_session(e, handlers):
        return [] if case == "plugin_disabled" else handlers

    monkeypatch.setattr(
        SessionPluginManager, "filter_handlers_by_session", handlers_for_session
    )
    monkeypatch.setattr(
        SessionServiceManager,
        "is_session_enabled",
        AsyncMock(return_value=case != "session_disabled"),
    )
    config = plugin.context.get_config("")
    config.update(
        {
            "wake_prefix": ["/"],
            "admins_id": ["admin"],
            "plugin_set": [metadata.name],
            "platform_settings": {
                "enable_id_white_list": True,
                "id_whitelist": ["different" if case == "whitelist_denied" else "g"],
                "wl_ignore_admin_on_group": False,
                "wl_ignore_admin_on_friend": False,
                "id_whitelist_log": False,
                "no_permission_reply": False,
            },
        }
    )
    config["provider_settings"].update({"prompt_prefix": "", "identifier": False})
    cm = SimpleNamespace(get_curr_conversation_id=AsyncMock(return_value="cid"))
    ctx = PipelineContext(
        config,
        SimpleNamespace(context=SimpleNamespace(conversation_manager=cm)),
        "test",
    )
    e = event()
    if case in {"slash_plain", "bang_plain", "fullbang_plain"}:
        config["wake_prefix"] = ["睦头"]
        prefix = {"slash_plain": "/", "bang_plain": "!", "fullbang_plain": "！"}[case]
        e = event(prefix + "机器人也来聊聊")
    if case == "named_invitation":
        e = event("睦头你也来聊聊")
    if case == "human_at_positive":
        e = event(parts=[At(qq="friend"), Plain("也请机器人聊聊")])
    if case == "human_quote_positive":
        e = event(
            parts=[
                Reply(id="q", sender_id="friend", message_str="机器人也来聊聊"),
                Plain("同意"),
            ]
        )
    if case == "negative":
        plugin.client.decide.return_value = 0.01
    if case == "passive":
        e = event(parts=[At(qq="bot"), Plain("请回答")])
    if case == "observe":
        object.__setattr__(plugin.cfg, "mode", "observe")
    if case in {"admin_command", "member_command"}:
        e = event(
            "/主动回复状态", sender="admin" if case == "admin_command" else "user"
        )
    # No real send operation; command output is captured at pipeline yield points.
    generated, outputs = [], []

    class AgentSpy:
        async def process(self, ev):
            req = ProviderRequest(prompt=ev.message_str, system_prompt="stable")
            stopped = await call_event_hook(ev, EventType.OnLLMRequestEvent, req)
            if not stopped:
                generated.append(req)
            if False:
                yield

    incoming = [e]
    if case in {"human_at_then_followup", "human_quote_then_followup"}:
        part = (
            At(qq="other")
            if case == "human_at_then_followup"
            else Reply(id="q", sender_id="other")
        )
        incoming.insert(0, event(mid="human", parts=[part, Plain("拼豆仙人")]))
        plugin.client.decide.side_effect = [0.1, 0.9]
    for current in incoming:
        e = current
        stages = [WakingCheckStage(), WhitelistCheckStage(), SessionStatusCheckStage()]
        for stage in stages:
            await stage.initialize(ctx)
            result = stage.process(e)
            if inspect.isasyncgen(result):
                async for _ in result:
                    pass
            else:
                await result
            if e.is_stopped():
                break
        if not e.is_stopped():
            process = ProcessStage()
            process.ctx = ctx
            process.star_request_sub_stage = StarRequestSubStage()
            await process.star_request_sub_stage.initialize(ctx)
            process.agent_sub_stage = AgentSpy()
            async for _ in process.process(e):
                if e.get_result():
                    outputs.append(e.get_result())
    assert len(generated) == expected
    if case in {
        "positive",
        "named_invitation",
        "human_at_positive",
        "human_quote_positive",
    }:
        assert len(generated[0].extra_user_content_parts) == 1
    if case == "passive":
        plugin.client.decide.assert_not_awaited()
        assert not generated[0].extra_user_content_parts
    if case == "admin_command":
        assert outputs
    if case == "member_command":
        assert not outputs
    if case in {"whitelist_denied", "session_disabled", "plugin_disabled"}:
        plugin.client.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_hooks_do_not_collect_without_enabled_adapter_handler(plugin):
    e = event()
    e._has_send_oper = True
    await plugin.prepare_reply(e, ProviderRequest())
    await plugin.remember_answer(
        e, None, SimpleNamespace(completion_text="private answer")
    )
    await plugin.after_sent(e)
    assert not plugin.rooms.items


@pytest.mark.asyncio
async def test_clear_invalidates_pending_and_does_not_repopulate(plugin):
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    clear_event = event("/主动回复清空", mid="clear")
    await plugin.collect(clear_event)
    async for response in plugin.clear(clear_event):
        clear_event.set_result(response)
    clear_event._has_send_oper = True
    await plugin.after_sent(clear_event)
    assert not plugin.rooms.items
    await plugin.prepare_reply(e, ProviderRequest())
    assert e.is_stopped()


def test_extended_budget_keeps_long_target_intact_and_retains_more_context():
    short = Settings.read({"max_request_bytes": 3072})
    long = Settings.read({})
    assert long.max_bytes == 7168 and long.max_messages == 80
    room = Rooms(long, lambda: 1000).get(("p", "b", "g"))
    for i in range(50):
        room.add("user", str(i) + "历史聊天内容" * 15, str(i), 1000, long)
    _, short_state = build_payload(short, room, "b")
    payload, long_state = build_payload(long, room, "b")
    assert len(json.loads(long_state)["history"]) > len(
        json.loads(short_state)["history"]
    )
    assert len(compact(payload).encode()) <= 7168
    target = "x" * 4000
    room.add("user", target, "long-target", 1000, long)
    _, state = build_payload(long, room, "b")
    assert json.loads(state)["target"]["text"] == target
    with pytest.raises(ValueError):
        build_payload(short, room, "b")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "part",
    [At(qq="other"), Reply(id="quote", sender_id="other"), Reply(id="quote"), AtAll()],
)
async def test_human_directed_messages_do_not_start_group_cooldown(plugin, part):
    plugin.client.decide.side_effect = [0.1, 0.9]
    human = event(mid="human", parts=[part, Plain("拼豆仙人")])
    await plugin.collect(human)
    await plugin.judge(human)
    room = plugin.rooms.items[plugin.key(human)]
    assert room.cooldown_until == 0
    plugin.client.decide.assert_awaited_once()
    plugin.clock = lambda: 1008
    followup = event("我有个小模型，如果生效了你就插话一下", mid="followup")
    await plugin.collect(followup)
    await plugin.judge(followup)
    assert plugin.client.decide.await_count == 2
    assert followup.get_extra(DECISION)


@pytest.mark.asyncio
@pytest.mark.parametrize("part", [At(qq="bot"), Reply(id="quote", sender_id="bot")])
async def test_addressed_to_bot_still_starts_cooldown(plugin, part):
    e = event(parts=[part, Plain("请回答")])
    e.is_at_or_wake_command = True
    await plugin.collect(e)
    await plugin.judge(e)
    assert plugin.rooms.items[plugin.key(e)].cooldown_until == 1030
    plugin.client.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_cooldown_expires_despite_repeated_human_quotes(plugin):
    e = event("睦头你好")
    e.is_at_or_wake_command = True
    await plugin.collect(e)
    await plugin.prepare_reply(e, ProviderRequest())
    plugin.clock = lambda: 1005
    e._has_send_oper = True
    await plugin.remember_answer(e, None, SimpleNamespace(completion_text="我在"))
    room = plugin.rooms.items[plugin.key(e)]
    assert room.cooldown_until == 1035 and room.pending_until == 0
    for tick in (1010, 1020, 1034):
        plugin.clock = lambda tick=tick: tick
        other = event(
            mid=str(tick),
            parts=[Reply(id="old", sender_id="human"), At(qq="human"), Plain("聊几句")],
        )
        await plugin.collect(other)
        await plugin.judge(other)
        assert room.cooldown_until == 1035
    ordinary = event(mid="before-expiry")
    await plugin.collect(ordinary)
    await plugin.judge(ordinary)
    assert plugin.client.decide.await_count == 4
    assert room.last["status"] == "reply_cooldown"
    assert not ordinary.get_extra(DECISION)
    plugin.clock = lambda: 1035
    ready = event(mid="expired")
    await plugin.collect(ready)
    await plugin.judge(ready)
    assert plugin.client.decide.await_count == 5
    assert ready.get_extra(DECISION)


def test_cooldown_default_and_existing_custom_values():
    assert Settings.read({}).cooldown == 30
    assert Settings.read({"cooldown_seconds": 120}).cooldown == 120


@pytest.mark.asyncio
async def test_unhandled_named_invitation_reaches_classifier(plugin):
    e = event("睦头帮我看看")
    await plugin.collect(e)
    await plugin.judge(e)
    plugin.client.decide.assert_awaited_once()
    assert e.get_extra(DECISION)
    payload = plugin.client.decide.await_args.args[0]
    assert json.loads(payload["state"])["bot_names"] == ["睦头", "木头"]


@pytest.mark.asyncio
async def test_core_woken_named_invitation_is_not_handled_twice(plugin):
    e = event("睦头帮我看看")
    e.is_at_or_wake_command = True
    await plugin.collect(e)
    await plugin.judge(e)
    plugin.client.decide.assert_not_awaited()
    assert not e.get_extra(DECISION)


def test_fewshot_budget_preserves_target_with_bounded_history():
    cfg = Settings.read({"fewshot": True})
    room = Rooms(cfg, lambda: 1000).get(("p", "b", "g"))
    for i in range(80):
        room.add("u", f"{i}:" + "历史聊天内容" * 20, str(i), 1000, cfg)
    room.add("u", "机器人也来发表下意见吧", "target", 1000, cfg)
    payload, state = build_payload(cfg, room, "b")
    parsed = json.loads(state)
    assert len(compact(payload).encode()) <= cfg.max_bytes
    assert parsed["target"]["text"] == "机器人也来发表下意见吧"
    assert 0 < len(parsed["history"]) < 79
    assert parsed["persona"] == cfg.persona


@pytest.mark.asyncio
async def test_quote_and_mentions_keep_sender_relationships(plugin):
    quoted = event(
        "邀请",
        parts=[
            At(qq="friend"),
            Reply(
                id="q",
                sender_id="friend",
                chain=[Plain("睦头也来聊聊"), Image(file="https://invalid")],
            ),
            Plain("你说得对，机器人来吧"),
        ],
    )
    await plugin.collect(quoted)
    await plugin.judge(quoted)
    state = json.loads(plugin.client.decide.await_args.args[0]["state"])
    target = state["target"]
    assert target["mentions"] == [target["reply_to"]["speaker"]]
    assert target["speaker"] != target["reply_to"]["speaker"]
    assert target["reply_to"]["text"] == "睦头也来聊聊"
    assert target["text"] == "你说得对，机器人来吧"
    assert "https://invalid" not in json.dumps(state)
    assert quoted.get_extra(DECISION)


@pytest.mark.asyncio
async def test_unhandled_bot_quote_is_judged_and_unknown_quote_preserved(plugin):
    e = event(
        parts=[
            Reply(id="q", sender_id="bot", message_str="有人来聊聊吗"),
            Plain("我来了"),
        ]
    )
    await plugin.collect(e)
    await plugin.judge(e)
    state = json.loads(plugin.client.decide.await_args.args[0]["state"])
    assert state["target"]["reply_to"]["speaker"] == "bot"
    assert e.get_extra(DECISION)


@pytest.mark.asyncio
async def test_zero_interval_even_if_legacy_config_was_ten(plugin):
    plugin.client.decide.return_value = 0.1
    for i in range(4):
        e = event(mid=str(i))
        await plugin.collect(e)
        await plugin.judge(e)
    assert plugin.client.decide.await_count == 4
    # Duplicate dispatch of one event cannot classify or wake a second time.
    await plugin.judge(e)
    assert plugin.client.decide.await_count == 4


@pytest.mark.asyncio
async def test_clear_while_queued_cancels_only_old_room(plugin):
    await plugin.decision_slot.acquire()
    e = event()
    await plugin.collect(e)
    task = asyncio.create_task(plugin.judge(e))
    await asyncio.sleep(0)
    plugin.rooms.items.pop(plugin.key(e))
    plugin.decision_slot.release()
    await task
    plugin.client.decide.assert_not_awaited()
    assert not e.get_extra(DECISION)


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_release_another_requests_slot(plugin):
    await plugin.decision_slot.acquire()
    e = event()
    await plugin.collect(e)
    task = asyncio.create_task(plugin.judge(e))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert plugin.decision_slot.locked()
    plugin.decision_slot.release()
    another = event(mid="2")
    await plugin.collect(another)
    await plugin.judge(another)
    assert another.get_extra(DECISION)


@pytest.mark.asyncio
async def test_expired_snapshot_and_backoff_do_not_call_model(plugin):
    e = event()
    await plugin.collect(e)
    plugin.clock = lambda: 1031
    await plugin.judge(e)
    plugin.client.decide.assert_not_awaited()
    assert plugin.rooms.items[plugin.key(e)].last["status"] == "decision_expired"


@pytest.mark.asyncio
async def test_passive_reply_takes_reservation_before_proactive_generation(plugin):
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    passive = event(mid="p")
    passive.is_at_or_wake_command = True
    await plugin.collect(passive)
    await plugin.prepare_reply(passive, ProviderRequest())
    await plugin.prepare_reply(e, ProviderRequest())
    assert e.is_stopped()
    assert not passive.is_stopped()


@pytest.mark.asyncio
async def test_image_arriving_during_judgment_keeps_original_target(plugin):
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(payload):
        entered.set()
        await release.wait()
        return 0.95

    plugin.client.decide.side_effect = slow
    first = event("睦头也来聊聊", mid="a")
    await plugin.collect(first)
    task = asyncio.create_task(plugin.judge(first))
    await entered.wait()
    image = event("", mid="b", parts=[Image(file="https://invalid")])
    await plugin.collect(image)
    await plugin.judge(image)
    release.set()
    await task
    assert first.get_extra(DECISION)
    assert plugin.client.decide.await_count == 1
    assert (
        json.loads(first.get_extra(DECISION)["state"])["target"]["text"]
        == "睦头也来聊聊"
    )


@pytest.mark.asyncio
async def test_queue_timeout_is_logged_without_releasing_owned_slot(
    plugin, monkeypatch
):
    import importlib

    module = importlib.import_module(type(plugin).__module__)
    monkeypatch.setattr(module, "MAX_DECISION_AGE", 0.02)
    await plugin.decision_slot.acquire()
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    assert plugin.rooms.items[plugin.key(e)].last["status"] == "queue_timeout"
    assert plugin.decision_slot.locked()
    plugin.decision_slot.release()
    plugin.client.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_quote_and_atall_preserve_available_context(plugin):
    e = event(
        parts=[
            AtAll(),
            Reply(id="q", message_str="某人建议机器人也来参与"),
            Plain("你怎么看"),
        ]
    )
    await plugin.collect(e)
    await plugin.judge(e)
    target = json.loads(plugin.client.decide.await_args.args[0]["state"])["target"]
    assert target["mentions"] == ["all"]
    assert target["reply_to"]["speaker"] == "unknown"
    assert target["reply_to"]["text"] == "某人建议机器人也来参与"


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/", "!", "！"])
async def test_unhandled_prefix_reaches_fastbrain_intact(plugin, prefix):
    e = event(prefix + "机器人也来聊聊")
    await plugin.collect(e)
    await plugin.judge(e)
    plugin.client.decide.assert_awaited_once()
    target = json.loads(plugin.client.decide.await_args.args[0]["state"])["target"]
    assert target["text"] == prefix + "机器人也来聊聊"
    assert e.get_extra(DECISION)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probability,action", [(0.9, "reply_cooldown"), (0.1, "silent")]
)
async def test_dataset_captures_model_not_actual_speech(
    plugin, tmp_path, probability, action
):
    from astrbot_plugin_fastbrain_proactive.dataset import Dataset

    plugin.dataset = Dataset(tmp_path, 1)
    plugin.client.decide.return_value = probability
    e = event("木头查下明天的安排")
    await plugin.collect(e)
    plugin.rooms.get(plugin.key(e)).cooldown_until = 1100
    await plugin.judge(e)
    with plugin.dataset.connect() as c:
        training, audit = c.execute("SELECT training,audit FROM samples").fetchone()
    assert json.loads(training)["label"] == (probability >= 0.5)
    assert json.loads(audit)["action"] == action
    assert not e.is_at_or_wake_command


@pytest.mark.asyncio
async def test_dataset_failure_does_not_fail_chat_or_backoff(
    plugin, tmp_path, monkeypatch
):
    from astrbot_plugin_fastbrain_proactive.dataset import Dataset

    plugin.dataset = Dataset(tmp_path, 1)

    def fail(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plugin.dataset, "record", fail)
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    assert e.is_at_or_wake_command and plugin.blocked_until == 0
    assert plugin.dataset_error == "OSError"


@pytest.mark.asyncio
async def test_failed_model_is_not_false_label(plugin, tmp_path):
    from astrbot_plugin_fastbrain_proactive.dataset import Dataset

    plugin.dataset = Dataset(tmp_path, 1)
    plugin.client.decide.side_effect = DecisionError("unavailable", 60)
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    with plugin.dataset.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM observed").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_expired_model_result_still_saved(plugin, tmp_path):
    from astrbot_plugin_fastbrain_proactive.dataset import Dataset

    plugin.dataset = Dataset(tmp_path, 1)

    async def decide(payload):
        plugin.clock = lambda: 1031.0
        return 0.9

    plugin.client.decide = decide
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    with plugin.dataset.connect() as c:
        audit = json.loads(c.execute("SELECT audit FROM samples").fetchone()[0])
    assert audit["action"] == "decision_expired" and audit["decision"] is True
    assert not e.is_at_or_wake_command


@pytest.mark.asyncio
async def test_dataset_outcome_once_and_private_context(plugin, tmp_path):
    from astrbot_plugin_fastbrain_proactive.dataset import Dataset

    plugin.dataset = Dataset(tmp_path, 1)
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    await plugin.prepare_reply(e, ProviderRequest())
    await plugin.prepare_reply(e, ProviderRequest())
    e._has_send_oper = True
    await plugin.after_sent(e)
    await plugin.after_sent(e)
    lines = [
        json.loads(line)
        for f in (tmp_path / "audit").glob("outcomes-*.jsonl")
        for line in f.read_text().splitlines()
    ]
    assert [x["phase"] for x in lines] == ["generation_started", "send_hook_observed"]
