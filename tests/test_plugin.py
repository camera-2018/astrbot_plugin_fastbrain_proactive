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
        "other_at",
        "image",
        "self",
        "command",
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
    if case == "other_at":
        e = event(parts=[At(qq="other"), Plain("hi")])
    if case == "image":
        e = event("", parts=[Image(file="https://invalid/image")])
    if case == "self":
        e = event(sender="bot")
    if case == "command":
        e = event("/unknown")
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
async def test_new_message_invalidates_inflight_and_global_busy_drops(plugin):
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(payload):
        entered.set()
        await release.wait()
        return 0.9

    plugin.client.decide.side_effect = slow
    first = event(mid="1")
    await plugin.collect(first)
    task = asyncio.create_task(plugin.judge(first))
    await entered.wait()
    second = event(mid="2")
    await plugin.collect(second)
    await plugin.judge(second)
    release.set()
    await task
    assert not first.is_at_or_wake_command and not second.is_at_or_wake_command
    assert plugin.client.decide.await_count == 1 and not plugin.busy


@pytest.mark.asyncio
async def test_new_message_before_generation_stops_only_own_request(plugin):
    first = event()
    await plugin.collect(first)
    await plugin.judge(first)
    await plugin.collect(event(mid="2"))
    await plugin.prepare_reply(first, ProviderRequest())
    assert first.is_stopped()


@pytest.mark.asyncio
async def test_timeout_and_cancellation_release_global_slot(plugin):
    plugin.client.decide.side_effect = DecisionError("timeout")
    e = event()
    await plugin.collect(e)
    await plugin.judge(e)
    assert not e.is_at_or_wake_command and not plugin.busy
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
    assert not plugin.busy


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
    assert plugin.client.decide.await_count == 1  # Generation reservation lasts 300 s.
    object.__setattr__(plugin.cfg, "groups", frozenset({"qq:g", "qq:g2"}))
    e4 = event(group="g2")
    await plugin.collect(e4)
    await plugin.judge(e4)
    assert plugin.client.decide.await_count == 2


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
async def test_real_http_client(scenario, unused_tcp_port):
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
    if case == "named_invitation":
        e = event("睦头你也来聊聊")
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
    if case in {"positive", "named_invitation"}:
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
    human = event(mid="human", parts=[part, Plain("拼豆仙人")])
    await plugin.collect(human)
    await plugin.judge(human)
    room = plugin.rooms.items[plugin.key(human)]
    assert room.cooldown_until == 0
    plugin.client.decide.assert_not_awaited()
    plugin.clock = lambda: 1008
    followup = event("我有个小模型，如果生效了你就插话一下", mid="followup")
    await plugin.collect(followup)
    await plugin.judge(followup)
    plugin.client.decide.assert_awaited_once()
    assert followup.get_extra(DECISION)


@pytest.mark.asyncio
@pytest.mark.parametrize("part", [At(qq="bot"), Reply(id="quote", sender_id="bot")])
async def test_addressed_to_bot_still_starts_cooldown(plugin, part):
    e = event(parts=[part, Plain("请回答")])
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
    plugin.client.decide.assert_not_awaited()
    assert room.last["status"] == "cooldown" and room.last["remaining_seconds"] == 1
    plugin.clock = lambda: 1035
    ready = event(mid="expired")
    await plugin.collect(ready)
    await plugin.judge(ready)
    plugin.client.decide.assert_awaited_once()


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
    assert json.loads(payload["state"])["bot_names"] == ["睦头"]


@pytest.mark.asyncio
async def test_core_woken_named_invitation_is_not_handled_twice(plugin):
    e = event("睦头帮我看看")
    e.is_at_or_wake_command = True
    await plugin.collect(e)
    await plugin.judge(e)
    plugin.client.decide.assert_not_awaited()
    assert not e.get_extra(DECISION)
