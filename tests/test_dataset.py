import json
import sqlite3

import pytest

from astrbot_plugin_fastbrain_proactive.dataset import COLUMNS, Dataset
from astrbot_plugin_fastbrain_proactive.engine import Room, Settings, build_payload

OUTER = set(
    "content_fingerprint context_hash date effort input label label_source label_text model policy_version preprocessing_version record_id stratum".split()
)
INNER = set(
    "columns context current_time image_policy limitations media_content_available pinned_references policy_version record_id target".split()
)


def args(mid="t", p=0.9, action="reply_reserved"):
    s = Settings.read({})
    r = Room()
    r.add("alice", "这个安排呢？", "h", 100, s, {"_id": "h"})
    r.add(
        "bob",
        "木头把这个安排整理一下",
        "t",
        105,
        s,
        {
            "_id": mid,
            "mentions": ["alice"],
            "_images": 1,
            "reply_to": {
                "speaker": "alice",
                "text": "这个安排呢？",
                "_message_id": "h",
                "_time": 123,
            },
        },
    )
    payload, _ = build_payload(s, r, "robot")
    return dict(
        scope="qq:raw-group",
        self_id="robot",
        message_id=mid,
        received_at=1793380000,
        rows=tuple(r.messages),
        payload=payload,
        p_true=p,
        threshold=0.5,
        action=action,
        elapsed_ms=700,
        queue_ms=12,
    )


def read_data(root):
    return [
        json.loads(line)
        for file in sorted((root / "data").glob("*.jsonl"))
        for line in file.read_text().splitlines()
    ]


def test_exact_25k_schema_and_model_visible_context(tmp_path):
    d = Dataset(tmp_path, 0.1)
    kw = args()
    rid = d.record(**kw)
    row = read_data(tmp_path)[0]
    assert set(row) == OUTER
    inp = json.loads(row["input"])
    assert set(inp) == INNER and inp["columns"] == COLUMNS
    assert all(len(r) == 11 for r in inp["context"] + [inp["target"]])
    assert inp["context"][0][1] == -5 and inp["target"][1] == 0
    assert inp["target"][5] == [["person_1", inp["context"][0][0]]]
    assert inp["target"][7] == 1
    assert inp["target"][4] == ["person_1"]
    assert row["label"] is True and row["label_text"] == "True"
    assert row["record_id"] == rid and row["label_source"] == "model_silver"
    assert row["model"] == "FastBrain-9B" and row["effort"] == "none"
    assert "_message_id" not in kw["payload"]["state"]
    assert "_time" not in kw["payload"]["state"]
    assert "raw-group" not in json.dumps(row)
    with d.connect() as c:
        audit = json.loads(c.execute("SELECT audit FROM samples").fetchone()[0])
    assert audit["model_input"] == kw["payload"]
    assert audit["sample_probability"] == 1 and audit["human_verified"] is False
    assert audit["reason"] is None


def test_sampling_stable_random_and_true_always(tmp_path):
    d = Dataset(tmp_path, 0.1)
    # Check broad distribution without relying on one exact random seed.
    draws = [
        int(d.ident("false-sample", str(i))[:16], 16) / 2**64 < 0.1
        for i in range(10000)
    ]
    assert 700 < sum(draws) < 1300
    for i in range(12):
        d.record(**args(str(i), 0.9))
    assert len(read_data(tmp_path)) == 12
    d.rate = 0
    assert d.record(**args("false", 0.1)) is None
    d = Dataset(tmp_path, 1)
    # A config change/restart must not resample the same observed message.
    assert d.record(**args("false", 0.1)) is None
    assert d.record(**args("new-false", 0.1))
    d.record(**args("0", 0.9))
    assert len(read_data(tmp_path)) == 13
    assert read_data(tmp_path)[-1]["label_text"] == "False"


def test_projection_failure_is_recoverable_without_duplicate(tmp_path, monkeypatch):
    d = Dataset(tmp_path, 1)
    original = d.project_locked

    def fail(c):
        raise OSError("simulated full disk")

    monkeypatch.setattr(d, "project_locked", fail)
    with pytest.raises(OSError):
        d.record(**args())
    with sqlite3.connect(d.db) as c:
        assert c.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1
    monkeypatch.setattr(d, "project_locked", original)
    d.project()
    path = next((tmp_path / "data").glob("*.jsonl"))
    with path.open("ab") as f:
        f.write(b"partial-uncommitted-tail")
    Dataset(tmp_path, 1).record(**args())
    assert len(read_data(tmp_path)) == 1
    path.unlink()
    Dataset(tmp_path, 1)
    assert len(read_data(tmp_path)) == 1


def test_pruned_context_and_missing_quote_time(tmp_path):
    kw = args()
    state = json.loads(kw["payload"]["state"])
    state["history"] = []
    kw["payload"]["state"] = json.dumps(state)
    kw["rows"][-1][3]["reply_to"]["_time"] = None
    d = Dataset(tmp_path, 1)
    d.record(**kw)
    inp = json.loads(read_data(tmp_path)[0]["input"])
    assert inp["context"] == []
    assert inp["pinned_references"][0][1] is None
    assert inp["pinned_references"][0][3] == "这个安排呢？"
    assert inp["target"][9] is False


def test_outcomes_do_not_rewrite_training_label(tmp_path):
    d = Dataset(tmp_path, 1)
    rid = d.record(**args(action="reply_cooldown"))
    before = read_data(tmp_path)
    d.outcome(rid, "generation_started")
    assert read_data(tmp_path) == before
    assert before[0]["label"] is True
    assert list((tmp_path / "audit").glob("outcomes-*.jsonl"))


def test_invalid_sample_rate_rejected():
    for rate in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            Settings.read({"false_sample_rate": rate})
