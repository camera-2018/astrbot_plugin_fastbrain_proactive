"""Local silver-label ledger and crash-recoverable JSONL projections.

Training rows match the existing 25k release. Audit/outcome data never enters input.
"""

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "fastbrain-proactive-v0.1.11"
COLUMNS = [
    "id",
    "seconds_relative_to_target",
    "speaker",
    "text",
    "mentions",
    "replies_speaker_id",
    "non_image_media",
    "ignored_image_count",
    "text_truncated",
    "relations_restored",
    "reply_metadata_present",
]


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


class Dataset:
    def __init__(self, directory: Path, false_rate: float):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = self.root / "ledger.sqlite"
        self.rate = false_rate
        self.lock = threading.Lock()
        with self.connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS observed(id TEXT PRIMARY KEY,label INTEGER NOT NULL,sampled INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS policies(id TEXT PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY,training TEXT NOT NULL,audit TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY,stream TEXT NOT NULL,line TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS journal_stream_seq ON journal(stream,seq);
            CREATE TABLE IF NOT EXISTS exports(stream TEXT PRIMARY KEY,seq INTEGER NOT NULL,offset INTEGER NOT NULL);
            """)
            c.execute("INSERT OR IGNORE INTO meta VALUES('salt',?)", (os.urandom(32),))
            self.salt = c.execute("SELECT value FROM meta WHERE key='salt'").fetchone()[
                0
            ]
        self.db.chmod(0o600)
        self.project()

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.db, timeout=2)
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=FULL")
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:
            c.close()

    def ident(self, *values):
        return hmac.new(
            self.salt, canonical(values).encode(), hashlib.sha256
        ).hexdigest()

    def record(
        self,
        *,
        scope,
        self_id,
        message_id,
        received_at,
        rows,
        payload,
        p_true,
        threshold,
        action,
        elapsed_ms,
        queue_ms,
    ):
        state = json.loads(payload["state"])
        identity = message_id or (received_at, state["target"])
        rid = "live_" + self.ident(scope, self_id, identity)
        label = p_true >= threshold
        draw = int(self.ident("false-sample", rid)[:16], 16) / 2**64
        selected = label or draw < self.rate
        with self.lock, self.connect() as c:
            if c.execute("SELECT 1 FROM observed WHERE id=?", (rid,)).fetchone():
                self.project_locked(c)
                return (
                    rid
                    if c.execute("SELECT 1 FROM samples WHERE id=?", (rid,)).fetchone()
                    else None
                )
            c.execute(
                "INSERT INTO observed VALUES(?,?,?)", (rid, int(label), int(selected))
            )
            if selected:
                training, audit = self.format_record(
                    rid,
                    scope,
                    self_id,
                    received_at,
                    rows,
                    payload,
                    p_true,
                    threshold,
                    action,
                    elapsed_ms,
                    queue_ms,
                )
                c.execute(
                    "INSERT INTO samples VALUES(?,?,?)",
                    (rid, canonical(training), canonical(audit)),
                )
                policy = {
                    "model": payload["model"],
                    "questions": payload["questions"],
                    "persona": state["persona"],
                    "bot_names": state["bot_names"],
                    "version": VERSION,
                }
                c.execute(
                    "INSERT OR IGNORE INTO policies VALUES(?,?)",
                    (audit["prompt_hash"], canonical(policy)),
                )
                day = training["date"]
                self.append(c, f"data/online-{day}.jsonl", training)
                self.append(c, f"audit/labels-{day}.jsonl", audit)
            c.commit()  # Ledger is authoritative even if projection fails.
            self.project_locked(c)
        return rid if selected else None

    def format_record(
        self,
        rid,
        scope,
        self_id,
        received_at,
        rows,
        payload,
        probability,
        threshold,
        action,
        elapsed_ms,
        queue_ms,
    ):
        state = json.loads(payload["state"])
        visible = state["history"] + [state["target"]]
        source = rows[-len(visible) :]
        assert len(source) == len(visible)
        target_mono = source[-1][0]

        def speaker(value):
            return (
                "target_bot"
                if value == "bot"
                else value.replace("user", "person_", 1)
                if value.startswith("user")
                else value
            )

        ids = [
            "live_"
            + self.ident(
                scope,
                self_id,
                row[3].get("_id")
                or (received_at + row[0] - target_mono, row[1], row[2]),
            )
            for row in source
        ]
        ids[-1] = rid
        known = {
            row[3].get("_id"): i for i, row in enumerate(source) if row[3].get("_id")
        }
        pinned, packed = {}, []
        for i, (row, item) in enumerate(zip(source, visible)):
            meta = row[3]
            reply = item.get("reply_to")
            edges = []
            restored = True
            if reply:
                raw_ref = meta.get("reply_to", {}).get("_message_id")
                index = known.get(raw_ref) if raw_ref else None
                if index is not None and index < i:
                    ref_id = ids[index]
                else:
                    ref_id = "quote_" + self.ident(
                        scope, self_id, raw_ref or (ids[i], reply)
                    )
                    quote_time = meta.get("reply_to", {}).get("_time")
                    relative = (
                        int(quote_time - received_at)
                        if quote_time and quote_time <= received_at
                        else None
                    )
                    pinned[ref_id] = [
                        ref_id,
                        relative,
                        speaker(reply["speaker"]),
                        reply["text"],
                        [],
                        [],
                        [],
                        0,
                        reply.get("truncated", False),
                        False,
                        False,
                    ]
                    restored = False
                edges = [[speaker(reply["speaker"]), ref_id]]
                restored = restored and reply["speaker"] != "unknown"
            packed.append(
                [
                    ids[i],
                    int(row[0] - target_mono),
                    speaker(item["speaker"]),
                    item["text"],
                    [speaker(x) for x in item.get("mentions", [])],
                    edges,
                    meta.get("_media", []),
                    max(0, meta.get("_images", 0) - meta.get("_captioned_images", 0)),
                    bool(meta.get("_text_truncated", False)),
                    restored,
                    bool(reply),
                ]
            )
        stamp = datetime.fromtimestamp(received_at, ZoneInfo("Asia/Shanghai"))
        inp = {
            "record_id": rid,
            "policy_version": VERSION,
            "current_time": stamp.isoformat(),
            "columns": COLUMNS,
            "context": packed[:-1],
            "pinned_references": list(pinned.values()),
            "target": packed[-1],
            "image_policy": "ignore_untranscribed_images",
            "media_content_available": any(
                row[3].get("_captioned_images", 0) for row in source
            ),
            "limitations": [
                "Online visible context; earlier messages may have been pruned.",
                "Times are receipt times; null reference times are unknown, not target time.",
                "Image descriptions, when present, are model-generated text; media bodies unavailable; unknown other bots are not identified.",
                "Missing reference metadata is not evidence of no reference.",
            ],
        }
        context_hash = sha(
            {
                "context": inp["context"],
                "pinned_references": inp["pinned_references"],
                "target": inp["target"],
            }
        )
        training = {
            "record_id": rid,
            "date": stamp.date().isoformat(),
            "stratum": "online_true"
            if probability >= threshold
            else "online_false_sample",
            "input": canonical(inp),
            "label": probability >= threshold,
            "label_text": str(probability >= threshold),
            "context_hash": context_hash,
            "policy_version": VERSION,
            "preprocessing_version": "online-compact-v1",
            "model": payload["model"],
            "effort": "none",
            "label_source": "model_silver",
            "content_fingerprint": sha(state),
        }
        policy = {
            "model": payload["model"],
            "questions": payload["questions"],
            "persona": state["persona"],
            "bot_names": state["bot_names"],
            "version": VERSION,
        }
        audit = {
            "record_id": rid,
            "group_hash": self.ident(scope, self_id),
            "label_source": "model_silver",
            "p_true": probability,
            "threshold": threshold,
            "decision": training["label"],
            "action": action,
            "sample_probability": 1.0 if training["label"] else self.rate,
            "false_sample_rate": self.rate,
            "policy_version": VERSION,
            "prompt_hash": sha(policy),
            "payload_hash": sha(payload),
            "model_input": payload,
            "elapsed_ms": elapsed_ms,
            "queue_ms": queue_ms,
            "recorded_at": time.time(),
            "human_verified": False,
            "reason": None,
        }
        return training, audit

    @staticmethod
    def append(c, stream, row):
        c.execute(
            "INSERT INTO journal(stream,line) VALUES(?,?)",
            (stream, canonical(row) + "\n"),
        )

    def outcome(self, rid, phase):
        with self.lock, self.connect() as c:
            if not c.execute("SELECT 1 FROM samples WHERE id=?", (rid,)).fetchone():
                return
            day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
            self.append(
                c,
                f"audit/outcomes-{day}.jsonl",
                {"record_id": rid, "phase": phase, "time": time.time()},
            )
            c.commit()
            self.project_locked(c)

    def project(self):
        with self.lock, self.connect() as c:
            self.project_locked(c)

    def project_locked(self, c):
        # Commit file offsets only after fsync; truncate uncommitted tails on recovery.
        c.execute("BEGIN IMMEDIATE")
        for (stream,) in c.execute("SELECT DISTINCT stream FROM journal").fetchall():
            row = c.execute(
                "SELECT seq,offset FROM exports WHERE stream=?", (stream,)
            ).fetchone()
            seq, offset = row or (0, 0)
            path = self.root / stream
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not path.exists() or path.stat().st_size < offset:
                seq, offset = 0, 0
            latest = c.execute(
                "SELECT MAX(seq) FROM journal WHERE stream=?", (stream,)
            ).fetchone()[0]
            if path.exists() and path.stat().st_size == offset and seq == latest:
                continue
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, "r+b") as f:
                f.truncate(offset)
                f.seek(offset)
                for seq, line in c.execute(
                    "SELECT seq,line FROM journal WHERE stream=? AND seq>? ORDER BY seq",
                    (stream, seq),
                ):
                    f.write(line.encode())
                f.flush()
                os.fsync(f.fileno())
                offset = f.tell()
            c.execute(
                "INSERT OR REPLACE INTO exports VALUES(?,?,?)", (stream, seq, offset)
            )
        c.commit()
