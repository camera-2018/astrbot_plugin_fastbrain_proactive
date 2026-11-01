"""Reuse completed core captions without fetching images or calling a provider.

AstrBot 4.29.x exposes no caption event. This instance-scoped adapter observes
GroupChatContext's formatter and caption method, retaining only their results.
ContextVar keeps simultaneous groups separate; unload restores both methods.
"""

from contextvars import ContextVar
from functools import wraps
import time

CAPTIONS = "fastbrain_proactive_image_captions_v1"
MAX_CAPTION_BYTES = 1200


class ImageCaptionBridge:
    def __init__(self, allowed):
        self.allowed = allowed
        self.current = ContextVar("fastbrain_caption_capture", default=None)
        self.target = None
        self.originals = {}
        self.wrappers = {}

    def attach(self, target):
        if target is self.target:
            return
        self.close()
        if not all(
            callable(getattr(target, name, None))
            for name in ("_format_message", "get_image_caption")
        ):
            return
        original_format = target._format_message
        original_caption = target.get_image_caption

        @wraps(original_format)
        async def format_message(event, cfg):
            capture = {"captions": [], "failed": 0, "bytes": 0, "truncated": False}
            enabled = (
                self.allowed(event)
                and not event.is_stopped()
                and cfg.get("image_caption")
            )
            token = self.current.set(capture if enabled else None)
            started = time.monotonic()
            try:
                return await original_format(event, cfg)
            finally:
                self.current.reset(token)
                if enabled and self.target is target:
                    capture["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                    event.set_extra(CAPTIONS, capture)

        @wraps(original_caption)
        async def get_caption(*args, **kwargs):
            capture = self.current.get()
            try:
                result = await original_caption(*args, **kwargs)
            except Exception:
                if capture is not None:
                    capture["failed"] += 1
                raise
            if capture is not None and isinstance(result, str) and result.strip():
                raw = result.strip().encode("utf-8")
                remaining = max(0, MAX_CAPTION_BYTES - capture["bytes"])
                clipped = raw[:remaining].decode("utf-8", errors="ignore")
                capture["truncated"] |= len(raw) > remaining
                if clipped:
                    capture["captions"].append(clipped)
                    capture["bytes"] += len(clipped.encode("utf-8"))
            return result

        self.target = target
        self.originals = {
            name: (name in vars(target), vars(target).get(name))
            for name in ("_format_message", "get_image_caption")
        }
        self.wrappers = {
            "_format_message": format_message,
            "get_image_caption": get_caption,
        }
        for name, wrapper in self.wrappers.items():
            setattr(target, name, wrapper)

    def close(self):
        if self.target is not None:
            for name, wrapper in self.wrappers.items():
                if getattr(self.target, name, None) is wrapper:
                    existed, value = self.originals[name]
                    if existed:
                        setattr(self.target, name, value)
                    else:
                        delattr(self.target, name)
        self.target = None


def caption_text(capture):
    if not capture or not capture.get("captions"):
        return ""
    text = "\n".join(f"[图片描述：{s}]" for s in capture["captions"])
    if capture.get("truncated"):
        text += "\n[图片描述已截短，未展示部分未知]"
    return text
