"""SystemOne HTTP client. No retries, redirects, response text or key logging."""

import asyncio
import json
import math
import time
from email.utils import parsedate_to_datetime

import aiohttp

from .fastbrain_queue import priority_slot
from .native_decision import NativeDecisionClient, NativeDecisionFailure


class DecisionError(Exception):
    def __init__(self, reason: str, retry_after: float = 60) -> None:
        super().__init__(reason)
        self.reason, self.retry_after = reason, retry_after


def probability(data: object, model: str) -> float:
    try:
        if not isinstance(data, dict) or data.get("model") != model:
            raise ValueError
        probs = data["answers"]["should_reply"]["probabilities"]
        values = [probs[k] for k in ("True", "False")]
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
            for x in values
        ):
            raise ValueError
        if abs(sum(values) - 1) > 0.01:
            raise ValueError
        return float(values[0])
    except (KeyError, TypeError, ValueError):
        raise DecisionError("invalid_response") from None


def retry_delay(value: str | None) -> float:
    try:
        seconds = float(value)
        if math.isfinite(seconds):
            return max(60, seconds)
    except (TypeError, ValueError):
        pass
    try:
        return max(60, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return 60


class SystemOneClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        timeout: float,
        *,
        priority_enabled=False,
        decision_provider_id="",
        decision_service=None,
        resolve_provider=None,
    ) -> None:
        self.endpoint, self._api_key, self.timeout = endpoint, api_key, timeout
        self.priority_enabled = priority_enabled
        self.decision_provider_id = decision_provider_id
        self.native = NativeDecisionClient(decision_service, resolve_provider)
        self._session = None
        self.last_queue_ms = 0
        self.last_request_ms = 0

    async def decide(self, payload: dict) -> float:
        if self.decision_provider_id:
            try:
                reply = await self.native.ask(
                    self.decision_provider_id,
                    payload,
                    timeout=self.timeout,
                    priority_enabled=self.priority_enabled,
                )
            except NativeDecisionFailure as exc:
                raise DecisionError(exc.reason, exc.retry_after) from None
            self.last_queue_ms, self.last_request_ms = reply.queue_ms, reply.request_ms
            return probability(
                {"model": reply.model, "answers": reply.answers}, payload["model"]
            )
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            async with priority_slot(
                self.endpoint, str(payload["model"]), enabled=self.priority_enabled
            ) as ticket:
                self.last_queue_ms = ticket.wait_ms
                started = time.monotonic()
                async with self._session.post(
                    self.endpoint, json=payload, headers=headers, allow_redirects=False
                ) as response:
                    if response.status != 200:
                        raise DecisionError(
                            f"http_{response.status}",
                            retry_delay(response.headers.get("Retry-After")),
                        )
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(8192):
                        body.extend(chunk)
                        if len(body) > 65536:
                            raise DecisionError("response_too_large")
                    try:
                        data = json.loads(body)
                    except (ValueError, UnicodeDecodeError):
                        raise DecisionError("invalid_json") from None
                    result = probability(data, str(payload["model"]))
                    self.last_request_ms = round((time.monotonic() - started) * 1000)
                    return result
        except asyncio.TimeoutError:
            raise DecisionError("timeout") from None
        except aiohttp.ClientError:
            raise DecisionError("network") from None

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
