"""UTC timestamps. Every stored instant is UTC ISO-8601 with a `Z` suffix and
millisecond precision; local presentation uses the camera's configured UTC offset
(`cameras.utc_offset_minutes`, default +05:30) -- there is no tz database dependency."""

from __future__ import annotations

import datetime as _dt

UTC = _dt.timezone.utc


def now_utc() -> _dt.datetime:
    return _dt.datetime.now(UTC)


def iso(dt: _dt.datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime: refusing to guess a timezone")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def now_iso() -> str:
    return iso(now_utc())


def parse_iso(text: str) -> _dt.datetime:
    if not text.endswith("Z") and "+" not in text[10:] and "-" not in text[10:]:
        raise ValueError(f"timestamp without a timezone: {text!r}")
    return _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def add_seconds(text: str, seconds: float) -> str:
    return iso(parse_iso(text) + _dt.timedelta(seconds=seconds))


def to_local_label(text: str, utc_offset_minutes: int) -> str:
    local = parse_iso(text).astimezone(_dt.timezone(_dt.timedelta(minutes=utc_offset_minutes)))
    sign = "+" if utc_offset_minutes >= 0 else "-"
    m = abs(utc_offset_minutes)
    return local.strftime("%Y-%m-%d %H:%M:%S") + f" UTC{sign}{m // 60:02d}:{m % 60:02d}"
