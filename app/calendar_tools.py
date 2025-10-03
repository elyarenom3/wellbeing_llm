from __future__ import annotations
from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from .models import CalendarBlock, Plan, UserContext


def _resolve_timezone(name: str | None) -> ZoneInfo:
    tz_name = name or "UTC"
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _round_to_next_quarter(dt: datetime) -> datetime:
    minutes = ((dt.minute // 15) + 1) * 15
    if minutes >= 60:
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=minutes, second=0, microsecond=0)
    return dt


def build_calendar_blocks(plan: Plan, context: UserContext, now: datetime | None = None) -> List[CalendarBlock]:
    tz = _resolve_timezone(context.timezone)
    current = now or datetime.now(tz)
    start_time = _round_to_next_quarter(current)
    blocks: List[CalendarBlock] = []
    gap = timedelta(minutes=5)
    for item in plan.items:
        duration = max(5, int(item.duration_minutes or 5))
        end_time = start_time + timedelta(minutes=duration)
        blocks.append(
            CalendarBlock(
                start_iso=start_time.isoformat(),
                end_iso=end_time.isoformat(),
                label=item.title,
                timezone=tz.key,
            )
        )
        start_time = end_time + gap
    return blocks
