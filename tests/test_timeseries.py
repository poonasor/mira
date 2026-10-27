"""Review metrics aggregate through the dashboard route with an isolated event store."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest


@pytest.fixture
def timeseries_events(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", ":memory:")
    monkeypatch.setenv("ADMIN_PASSWORD", "offline-test-only")
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    from mira.dashboard.routers import core

    def load(dates):
        events = [
            SimpleNamespace(
                created_at=dt.timestamp(),
                comments_posted=1,
                blockers=0,
                warnings=1,
                suggestions=0,
                lines_changed=10,
                tokens_used=100,
                categories="correctness",
            )
            for dt in dates
        ]
        store = SimpleNamespace(list_review_events=lambda limit: events, close=lambda: None)
        database = SimpleNamespace(
            list_repos=lambda: [SimpleNamespace(owner="example", repo="app", platform="github")]
        )
        monkeypatch.setattr(core._api, "_app_db", database)
        monkeypatch.setattr(core, "IndexStore", SimpleNamespace(open=lambda *a, **k: store))
        return core.get_timeseries

    return load


@pytest.mark.parametrize(
    "dates, expected",
    [
        ([(2024, 12, 30), (2025, 1, 1)], "2025-W01"),
        ([(2020, 12, 31), (2021, 1, 1)], "2020-W53"),
        ([(2018, 12, 31), (2019, 1, 1)], "2019-W01"),
    ],
)
def test_iso_week_spans_calendar_year(timeseries_events, dates, expected):
    get_timeseries = timeseries_events([datetime(*date, tzinfo=UTC) for date in dates])

    points = get_timeseries("week")

    assert len(points) == 1
    assert points[0].model_dump() == {
        "date": expected,
        "reviews": 2,
        "comments": 2,
        "blockers": 0,
        "warnings": 2,
        "suggestions": 0,
        "lines_changed": 20,
        "tokens_used": 200,
        "categories": {"correctness": 2},
    }


def test_distinct_iso_weeks_remain_separate(timeseries_events):
    get_timeseries = timeseries_events(
        [datetime(2025, 1, 8, tzinfo=UTC), datetime(2025, 1, 2, tzinfo=UTC)]
    )

    points = get_timeseries("week")

    assert [(p.date, p.reviews) for p in points] == [("2025-W01", 1), ("2025-W02", 1)]


@pytest.mark.parametrize(
    "period, expected",
    [
        ("day", ["2024-12-30", "2025-01-01"]),
        ("month", ["2024-12", "2025-01"]),
    ],
)
def test_day_and_month_keys_are_unchanged(timeseries_events, period, expected):
    get_timeseries = timeseries_events(
        [datetime(2024, 12, 30, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)]
    )

    points = get_timeseries(period)

    assert [p.date for p in points] == expected
    assert [p.reviews for p in points] == [1, 1]


def test_empty_event_store(timeseries_events):
    get_timeseries = timeseries_events([])

    assert get_timeseries("week") == []
