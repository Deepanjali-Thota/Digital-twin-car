"""Step 2: rebuild the full race state at any moment from the files saved in step 1.

Every time in here is "seconds since the session start" (date_start in session.json),
so the rest of the project never has to deal with time zones or date strings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RETIRED_AFTER_S = 240    # the leader started a lap this much later than your latest lap -> you stopped
VSC_CLEAR_AFTER_S = 20   # fallback if race control posts no explicit green after "VSC ENDING"


# ---------- small helpers ----------

def _load(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.json"
    return pd.DataFrame(json.loads(path.read_text())) if path.exists() else pd.DataFrame()


def _num(v):
    """float, or None for missing values (None / NaN / text)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _text(v, default: str = "") -> str:
    return v if isinstance(v, str) and v else default


def _gap(v) -> str:
    if isinstance(v, str):          # OpenF1 sends "+1 LAP" etc. as text
        return v
    f = _num(v)
    return "" if f is None else f"+{f:.3f}"


class Latest:
    """Answers "what was the most recent row at time t?" in O(log n), per driver or globally.

    Stores only the columns you ask for, as numpy arrays, so ~700k location rows stay small.
    """

    def __init__(self, df: pd.DataFrame, time_col: str, cols: list[str], key: str | None = "driver_number"):
        self._index: dict = {}
        if df.empty or time_col not in df:
            return
        df = df.dropna(subset=[time_col]).sort_values(time_col, kind="stable")
        groups = df.groupby(key) if key else [(None, df)]
        for k, g in groups:
            self._index[int(k) if key else None] = (
                g[time_col].to_numpy(dtype=float),
                {c: g[c].to_numpy() for c in cols if c in g},
            )

    def _pos(self, t: float, key) -> tuple[int, tuple | None]:
        entry = self._index.get(key)
        if entry is None:
            return 0, None
        return int(np.searchsorted(entry[0], t, side="right")), entry

    def at(self, t: float, key=None) -> dict | None:
        n, entry = self._pos(t, key)
        if n == 0:
            return None
        times, cols = entry
        return {"t": float(times[n - 1]), **{c: a[n - 1] for c, a in cols.items()}}

    def count(self, t: float, key=None) -> int:
        return self._pos(t, key)[0]

    def recent(self, t: float, k: int, key=None) -> list[dict]:
        n, entry = self._pos(t, key)
        if n == 0:
            return []
        times, cols = entry
        return [{"t": float(times[i]), **{c: a[i] for c, a in cols.items()}} for i in range(max(0, n - k), n)]


# ---------- the state you get back ----------

@dataclass
class CarState:
    driver_number: int
    code: str
    team: str
    colour: str
    position: int | None = None
    lap: int = 0
    gap_to_leader: str = ""
    interval: str = ""
    last_lap: float | None = None
    best_lap: float | None = None
    compound: str | None = None
    tyre_age: int | None = None
    pit_stops: int = 0
    in_pit: bool = False
    retired: bool = False
    x: float | None = None
    y: float | None = None


@dataclass
class RaceState:
    t: float                       # seconds since session start
    leader_lap: int
    total_laps: int
    track_status: str              # GREEN, VSC, VSC ENDING, SC, SC ENDING, RED FLAG, CHEQUERED
    air_temp: float | None
    track_temp: float | None
    rainfall: bool
    cars: list[CarState] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


# ---------- the race ----------

class Race:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.session = _load(folder, "session").iloc[0].to_dict()
        self.t0 = pd.to_datetime(self.session["date_start"], utc=True)
        self.drivers = _load(folder, "drivers").set_index("driver_number").to_dict("index")

        # Laps: a lap ends when the next one starts (or start + duration for the last lap).
        laps = _load(folder, "laps")
        laps["start"] = self._secs(laps["date_start"])
        laps["lap_duration"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
        laps = laps.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
        laps["end"] = laps.groupby("driver_number")["start"].shift(-1).fillna(laps["start"] + laps["lap_duration"])
        laps["best_so_far"] = laps.groupby("driver_number")["lap_duration"].cummin()
        laps["best_so_far"] = laps.groupby("driver_number")["best_so_far"].ffill()
        self._laps = laps
        self.total_laps = int(laps["lap_number"].max())
        self.lap_started = Latest(laps, "start", ["lap_number"])
        self.lap_finished = Latest(laps, "end", ["lap_duration", "best_so_far"])

        lap1 = laps.loc[laps["lap_number"] == 1, "start"].dropna()
        lap2 = laps.loc[laps["lap_number"] == 2, "start"].dropna()
        if len(lap1):
            self.start_s = float(lap1.min())
        elif len(lap2):
            self.start_s = float(lap2.min()) - 120.0   # lap 1 has no start time in this data
        else:
            self.start_s = 0.0
        self.end_s = float(laps["end"].max())

        stints = _load(folder, "stints")
        self.stints = {} if stints.empty else {
            int(k): g.sort_values("stint_number").to_dict("records") for k, g in stints.groupby("driver_number")
        }

        pit = _load(folder, "pit")
        if not pit.empty:
            pit["t"] = self._secs(pit["date"])
            dur_col = next((c for c in ("pit_duration", "lane_duration") if c in pit), None)
            pit["dur"] = pd.to_numeric(pit[dur_col], errors="coerce") if dur_col else np.nan
            pit["dur"] = pit["dur"].fillna(25.0)
        self.pits = Latest(pit, "t", ["dur", "lap_number"])

        self.position = self._timeline(_load(folder, "position"), ["position"])
        self.intervals = self._timeline(_load(folder, "intervals"), ["gap_to_leader", "interval"])

        loc_files = sorted((folder / "location").glob("*.json"))
        loc = pd.concat([pd.DataFrame(json.loads(p.read_text())) for p in loc_files], ignore_index=True) \
            if loc_files else pd.DataFrame()
        self.location = self._timeline(loc, ["x", "y"])

        rc = _load(folder, "race_control")
        if not rc.empty:
            rc["t"] = self._secs(rc["date"])
            rc = rc.sort_values("t", kind="stable")
        self.race_control = Latest(rc, "t", ["message", "lap_number"], key=None)
        self.status_changes = self._status_timeline(rc)
        self._status_times = np.array([s[0] for s in self.status_changes], dtype=float)

        self.weather = self._timeline(_load(folder, "weather"),
                                      ["air_temperature", "track_temperature", "rainfall"], key=None)

    # --- building blocks ---

    def _secs(self, s: pd.Series) -> pd.Series:
        return (pd.to_datetime(s, utc=True, format="ISO8601") - self.t0).dt.total_seconds()

    def _timeline(self, df: pd.DataFrame, cols: list[str], key: str | None = "driver_number") -> Latest:
        if not df.empty:
            df["t"] = self._secs(df["date"])
        return Latest(df, "t", cols, key)

    @staticmethod
    def _status_timeline(rc: pd.DataFrame) -> list[tuple[float, str]]:
        """Turn race-control messages into a list of (time, track status) changes."""
        changes = [(-1e12, "GREEN")]
        for r in rc.to_dict("records") if not rc.empty else []:
            msg = _text(r.get("message")).upper()
            flag = _text(r.get("flag")).upper()
            scope = _text(r.get("scope")).upper()
            if "VIRTUAL SAFETY CAR DEPLOYED" in msg:
                new = "VSC"
            elif "VIRTUAL SAFETY CAR ENDING" in msg or "VSC ENDING" in msg:
                new = "VSC ENDING"
            elif "SAFETY CAR DEPLOYED" in msg:
                new = "SC"
            elif "SAFETY CAR IN THIS LAP" in msg:
                new = "SC ENDING"
            elif flag == "RED":
                new = "RED FLAG"
            elif flag == "CHEQUERED":
                new = "CHEQUERED"
            elif scope == "TRACK" and flag in ("GREEN", "CLEAR"):
                new = "GREEN"
            else:
                continue
            if new != changes[-1][1]:
                changes.append((float(r["t"]), new))
        return changes

    def _track_status(self, t: float, leader_lap_start: float) -> str:
        i = int(np.searchsorted(self._status_times, t, side="right")) - 1
        when, status = self.status_changes[max(i, 0)]
        if status == "SC ENDING" and leader_lap_start > when:
            return "GREEN"      # SC came in and the leader has crossed the line: racing again
        if status == "VSC ENDING" and t - when > VSC_CLEAR_AFTER_S:
            return "GREEN"
        return status

    def _tyre(self, driver: int, lap: int) -> tuple[str | None, int | None]:
        stints = self.stints.get(driver, [])
        current = stints[0] if stints else None
        for s in stints:
            start = _num(s.get("lap_start"))
            if start is not None and start <= max(lap, 1):
                current = s
        if current is None:
            return None, None
        start = int(_num(current.get("lap_start")) or 1)
        age0 = int(_num(current.get("tyre_age_at_start")) or 0)
        return _text(current.get("compound")) or None, age0 + max(0, lap - start)

    # --- public API ---

    def time_at_lap(self, lap: int) -> float:
        """Moment the leader started this lap."""
        if lap <= 1:
            return self.start_s
        starts = self._laps.loc[self._laps["lap_number"] == lap, "start"].dropna()
        return float(starts.min()) if len(starts) else self.end_s

    def state_at(self, t: float) -> RaceState:
        cars, lap_starts = [], {}
        for num, d in self.drivers.items():
            num = int(num)
            started = self.lap_started.at(t, num)
            lap = int(started["lap_number"]) if started else (1 if t >= self.start_s else 0)
            lap_starts[num] = started["t"] if started else self.start_s
            done = self.lap_finished.at(t, num)
            pos = self.position.at(t, num)
            gap = self.intervals.at(t, num)
            loc = self.location.at(t, num)
            pit = self.pits.at(t, num)
            compound, age = self._tyre(num, lap)
            x, y = (_num(loc["x"]), _num(loc["y"])) if loc else (None, None)
            if x == 0 and y == 0:          # car not on track yet: OpenF1 reports (0, 0)
                x = y = None
            cars.append(CarState(
                driver_number=num,
                code=_text(d.get("name_acronym"), str(num)),
                team=_text(d.get("team_name")),
                colour="#" + _text(d.get("team_colour"), "888888"),
                position=int(pos["position"]) if pos else None,
                lap=lap,
                gap_to_leader=_gap(gap["gap_to_leader"]) if gap else "",
                interval=_gap(gap["interval"]) if gap else "",
                last_lap=_num(done["lap_duration"]) if done else None,
                best_lap=_num(done["best_so_far"]) if done else None,
                compound=compound,
                tyre_age=age,
                pit_stops=self.pits.count(t, num),
                in_pit=bool(pit) and t <= pit["t"] + float(pit["dur"]),
                x=x,
                y=y,
            ))

        leader_lap_start = max(lap_starts.values(), default=self.start_s)
        for car in cars:
            car.retired = leader_lap_start - lap_starts[car.driver_number] > RETIRED_AFTER_S
        cars.sort(key=lambda c: (c.position is None, c.position or 99))

        leader_lap = max((c.lap for c in cars), default=0)   # race lap = lap of the car furthest ahead
        w = self.weather.at(t)
        msgs = [f"Lap {int(_num(m['lap_number']) or 0)}: {_text(m['message'])}"
                for m in self.race_control.recent(t, 3)]

        return RaceState(
            t=t,
            leader_lap=min(leader_lap, self.total_laps),
            total_laps=self.total_laps,
            track_status=self._track_status(t, leader_lap_start),
            air_temp=_num(w["air_temperature"]) if w else None,
            track_temp=_num(w["track_temperature"]) if w else None,
            rainfall=bool(_num(w["rainfall"])) if w else False,
            cars=cars,
            messages=msgs,
        )