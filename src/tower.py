"""Print the timing tower at any lap, or replay the whole race in the terminal.

  python tower.py 9987               final classification
  python tower.py 9987 --lap 30      the moment the leader starts lap 30
  python tower.py 9987 --replay      replay the race at 20x (Ctrl+C to stop)
"""
import argparse
import time

from race_state import Race, RaceState

COMPOUND = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}


def fmt_lap(s):
    if s is None:
        return "-"
    m, sec = divmod(s, 60)
    return f"{int(m)}:{sec:06.3f}"


def fmt_clock(s):
    s = max(0, int(s))
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def fmt_temp(v):
    return "-" if v is None else f"{v:.1f}°C"


def render(race: Race, st: RaceState) -> str:
    s = race.session
    header = (f"{s.get('circuit_short_name')} {s.get('year')}  |  Lap {st.leader_lap}/{st.total_laps}  |  "
              f"{st.track_status}  |  Air {fmt_temp(st.air_temp)}  Track {fmt_temp(st.track_temp)}"
              f"{'  |  RAIN' if st.rainfall else ''}  |  T+{fmt_clock(st.t)}")
    lines = [header, "",
             f"{'P':>2}  {'DRIVER':<6} {'TEAM':<15} {'GAP':>9} {'INT':>9} {'LAST':>9} {'BEST':>9}  {'TYRE':<5} {'STOPS':>5}"]
    for c in st.cars:
        gap = "Leader" if c.position == 1 else c.gap_to_leader
        tyre = f"{COMPOUND.get(c.compound, '?')} {c.tyre_age}" if c.compound else "-"
        flag = "OUT" if c.retired else ("PIT" if c.in_pit else "")
        lines.append(f"{c.position or '-':>2}  {c.code:<6} {c.team[:15]:<15} {gap:>9} {c.interval:>9} "
                     f"{fmt_lap(c.last_lap):>9} {fmt_lap(c.best_lap):>9}  {tyre:<5} {c.pit_stops:>5}  {flag}")
    lines += ["", "Race control:"] + [f"  {m}" for m in st.messages]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session_key")
    p.add_argument("--lap", type=int, help="show the tower when the leader starts this lap")
    p.add_argument("--replay", action="store_true", help="replay the race in the terminal")
    p.add_argument("--speed", type=float, default=20, help="replay speed (default 20x)")
    p.add_argument("--from-lap", type=int, default=1, help="lap to start the replay from")
    a = p.parse_args()

    print("Loading race data...")
    race = Race(f"data/{a.session_key}")

    if a.replay:
        t, tick = race.time_at_lap(a.from_lap), 0.5
        try:
            while t <= race.end_s + 5:
                print("\033[2J\033[H" + render(race, race.state_at(t)), flush=True)
                time.sleep(tick)
                t += tick * a.speed
        except KeyboardInterrupt:
            pass
    else:
        t = race.time_at_lap(a.lap) if a.lap else race.end_s + 1
        print(render(race, race.state_at(t)))


if __name__ == "__main__":
    main()