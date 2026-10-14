"""Download one F1 race from OpenF1 and save it to data/<session_key>/."""
import json, sys, time
from pathlib import Path
import httpx

YEAR, COUNTRY, SESSION_NAME = 2025, "Italy", "Race"
client = httpx.Client(base_url="https://api.openf1.org/v1", timeout=60)

def get(endpoint, **params):
    for attempt in range(5):
        r = client.get(f"/{endpoint}", params=params)
        if r.status_code == 429:                 # rate limited: back off and retry
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:                 # OpenF1 uses 404 for "no results"
            return []
        r.raise_for_status()
        time.sleep(0.4)                          # be polite to the free API
        return r.json()
    raise RuntimeError(f"Gave up on {endpoint}")

def save(path, rows, label):
    path.write_text(json.dumps(rows))
    print(f"  {label:<16} {len(rows):>8,} rows")

sessions = get("sessions", year=YEAR, country_name=COUNTRY, session_name=SESSION_NAME)
if not sessions:
    sys.exit("No session found. Check YEAR / COUNTRY / SESSION_NAME.")
s = sessions[0]
key = s["session_key"]
print(f"{s['year']} {s['country_name']} {s['session_name']} at {s['circuit_short_name']} (session_key={key})")

out = Path("data") / str(key)
(out / "location").mkdir(parents=True, exist_ok=True)
save(out / "session.json", [s], "session")

for name in ["drivers", "laps", "stints", "pit", "position", "intervals", "race_control", "weather"]:
    save(out / f"{name}.json", get(name, session_key=key), name)

# Car X/Y positions (~3.7 Hz), one file per driver because this data is large
for d in json.loads((out / "drivers.json").read_text()):
    n = d["driver_number"]
    save(out / "location" / f"{n}.json",
         get("location", session_key=key, driver_number=n),
         f"location {d['name_acronym']}")

print(f"\nDone. Saved to {out}")