import json, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

folder = Path("data") / sys.argv[1]
load = lambda name: pd.DataFrame(json.loads((folder / f"{name}.json").read_text()))

drivers = load("drivers").set_index("driver_number")
laps, pit, stints, pos = load("laps"), load("pit"), load("stints"), load("position")

print("Drivers:", len(drivers))
print("Race laps:", int(laps["lap_number"].max()))
f = laps.loc[laps["lap_duration"].idxmin()]
print(f"Fastest lap: {drivers.loc[f.driver_number, 'name_acronym']} "
      f"{f.lap_duration:.3f}s (lap {int(f.lap_number)})")
print("Pit stops:", len(pit))
print("Stints per compound:", stints.groupby("compound").size().to_dict())

final = pos.sort_values("date").groupby("driver_number").last().sort_values("position")
print("Top 3:", [drivers.loc[n, "name_acronym"] for n in final.index[:3]])

# Track map from the winner's car positions
winner = final.index[0]
loc = pd.DataFrame(json.loads((folder / "location" / f"{winner}.json").read_text()))
loc = loc[(loc.x != 0) | (loc.y != 0)]
plt.figure(figsize=(7, 7))
plt.plot(loc.x, loc.y, linewidth=0.6)
plt.axis("equal"); plt.axis("off")
plt.title(f"Track from {drivers.loc[winner, 'name_acronym']}'s position data")
plt.savefig(folder / "track.png", dpi=150)
print(f"Track map saved to {folder / 'track.png'}")