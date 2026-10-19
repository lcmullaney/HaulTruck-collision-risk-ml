import pandas as pd
import numpy as np
#edit file path
INPUT_FILE = r"XXXXXXXX\segment_traversal.csv"
OUTPUT_FILE = r"XXXXXXXXX\observed_risk_dataset.csv"

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["segment_id", "entry_time"]
).reset_index(drop=True)

records = []

for segment_id, group in df.groupby("segment_id"):

    group = group.sort_values("entry_time").reset_index(drop=True)

    for i in range(len(group) - 1):
        a = group.iloc[i]
        b = group.iloc[i + 1]

        if a["truck_id"] == b["truck_id"]:
            continue

        overlap = max(
            0.0,
            min(a["exit_time"], b["exit_time"])
            - max(a["entry_time"], b["entry_time"])
        )

        same_direction = int(
            a["direction"] == b["direction"]
        )

        if same_direction:
            relative_speed = abs(
                a["speed_ms"] - b["speed_ms"]
            )
        else:
            relative_speed = (
                a["speed_ms"] + b["speed_ms"]
            )

        if overlap == 0:
            risk = "Safe"
        elif same_direction:
            risk = "Warning"
        else:
            risk = "Critical"

        records.append({
            "truck_A": int(a["truck_id"]),
            "truck_B": int(b["truck_id"]),
            "segment_id": int(segment_id),

            "entry_time_A": a["entry_time"],
            "entry_time_B": b["entry_time"],

            "overlap_s": overlap,
            "relative_speed_ms": relative_speed,
            "same_direction": same_direction,

            "speed_A_ms": a["speed_ms"],
            "speed_B_ms": b["speed_ms"],

            "loaded_A": int(a["loaded"]),
            "loaded_B": int(b["loaded"]),

            "gvw_A_t": a["gvw_t"],
            "gvw_B_t": b["gvw_t"],

            "segment_length_m": a["segment_length_m"],
            "grade": a["grade"],
            "speed_limit_kmh": a["speed_limit_kmh"],

            "risk": risk
        })

result = pd.DataFrame(records)

result.to_csv(OUTPUT_FILE, index=False)

print("\n=== Observed Encounter Dataset ===")
print(f"Total encounters: {len(result)}")

print("\nRisk class distribution:")
print(result["risk"].value_counts())

print("\nRisk class percentages:")
print(
    (result["risk"].value_counts(normalize=True) * 100).round(2)
)

print("\nOverlap statistics:")
print(result["overlap_s"].describe())
print(f"\nSaved to: {OUTPUT_FILE}")
