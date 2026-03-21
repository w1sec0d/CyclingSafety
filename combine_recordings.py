#!/usr/bin/env python3
"""
Combine Sensor Logger per-sensor CSV files into the unified combined format
expected by MLP_Cycling_Safety.ipynb.

Each recording folder in own_data/cycling_recordings/ contains separate CSV
files per sensor (Accelerometer.csv, Gyroscope.csv, etc.).  This script
merges them into a single CSV per recording at 50 Hz (20 ms bins),
aggregating by average when multiple readings fall in the same bin.

Output goes to own_data/<folder_name>.csv.
"""

from pathlib import Path
import argparse
import math
import sys

import numpy as np
import pandas as pd

TARGET_HZ = 50
BIN_WIDTH = 1.0 / TARGET_HZ  # 0.02 s

# ── Sensor-file-to-column mapping ─────────────────────────────────────────
# Each entry: (filename, [(src_col, dst_col), ...])
# For xyz sensors the source file has columns: time, seconds_elapsed, z, y, x

XYZ_SENSORS = [
    ("Accelerometer.csv",            "accelerometer"),
    ("Gyroscope.csv",                "gyroscope"),
    ("Magnetometer.csv",             "magnetometer"),
    ("TotalAcceleration.csv",        "totalAcceleration"),
    ("Gravity.csv",                  "gravity"),
    ("MagnetometerUncalibrated.csv", "magnetometerUncalibrated"),
    ("GyroscopeUncalibrated.csv",    "gyroscopeUncalibrated"),
    ("AccelerometerUncalibrated.csv", "accelerometerUncalibrated"),
]

LOCATION_COLS = [
    "bearingAccuracy", "speedAccuracy", "verticalAccuracy",
    "horizontalAccuracy", "speed", "bearing", "altitude",
    "longitude", "latitude",
]

# Column order in the final CSV (matches existing combined format,
# plus gravity_* and annotation at the end).
COLUMN_ORDER = [
    "seconds_elapsed",
    # Accelerometer
    "accelerometer_z", "accelerometer_y", "accelerometer_x",
    # Gyroscope
    "gyroscope_z", "gyroscope_y", "gyroscope_x",
    # Magnetometer
    "magnetometer_z", "magnetometer_y", "magnetometer_x",
    # Compass
    "compass_magneticBearing",
    # Location
    "location_bearingAccuracy", "location_speedAccuracy",
    "location_verticalAccuracy", "location_horizontalAccuracy",
    "location_speed", "location_bearing", "location_altitude",
    "location_longitude", "location_latitude",
    # TotalAcceleration
    "totalAcceleration_z", "totalAcceleration_y", "totalAcceleration_x",
    # Magnetometer Uncalibrated
    "magnetometerUncalibrated_z", "magnetometerUncalibrated_y", "magnetometerUncalibrated_x",
    # Gyroscope Uncalibrated
    "gyroscopeUncalibrated_z", "gyroscopeUncalibrated_y", "gyroscopeUncalibrated_x",
    # Accelerometer Uncalibrated
    "accelerometerUncalibrated_z", "accelerometerUncalibrated_y", "accelerometerUncalibrated_x",
    # Gravity (new)
    "gravity_z", "gravity_y", "gravity_x",
    # Annotation (new)
    "annotation",
]


def _assign_bin(seconds: pd.Series) -> pd.Series:
    """Map each timestamp to its nearest 20 ms bin centre."""
    return (seconds / BIN_WIDTH).round() * BIN_WIDTH


def _load_xyz_sensor(folder: Path, filename: str, prefix: str) -> pd.DataFrame | None:
    df = _safe_read_csv(folder / filename)
    if df is None or "seconds_elapsed" not in df.columns:
        return None
    df["bin"] = _assign_bin(df["seconds_elapsed"].astype(float))
    rename = {"z": f"{prefix}_z", "y": f"{prefix}_y", "x": f"{prefix}_x"}
    df = df.rename(columns=rename)
    cols = ["bin"] + [rename[c] for c in ("z", "y", "x") if rename[c] in df.columns]
    return df[cols].groupby("bin", as_index=False).mean()


def _load_compass(folder: Path) -> pd.DataFrame | None:
    df = _safe_read_csv(folder / "Compass.csv")
    if df is None or "seconds_elapsed" not in df.columns:
        return None
    df["bin"] = _assign_bin(df["seconds_elapsed"].astype(float))
    df = df.rename(columns={"magneticBearing": "compass_magneticBearing"})
    return df[["bin", "compass_magneticBearing"]].groupby("bin", as_index=False).mean()


def _load_location(folder: Path) -> pd.DataFrame | None:
    df = _safe_read_csv(folder / "Location.csv")
    if df is None or "seconds_elapsed" not in df.columns:
        return None
    df["bin"] = _assign_bin(df["seconds_elapsed"].astype(float))
    rename = {c: f"location_{c}" for c in LOCATION_COLS if c in df.columns}
    df = df.rename(columns=rename)
    cols = ["bin"] + list(rename.values())
    return df[cols].groupby("bin", as_index=False).mean()


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV, returning None for empty or unparseable files."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_annotation(folder: Path) -> pd.DataFrame | None:
    df = _safe_read_csv(folder / "Annotation.csv")
    if df is None or df.empty:
        return None
    if "seconds_elapsed" not in df.columns or "text" not in df.columns:
        return None
    df["bin"] = _assign_bin(df["seconds_elapsed"].astype(float))
    df = df.rename(columns={"text": "annotation"})
    return df[["bin", "annotation"]].drop_duplicates(subset="bin", keep="first")


def combine_recording(folder: Path) -> pd.DataFrame:
    """Combine all sensor CSVs in *folder* into a single 50 Hz DataFrame."""
    parts: list[pd.DataFrame] = []

    for filename, prefix in XYZ_SENSORS:
        part = _load_xyz_sensor(folder, filename, prefix)
        if part is not None:
            parts.append(part)

    compass = _load_compass(folder)
    if compass is not None:
        parts.append(compass)

    location = _load_location(folder)
    if location is not None:
        parts.append(location)

    annotation = _load_annotation(folder)

    if not parts:
        raise ValueError(f"No sensor data found in {folder}")

    # Build the unified time grid from the earliest to latest bin
    all_bins = pd.concat([p["bin"] for p in parts]).drop_duplicates().sort_values()
    grid = pd.DataFrame({"bin": all_bins})

    for part in parts:
        grid = grid.merge(part, on="bin", how="left")

    if annotation is not None and not annotation.empty:
        grid = grid.merge(annotation, on="bin", how="left")

    grid = grid.rename(columns={"bin": "seconds_elapsed"})
    grid["seconds_elapsed"] = grid["seconds_elapsed"].round(6)
    grid = grid.sort_values("seconds_elapsed").reset_index(drop=True)

    # Reindex to the expected column order, adding missing columns as NaN
    for col in COLUMN_ORDER:
        if col not in grid.columns:
            grid[col] = np.nan

    return grid[COLUMN_ORDER]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "own_data" / "cycling_recordings",
        help="Directory containing recording folders (default: own_data/cycling_recordings)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "own_data",
        help="Directory to write combined CSVs (default: own_data/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing combined CSVs (default: True)",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])
    if not folders:
        print("No recording folders found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(folders)} recording folders in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target frequency: {TARGET_HZ} Hz (bin width: {BIN_WIDTH*1000:.0f} ms)\n")

    success, skipped, failed = 0, 0, 0
    for folder in folders:
        out_path = args.output_dir / f"{folder.name}.csv"

        if out_path.exists() and not args.overwrite:
            print(f"  SKIP {folder.name} (already exists)")
            skipped += 1
            continue

        try:
            combined = combine_recording(folder)
            combined.to_csv(out_path, index=False)

            duration = combined["seconds_elapsed"].max()
            n_rows = len(combined)
            n_annot = combined["annotation"].notna().sum()
            print(
                f"  OK   {folder.name}\n"
                f"       {n_rows} rows, {duration:.1f}s duration, "
                f"{n_annot} annotations"
            )
            success += 1
        except Exception as e:
            print(f"  FAIL {folder.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {success} combined, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
