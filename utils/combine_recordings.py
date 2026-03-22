#!/usr/bin/env python3
"""
Combine Sensor Logger per-sensor CSV files into the unified combined format
expected by MLP_Cycling_Safety.ipynb.

Each recording folder contains separate CSV files per sensor
(Accelerometer.csv, Gyroscope.csv, etc.).  This script merges them into a
single CSV per recording at 50 Hz (20 ms bins), aggregating by average when
multiple readings fall in the same bin.

For artificial-event recordings that include Annotation.csv, two additional
preprocessing steps are available:

  --dedup-gap SECS   Collapse rapid consecutive annotations within SECS of
                     each other, keeping only the last (handles mis-clicks).
                     Default: 2.0 s.

  --correct-timing   Shift each annotation's timestamp back to the nearest
                     sensor peak before the button-press, compensating for the
                     human reaction-time delay (~1.5–3 s) between the physical
                     event and the annotation.  Uses vertical accelerometer for
                     "Bache", yaw-rate gyroscope for "Esquivada", and total
                     acceleration magnitude for "Freno de emergencia".

Output goes to own_data/<folder_name>.csv by default; use --output-dir to
write elsewhere (e.g. data/artificial_combined/).
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

TARGET_HZ = 50
BIN_WIDTH = 1.0 / TARGET_HZ  # 0.02 s

# ── Annotation timing-correction parameters ───────────────────────────────────
# Search for a sensor peak in the window [t_ann - PEAK_SEARCH_MAX_LAG,
# t_ann - PEAK_SEARCH_MIN_LAG] before the button press.
PEAK_SEARCH_MAX_LAG = 4.0  # seconds before annotation to start searching
PEAK_SEARCH_MIN_LAG = 0.3  # seconds before annotation to stop searching
# A peak must be at least this many times the window median to be accepted.
PEAK_SNR_FACTOR = 1.5
# Fallback fixed offset when no clear peak is found.
DEFAULT_TIMING_OFFSET = 2.0  # seconds

# ── Event-type to sensor signal mapping ──────────────────────────────────────
# Maps the annotation text (lowercased) to (sensor_file, signal_kind).
# signal_kind: "z_abs" -> |z| from xyz sensor, "magnitude" -> sqrt(x²+y²+z²)
EVENT_SENSOR_MAP = {
    "bache": ("Accelerometer.csv", "z_abs"),
    "esquivada": ("Gyroscope.csv", "z_abs"),
    "freno de emergencia": ("Accelerometer.csv", "magnitude"),
}

# ── Sensor-file-to-column mapping ─────────────────────────────────────────
# Each entry: (filename, [(src_col, dst_col), ...])
# For xyz sensors the source file has columns: time, seconds_elapsed, z, y, x

XYZ_SENSORS = [
    ("Accelerometer.csv", "accelerometer"),
    ("Gyroscope.csv", "gyroscope"),
    ("Magnetometer.csv", "magnetometer"),
    ("TotalAcceleration.csv", "totalAcceleration"),
    ("Gravity.csv", "gravity"),
    ("MagnetometerUncalibrated.csv", "magnetometerUncalibrated"),
    ("GyroscopeUncalibrated.csv", "gyroscopeUncalibrated"),
    ("AccelerometerUncalibrated.csv", "accelerometerUncalibrated"),
]

LOCATION_COLS = [
    "bearingAccuracy",
    "speedAccuracy",
    "verticalAccuracy",
    "horizontalAccuracy",
    "speed",
    "bearing",
    "altitude",
    "longitude",
    "latitude",
]

# Column order in the final CSV (matches existing combined format,
# plus gravity_* and annotation at the end).
COLUMN_ORDER = [
    "seconds_elapsed",
    # Accelerometer
    "accelerometer_z",
    "accelerometer_y",
    "accelerometer_x",
    # Gyroscope
    "gyroscope_z",
    "gyroscope_y",
    "gyroscope_x",
    # Magnetometer
    "magnetometer_z",
    "magnetometer_y",
    "magnetometer_x",
    # Compass
    "compass_magneticBearing",
    # Location
    "location_bearingAccuracy",
    "location_speedAccuracy",
    "location_verticalAccuracy",
    "location_horizontalAccuracy",
    "location_speed",
    "location_bearing",
    "location_altitude",
    "location_longitude",
    "location_latitude",
    # TotalAcceleration
    "totalAcceleration_z",
    "totalAcceleration_y",
    "totalAcceleration_x",
    # Magnetometer Uncalibrated
    "magnetometerUncalibrated_z",
    "magnetometerUncalibrated_y",
    "magnetometerUncalibrated_x",
    # Gyroscope Uncalibrated
    "gyroscopeUncalibrated_z",
    "gyroscopeUncalibrated_y",
    "gyroscopeUncalibrated_x",
    # Accelerometer Uncalibrated
    "accelerometerUncalibrated_z",
    "accelerometerUncalibrated_y",
    "accelerometerUncalibrated_x",
    # Gravity
    "gravity_z",
    "gravity_y",
    "gravity_x",
    # Annotation
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


def _deduplicate_annotations(df: pd.DataFrame, max_gap: float = 2.0) -> pd.DataFrame:
    """Collapse rapid consecutive annotations caused by mis-clicks.

    Annotations within *max_gap* seconds of the previous one are grouped
    together; only the last annotation in each group is kept (the rider's
    intended button press).
    """
    if df.empty:
        return df
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)
    gaps = df["seconds_elapsed"].diff().fillna(max_gap + 1.0)
    group_id = (gaps > max_gap).cumsum()
    return df.groupby(group_id, sort=True).last().reset_index(drop=True)


def _load_sensor_signal(folder: Path, sensor_file: str) -> pd.DataFrame | None:
    """Load a raw xyz sensor CSV and return a DataFrame with seconds_elapsed
    and a pre-computed *signal* column (the series used for peak detection).

    Returns None if the file is missing or unreadable.
    """
    df = _safe_read_csv(folder / sensor_file)
    if df is None or "seconds_elapsed" not in df.columns:
        return None
    df = df[["seconds_elapsed", "x", "y", "z"]].dropna()
    return df.sort_values("seconds_elapsed").reset_index(drop=True)


def _compute_signal(df: pd.DataFrame, signal_kind: str) -> pd.Series:
    if signal_kind == "z_abs":
        return df["z"].abs()
    if signal_kind == "magnitude":
        return np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    raise ValueError(f"Unknown signal_kind: {signal_kind!r}")


def _correct_annotation_timing(
    folder: Path,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Shift each annotation timestamp to the nearest sensor peak that
    occurred just before the button press.

    For each annotated event the algorithm:
      1. Selects the relevant sensor and signal kind based on the event type.
      2. Extracts the sensor readings in the window
         [t_ann - PEAK_SEARCH_MAX_LAG, t_ann - PEAK_SEARCH_MIN_LAG].
      3. Finds the sample with the highest signal value in that window.
      4. Accepts the peak if its value exceeds PEAK_SNR_FACTOR * median of
         the window (confirming it is a genuine spike).
      5. Falls back to t_ann - DEFAULT_TIMING_OFFSET when no clear peak is
         found (e.g. signal is flat or the window is too short).

    Sensor DataFrames are loaded once per recording and cached locally.
    """
    if annotations.empty:
        return annotations

    sensor_cache: dict[str, pd.DataFrame | None] = {}
    corrected_times: list[float] = []

    for _, row in annotations.iterrows():
        t_ann = float(row["seconds_elapsed"])
        event_key = str(row["text"]).strip().lower()

        sensor_file, signal_kind = EVENT_SENSOR_MAP.get(
            event_key, ("Accelerometer.csv", "magnitude")
        )

        if sensor_file not in sensor_cache:
            sensor_cache[sensor_file] = _load_sensor_signal(folder, sensor_file)

        raw = sensor_cache[sensor_file]

        t_corrected = t_ann - DEFAULT_TIMING_OFFSET  # default fallback

        if raw is not None and len(raw) > 0:
            t_lo = t_ann - PEAK_SEARCH_MAX_LAG
            t_hi = t_ann - PEAK_SEARCH_MIN_LAG
            mask = (raw["seconds_elapsed"] >= t_lo) & (raw["seconds_elapsed"] <= t_hi)
            window = raw[mask]

            if len(window) >= 5:
                signal = _compute_signal(window, signal_kind)
                peak_pos = signal.idxmax()
                peak_val = signal.loc[peak_pos]
                median_val = signal.median()

                if median_val > 0 and peak_val >= PEAK_SNR_FACTOR * median_val:
                    t_corrected = float(raw.loc[peak_pos, "seconds_elapsed"])
                # If SNR is too low keep the default fixed-offset fallback

        corrected_times.append(t_corrected)

    result = annotations.copy()
    result["seconds_elapsed"] = corrected_times
    return result


def _load_annotation(
    folder: Path,
    dedup_gap: float | None = None,
    correct_timing: bool = False,
) -> pd.DataFrame | None:
    """Load and optionally clean the Annotation.csv for a recording folder.

    Parameters
    ----------
    folder:
        Recording folder containing Annotation.csv.
    dedup_gap:
        If not None, consecutive annotations within this many seconds are
        collapsed to the last one (mis-click removal).
    correct_timing:
        If True, each annotation timestamp is adjusted to the nearest sensor
        peak preceding the button press.
    """
    df = _safe_read_csv(folder / "Annotation.csv")
    if df is None or df.empty:
        return None
    if "seconds_elapsed" not in df.columns or "text" not in df.columns:
        return None

    df = df[["seconds_elapsed", "text"]].copy()
    df["seconds_elapsed"] = df["seconds_elapsed"].astype(float)

    # Step 1: remove mis-clicks
    if dedup_gap is not None:
        df = _deduplicate_annotations(df, max_gap=dedup_gap)

    # Step 2: shift timestamps to sensor peaks
    if correct_timing:
        df = _correct_annotation_timing(folder, df)

    # Bin to the 50 Hz grid and rename
    df["bin"] = _assign_bin(df["seconds_elapsed"])
    df = df.rename(columns={"text": "annotation"})
    return df[["bin", "annotation"]].drop_duplicates(subset="bin", keep="last")


def combine_recording(
    folder: Path,
    dedup_gap: float | None = None,
    correct_timing: bool = False,
) -> pd.DataFrame:
    """Combine all sensor CSVs in *folder* into a single 50 Hz DataFrame.

    Parameters
    ----------
    folder:
        Recording folder with per-sensor CSV files.
    dedup_gap:
        Passed to _load_annotation(); collapses mis-click annotation chains.
    correct_timing:
        Passed to _load_annotation(); aligns annotations to sensor peaks.
    """
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

    annotation = _load_annotation(
        folder, dedup_gap=dedup_gap, correct_timing=correct_timing
    )

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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument(
        "--dedup-gap",
        type=float,
        default=2.0,
        metavar="SECS",
        help=(
            "Collapse annotation chains where consecutive events are within "
            "SECS of each other, keeping only the last (handles mis-clicks). "
            "Set to 0 to disable. Default: 2.0"
        ),
    )
    parser.add_argument(
        "--correct-timing",
        action="store_true",
        default=False,
        help=(
            "Shift annotation timestamps to the nearest preceding sensor peak, "
            "compensating for human reaction-time delay between event and "
            "button press. Recommended for artificial-event recordings."
        ),
    )
    args = parser.parse_args()

    dedup_gap: float | None = args.dedup_gap if args.dedup_gap > 0 else None

    if not args.input_dir.exists():
        print(
            f"Error: input directory does not exist: {args.input_dir}", file=sys.stderr
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])
    if not folders:
        print("No recording folders found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(folders)} recording folders in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target frequency: {TARGET_HZ} Hz (bin width: {BIN_WIDTH*1000:.0f} ms)")
    print(
        f"Annotation dedup gap: {dedup_gap}s"
        if dedup_gap
        else "Annotation dedup: disabled"
    )
    print(f"Timing correction: {'enabled' if args.correct_timing else 'disabled'}\n")

    success, skipped, failed = 0, 0, 0
    for folder in folders:
        out_path = args.output_dir / f"{folder.name}.csv"

        if out_path.exists() and not args.overwrite:
            print(f"  SKIP {folder.name} (already exists)")
            skipped += 1
            continue

        try:
            combined = combine_recording(
                folder,
                dedup_gap=dedup_gap,
                correct_timing=args.correct_timing,
            )
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
