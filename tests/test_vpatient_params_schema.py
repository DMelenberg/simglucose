"""Schema + steady-state sentinel test for simglucose/params/vpatient_params.csv. Closes simglucose ROADMAP finding M5."""

import math
from pathlib import Path

import pandas as pd
import simglucose

_EXPECTED_COLUMNS = [
    "Name", "i", "x0_ 1", "x0_ 2", "x0_ 3", "x0_ 4", "x0_ 5", "x0_ 6",
    "x0_ 7", "x0_ 8", "x0_ 9", "x0_10", "x0_11", "x0_12", "x0_13",
    "BW", "EGPb", "Gb", "Ib", "kabs", "kmax", "kmin", "b", "d", "Vg", "Vi",
    "Ipb", "Vmx", "Km0", "k2", "k1", "p2u", "m1", "m5", "CL", "HEb", "m2",
    "m4", "m30", "Ilb", "ki", "kp2", "kp3", "f", "Gpb", "ke1", "ke2",
    "Fsnc", "Gtb", "Vm0", "Rdb", "PCRb", "kd", "ksc", "ka1", "ka2",
    "dosekempt", "u2ss", "isc1ss", "isc2ss", "kp1", "patient_history",
]


def _csv_path() -> Path:
    return Path(simglucose.__file__).parent / "params" / "vpatient_params.csv"


def test_vpatient_params_csv_exists():
    assert _csv_path().is_file()


def test_vpatient_params_csv_columns():
    df = pd.read_csv(_csv_path())
    assert list(df.columns) == _EXPECTED_COLUMNS


def test_vpatient_params_csv_row_count_and_cohorts():
    df = pd.read_csv(_csv_path())
    assert len(df) == 30
    adolescent_count = df["Name"].str.startswith("adolescent#").sum()
    adult_count = df["Name"].str.startswith("adult#").sum()
    child_count = df["Name"].str.startswith("child#").sum()
    assert adolescent_count == 10
    assert adult_count == 10
    assert child_count == 10
    assert adolescent_count + adult_count + child_count == 30 == len(df)


def test_vpatient_params_csv_dtypes():
    df = pd.read_csv(_csv_path())
    # pandas 3.x returns StringDtype instead of object for string columns
    assert pd.api.types.is_string_dtype(df["Name"])
    for col in df.columns:
        if col == "Name":
            continue
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col!r} is not numeric"


def test_adult001_steady_state_sentinel():
    df = pd.read_csv(_csv_path())
    row = df[df["Name"] == "adult#001"].iloc[0]

    # Body weight (kg) — patient size parameter
    assert math.isclose(row["BW"], 102.32, abs_tol=1e-2)

    # Total glucose mass in plasma compartment at steady state (mg/kg)
    assert math.isclose(row["Gpb"], 265.370112, abs_tol=1e-6)

    # Glucose distribution volume (dL/kg)
    assert math.isclose(row["Vg"], 1.9152, abs_tol=1e-6)

    # Steady-state plasma blood glucose (mg/dL)
    assert math.isclose(row["Gb"], 138.56, abs_tol=1e-2)

    # Derived-vs-stored consistency: ODE expects Gb = Gpb / Vg at steady state
    assert math.isclose(row["Gpb"] / row["Vg"], row["Gb"], abs_tol=1e-2)

    # Steady-state basal insulin infusion rate (pmol/(L·kg))
    assert math.isclose(row["u2ss"], 1.2386244136, abs_tol=1e-6)
