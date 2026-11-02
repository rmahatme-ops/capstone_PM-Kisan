"""
pmkisan_utils.py  -  USER-DEFINED MODULE for the PM-KISAN project   (Syllabus 2.2)

Imported in the main script with:   import pmkisan_utils as pu
Uses standard modules: os, re, math, time, random, statistics, array, datetime
"""

import os
import re
import math
import random
import statistics
from array import array
from datetime import datetime

import pandas as pd

# ----- constants (variables) -----------------------------------------------
MIN_BASE = 10000          # ignore % change when previous count is under 10,000 farmers
DROP_LIMIT = -10          # a "sharp drop" = fall of 10% or more
NOMINAL_RS = 2000         # scheme pays Rs 2,000 per instalment
SOURCE_RANK = {"AU1297_RS266": 1, "AU321_RS258": 2, "AU520_RS266": 3}   # dictionary


# ----- Errors and exception handling (1.4) ----------------------------------
def load_csv(path):
    """Read a CSV file. Function with a single argument and exception handling."""
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("ERROR: file not found ->", path)
        raise
    except pd.errors.EmptyDataError:
        print("ERROR: file is empty ->", path)
        raise
    else:
        print("Loaded", path, "with shape", data.shape)
        return data
    finally:
        pass          # nothing to clean up here; 'pass' does nothing (syllabus: pass statement)


# ----- 're' module (2.1) ----------------------------------------------------
def get_instalment(label):
    """Turn a period label into an instalment number 1-18 (used by the cleaning step)."""
    match = re.search(r"(\d+)(st|nd|rd|th)", label)         # '8th Period' -> 8
    if match:
        return int(match.group(1))
    if label[:3] == "Dec":
        season = 2
    elif label[:3] == "Apr":
        season = 0
    else:
        season = 1
    if "(" in label:
        year = int(label.split("(")[1][:4])
    else:
        year = int(re.search(r"\d{4}", label).group())
    return (year - 2019) * 3 + season + 2


# ----- 'datetime' module (2.1) ----------------------------------------------
def instalment_dates(count=18):
    """Start date of every instalment. Instalment 1 = Dec 2018, then every 4 months."""
    dates = []
    year, month = 2018, 12
    for _ in range(count):                  # for loop
        dates.append(datetime(year, month, 1))
        month += 4
        if month > 12:                      # nested decision inside a loop
            month -= 12
            year += 1
    return dates


def financial_year(instalment):
    """Financial year of an instalment, e.g. 5 -> '2020-21'."""
    start = instalment_dates()[instalment - 1]
    fy_start = start.year if start.month >= 4 else start.year - 1
    return str(fy_start) + "-" + str(fy_start + 1)[-2:]


# ----- if / elif / else ladder (1.3) -----------------------------------------
def classify_consistency(rmse):
    """Rate how closely a state follows the national trend."""
    if rmse < 5:
        return "High"
    elif rmse < 12:
        return "Moderate"
    else:
        return "Low"


# ----- Recursive function (1.4) ----------------------------------------------
def total_recursive(values):
    """Sum of a list written as a recursive function (used as a cross-check)."""
    if len(values) == 0:
        return 0
    return values[0] + total_recursive(values[1:])


# ----- 'math', 'statistics', 'random', 'array' modules (2.1) -------------------
def rmse(gaps):
    """Root mean square of a list of gaps (uses math.sqrt)."""
    return math.sqrt(sum(g * g for g in gaps) / len(gaps))


def bootstrap_ci(values, repeats=1000, seed=42):
    """95% bootstrap confidence interval of the mean (uses random + statistics)."""
    random.seed(seed)
    means = sorted(statistics.mean(random.choices(values, k=len(values)))
                   for _ in range(repeats))
    return means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]


def peak_of(values_in_millions):
    """Return (peak value, position) of a series stored in an 'array' object."""
    arr = array("d", values_in_millions)           # typed array from the array module
    peak = max(arr)
    return peak, arr.tolist().index(peak)


# ----- String functions: case conversion and alignment (2.1) -----------------
def line(label, value, width=42):
    """One neat output line: text aligned with ljust / rjust."""
    return label.strip().title().ljust(width, ".") + str(value).rjust(14)


def heading(text):
    print("\n" + text.upper().center(70, "="))


# ----- 'os' module (2.1) ------------------------------------------------------
def make_folder(name):
    os.makedirs(name, exist_ok=True)
    return name


def list_outputs(folder):
    return sorted(os.listdir(folder))


# ----- Data helpers used by the main script -----------------------------------
def one_row_per_instalment(national):
    """National file has several rows per instalment (different sources); keep one."""
    nat = national.copy()
    nat["rank"] = nat["Source"].map(SOURCE_RANK)
    nat = (nat.sort_values("rank").drop_duplicates("Instalment")
           .sort_values("Instalment").set_index("Instalment"))
    nat["Pct_Change"] = nat["Beneficiaries"].pct_change() * 100
    return nat


def top_states(pivot, last=18, how_many=5):
    """Print the largest states using continue and break (syllabus: loop control)."""
    shown = 0
    names = []
    for state, value in pivot[last].sort_values(ascending=False).items():
        if pd.isna(value):
            continue                         # skip states with no data
        names.append((state, int(value)))
        shown += 1
        if shown == how_many:
            break                            # stop after the top few
    return names


# ----- Shared data helpers (used by data_statistics.py and data_timeseries.py) -----
STATE_FILE = "pmkisan_cleaned_state_records.csv"
NATIONAL_FILE = "pmkisan_cleaned_national_records.csv"


def build_pivot(state):
    """States as rows, instalments as columns (beneficiaries)."""
    return state.pivot_table(index="State", columns="Instalment",
                             values="Beneficiaries", aggfunc="first")


def build_money(state):
    """Rupees per beneficiary for every state-instalment that has an amount.
    Returns (all rows, rows that are safe to test = no outliers and base >= MIN_BASE)."""
    money = state.dropna(subset=["Amount_Disbursed_Rs_Cr", "Beneficiaries"]).copy()
    money["Rs"] = money["Amount_Disbursed_Rs_Cr"] * 1e7 / money["Beneficiaries"]
    money["Outlier"] = ~money["Rs"].between(1000, 3000)      # amount and count do not match
    money["FY"] = money["Instalment"].map(financial_year)
    ok = money[(~money["Outlier"]) & (money["Beneficiaries"] >= MIN_BASE)]
    return money, ok


def change_tables(pivot):
    """% change of every state vs the previous instalment (bases under MIN_BASE hidden).
    Saves p2_11 (state table) and p2_12 (national change, 30 complete states)."""
    prev = pivot.shift(1, axis=1)
    change = pivot.pct_change(axis=1, fill_method=None) * 100
    change_masked = change.where(prev >= MIN_BASE)
    change_masked.round(2).to_csv("p2_11_state_pct_change_by_instalment.csv")
    comp_change = pivot.dropna().sum().pct_change() * 100
    comp_change.round(2).to_csv("p2_12_national_pct_change_complete_states.csv",
                                header=["Pct_Change"])
    return prev, change_masked, comp_change
