# ============================================
# PM-KISAN PROJECT - PART 1: DATA LOADING & CLEANING
# ============================================

import re
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------
# STEP 1: LOAD THE DATA
# ---------------------------------------------
df = pd.read_csv("pmkisan_master_merged.csv")

print("===== ORIGINAL DATASET: first 5 rows =====")
print(df.head())

print("\n===== Shape (rows, columns) =====")
print(df.shape)

print("\n===== Structure and data types =====")
df.info()

# ---------------------------------------------
# STEP 2: CHECK DATA-QUALITY PROBLEMS (BEFORE)
# ---------------------------------------------
print("\n===== Missing values per column =====")
print(df.isnull().sum())

print("\nExact duplicate rows:", df.duplicated().sum())
print("Different state names:", df["State"].nunique())
print("Different period labels:", df["Period_Label"].nunique())
print("Rows where amount is exactly 0:", (df["Amount_Disbursed_Rs_Cr"] == 0).sum())

totals = ["ALL INDIA", "Total", "Grand Total"]
print("National total rows mixed with states:", df["State"].isin(totals).sum())

# Keep the "before" numbers so we can compare later
before = {
    "Rows": len(df),
    "Different state names": df["State"].nunique(),
    "Different period labels (after: instalments)": df["Period_Label"].nunique(),
    "Zero amounts": int((df["Amount_Disbursed_Rs_Cr"] == 0).sum()),
    "National total rows mixed with states": int(df["State"].isin(totals).sum()),
}

# ---------------------------------------------
# STEP 3: CLEAN THE DATA
# ---------------------------------------------
clean_df = df.copy()                       # keep the original safe

# 3a. remove extra spaces from text columns
for col in ["Source", "State", "Period_Label"]:
    clean_df[col] = clean_df[col].str.strip()

# 3b. fix the inconsistent state name
clean_df["State"] = clean_df["State"].replace({
    "The Dadra and Nagar Haveli and Daman and Diu":
    "Dadra and Nagar Haveli and Daman and Diu"
})

# 3c. remove exact duplicate rows
clean_df = clean_df.drop_duplicates()

# 3d. zero amount is not a real value, make it missing
clean_df.loc[clean_df["Amount_Disbursed_Rs_Cr"] == 0,
             "Amount_Disbursed_Rs_Cr"] = np.nan

# 3e. turn the many period labels into one instalment number (1 to 18)
def get_instalment(label):
    match = re.search(r"(\d+)(st|nd|rd|th)", label)   # "8th Period" -> 8
    if match:
        return int(match.group(1))
    if label[:3] == "Dec":
        season = 2
    elif label[:3] == "Apr":
        season = 0
    else:
        season = 1
    if "(" in label:                                   # "Apr-Jul(2019-20)"
        year = int(label.split("(")[1][:4])
    else:                                              # "April 2023 - July, 2023"
        year = int(re.search(r"\d{4}", label).group())
    return (year - 2019) * 3 + season + 2

clean_df["Instalment"] = clean_df["Period_Label"].apply(get_instalment)

# 3f. separate national totals from states
is_total = clean_df["State"].isin(totals)
national_df = clean_df[is_total].copy()
state_df = clean_df[~is_total].copy()

# 3g. keep one row per state per instalment (newest source first)
order = {"AU1297_RS266": 1, "AU321_RS258": 2, "AU795_LS_2023": 3, "AU520_RS266": 4}
state_df["rank"] = state_df["Source"].map(order)
state_df = state_df.sort_values("rank")
repeated = state_df.duplicated(subset=["State", "Instalment"]).sum()
state_df = state_df.drop_duplicates(subset=["State", "Instalment"])
state_df = state_df.drop(columns="rank")
state_df = state_df.sort_values(["State", "Instalment"]).reset_index(drop=True)

# 3h. extra columns
state_df["Amount_Available"] = state_df["Amount_Disbursed_Rs_Cr"].notna()
state_df["Avg_Rs_Per_Beneficiary"] = (
    state_df["Amount_Disbursed_Rs_Cr"] * 10000000 / state_df["Beneficiaries"]
)

# ---------------------------------------------
# STEP 4: CLEANED DATASET
# ---------------------------------------------
print("\n===== CLEANED STATE DATASET: first 5 rows =====")
print(state_df.head())
print("\nShape:", state_df.shape)
state_df.info()

# ---------------------------------------------
# STEP 5: BEFORE vs AFTER
# ---------------------------------------------
after = {
    "Rows": len(state_df),
    "Different state names": state_df["State"].nunique(),
    "Different period labels (after: instalments)": state_df["Instalment"].nunique(),
    "Zero amounts": int((state_df["Amount_Disbursed_Rs_Cr"] == 0).sum()),
    "National total rows mixed with states": int(state_df["State"].isin(totals).sum()),
}
comparison = pd.DataFrame({"Before": before, "After": after})
comparison.loc["Repeated state+instalment rows"] = [int(repeated), 0]
comparison.loc["Missing beneficiaries"] = [int(df["Beneficiaries"].isnull().sum()),
                                           int(state_df["Beneficiaries"].isnull().sum())]
comparison.loc["Missing amounts"] = [int(df["Amount_Disbursed_Rs_Cr"].isnull().sum()),
                                     int(state_df["Amount_Disbursed_Rs_Cr"].isnull().sum())]

print("\n===== BEFORE vs AFTER =====")
print(comparison)

print("\n===== Example: how labels became instalment numbers =====")
sample = clean_df.drop_duplicates("Period_Label")[["Period_Label", "Instalment"]]
print(sample[sample["Instalment"].isin([1, 8, 13, 18])].sort_values("Instalment"))

# ---------------------------------------------
# STEP 6: SAVE
# ---------------------------------------------
state_df.to_csv("pmkisan_cleaned_state_records.csv", index=False)
national_df.to_csv("pmkisan_cleaned_national_records.csv", index=False)