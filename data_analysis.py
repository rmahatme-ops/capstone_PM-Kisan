"""
PART 2 - DATA MANIPULATION AND ANALYSIS (pandas)
Run AFTER data_cleaning.py. Produces p2_01 ... p2_10 CSV files.
"""

import pandas as pd
import pmkisan_utils as pu          # our own module

df = pd.read_csv("pmkisan_cleaned_state_records.csv", sep=None, engine="python")
df.columns = df.columns.str.strip()      # remove any stray spaces in column names

print(df.columns)                        # check the column names
print(df.shape)                          # should be 648 rows, 8 columns

# 1. Select the columns we need
data = df[["State", "Instalment", "Beneficiaries",
           "Amount_Disbursed_Rs_Cr", "Avg_Rs_Per_Beneficiary"]].copy()
print(data.head())

# 2. Check missing values
print(data.isnull().sum())

# 3. Group by instalment: total beneficiaries
total_inst = data.groupby("Instalment")["Beneficiaries"].sum().reset_index()
total_inst.columns = ["Instalment", "Total_Beneficiaries"]
total_inst["Change_vs_Previous"] = total_inst["Total_Beneficiaries"].diff()
print(total_inst)

# 4. Group by state: average and peak, sorted high to low
state_summary = (data.groupby("State")["Beneficiaries"]
                 .agg(Avg_Per_Instalment="mean", Peak="max")
                 .sort_values("Avg_Per_Instalment", ascending=False)
                 .round(0)
                 .reset_index())
print(state_summary.head(10))

# 5. Pivot table: states as rows, instalments as columns
pivot = data.pivot_table(index="State", columns="Instalment",
                         values="Beneficiaries", aggfunc="first")
print(pivot.head())

# 6. Percentage change between instalments (all states)
national = data.groupby("Instalment")["Beneficiaries"].sum()
pct_change = (national.pct_change() * 100).round(2)
print(pct_change)

# 7. Percentage change using only states with data in ALL 18 instalments
complete_states = pivot.dropna().index
print("States with complete data:", len(complete_states))

comparable = (data[data["State"].isin(complete_states)]
              .groupby("Instalment")["Beneficiaries"].sum())
pct_comparable = (comparable.pct_change() * 100).round(2)
print(pct_comparable)

# 8. State-wise change from Instalment 5 to Instalment 18
sc = pivot[[5, 18]].dropna().copy()
sc.columns = ["Instalment_5", "Instalment_18"]
sc["Pct_Change_5_to_18"] = ((sc["Instalment_18"] - sc["Instalment_5"])
                            / sc["Instalment_5"] * 100).round(2)
sc = sc.sort_values("Pct_Change_5_to_18", ascending=False)
print(sc.head(5))   # biggest growth
print(sc.tail(5))   # biggest decline

# 9. Disbursement per beneficiary
# Rs per beneficiary = Amount (Rs crore) * 1,00,00,000 / Beneficiaries
money = data.dropna(subset=["Amount_Disbursed_Rs_Cr", "Beneficiaries"]).copy()
money["Rs_per_Beneficiary"] = money["Amount_Disbursed_Rs_Cr"] * 1e7 / money["Beneficiaries"]

# National figure for each instalment
nat_money = money.groupby("Instalment")[["Amount_Disbursed_Rs_Cr", "Beneficiaries"]].sum()
nat_money["Rs_per_Beneficiary"] = (nat_money["Amount_Disbursed_Rs_Cr"] * 1e7
                                   / nat_money["Beneficiaries"]).round(2)
print(nat_money)

# Median by state, highest first (median, because a few rows are outliers)
state_rs = (money.groupby("State")["Rs_per_Beneficiary"].median()
            .sort_values(ascending=False).round(0).reset_index())
print(state_rs.head(5))   # highest
print(state_rs.tail(5))   # lowest

# 10. Save every result as a CSV file
data.to_csv("p2_01_selected_columns.csv", index=False)

data.isnull().sum().reset_index(name="Missing_Values").rename(
    columns={"index": "Column"}).to_csv("p2_02_missing_values.csv", index=False)

total_inst.to_csv("p2_03_total_beneficiaries_by_instalment.csv", index=False)
state_summary.to_csv("p2_04_state_summary.csv", index=False)
pivot.to_csv("p2_05_pivot_state_by_instalment.csv")

# combine both percentage-change results into one table
pct = pd.DataFrame({
    "Total_All_States": national,
    "Pct_Change_All_States": pct_change,
    "Total_Comparable_States": comparable,
    "Pct_Change_Comparable_States": pct_comparable,
}).reset_index()
pct.to_csv("p2_06_pct_change_national.csv", index=False)

sc.reset_index().to_csv("p2_07_state_pct_change_instalment5_to_18.csv", index=False)
nat_money.reset_index().to_csv("p2_08_disbursement_per_beneficiary_national.csv", index=False)

state_rs.columns = ["State", "Avg_Rs_per_Beneficiary"]
state_rs.to_csv("p2_09_disbursement_per_beneficiary_by_state.csv", index=False)

print("All CSV files saved")

# 11. Key findings: what the results mean
inst18 = data[data["Instalment"] == 18].dropna(subset=["Beneficiaries"])
top10_share = inst18.nlargest(10, "Beneficiaries")["Beneficiaries"].sum() / inst18["Beneficiaries"].sum() * 100
drop12 = (pivot[12] - pivot[11]).sort_values()

findings = [
    ("Peak national beneficiaries",
     f"{national.max():,.0f} (Instalment {national.idxmax()})",
     "Coverage grew from about 32 million to over 100 million as the scheme rolled out."),
    ("Latest national beneficiaries",
     f"{national[18]:,.0f} (Instalment 18)",
     "Numbers recovered after the dip but are still below the peak."),
    ("Largest single fall",
     f"{pct_comparable.min():.2f}% at Instalment {pct_comparable.idxmin()} (comparable states)",
     "A sharp one-instalment drop; the biggest in the whole series."),
    ("Largest single rise",
     f"{pct_comparable.max():.2f}% at Instalment {pct_comparable.idxmax()} (comparable states)",
     "Early rollout: many new beneficiaries added at once."),
    ("State with biggest fall at Instalment 12",
     f"{drop12.index[0]} ({drop12.iloc[0]:,.0f} beneficiaries)",
     "The national dip is driven largely by big states."),
    ("Top 10 states' share of beneficiaries (Instalment 18)",
     f"{top10_share:.1f}%",
     "The scheme is concentrated in a few large states."),
    ("National Rs per beneficiary (range, Instalments 5-18)",
     f"{nat_money['Rs_per_Beneficiary'].min():,.0f} to {nat_money['Rs_per_Beneficiary'].max():,.0f}",
     "Slightly above the Rs 2,000 paid per instalment. The data does not say why."),
    ("Missing data",
     "Money missing for Instalments 1-4; beneficiaries missing for 6 states in some instalments",
     "Missing values were left blank (not filled with 0); a comparable-states series was used as a cross-check."),
]
key_findings = pd.DataFrame(findings, columns=["Finding", "Value", "What_it_means"])
key_findings.to_csv("p2_10_key_findings.csv", index=False)
print(key_findings.to_string(index=False))


# =============================================================================
# 12. PANDAS BASICS AND MISSING VALUES (syllabus Unit 3.2 demonstration)
# =============================================================================
def pandas_basics(state):
    pu.heading("Unit 3.2 Pandas basics (on a small sample)")
    small = state.loc[state["State"].isin(["Bihar", "Gujarat", "Kerala"]),
                      ["State", "Instalment", "Beneficiaries"]].head(6).copy()
    small["Lakh"] = small["Beneficiaries"] / 1e5             # add a column
    small = small.drop(columns="Lakh")                       # delete a column
    extra = pd.DataFrame({"State": ["Demo"], "Instalment": [0], "Beneficiaries": [1]})
    small = pd.concat([small, extra], ignore_index=True)     # add a row
    small = small.drop(index=small.index[-1])                # delete a row
    small.info()
    print("iloc first row :", small.iloc[0].to_dict())
    print("loc by label   :", small.loc[0, "State"])
    print("Filter (relational + logical operators):")
    print(state[(state["Instalment"] == 12) & (state["Beneficiaries"] > 5_000_000)]
          [["State", "Beneficiaries"]])


def missing_values_section(state):
    pu.heading("Unit 3.2 Missing values: determine, delete, impute (demo)")
    print("Missing per column:\n", state[["Beneficiaries", "Amount_Disbursed_Rs_Cr"]].isnull().sum())
    rows_left = state.dropna(subset=["Beneficiaries"]).shape[0]
    print("Rows left if we DELETE missing beneficiaries:", rows_left, "of", state.shape[0])
    demo = state.sort_values(["State", "Instalment"]).copy()
    demo["Ben_Imputed"] = demo.groupby("State")["Beneficiaries"].transform(
        lambda s: s.interpolate(limit=2, limit_area="inside"))        # imputation
    print("Cells filled by interpolation (demo only, NOT used in results):",
          int(demo["Beneficiaries"].isna().sum() - demo["Ben_Imputed"].isna().sum()))


pandas_basics(data)
missing_values_section(data)
