"""
PART 3 - STATISTICS (SciPy)  : sharp drops, hypothesis tests, consistency with the national trend
Run AFTER data_analysis.py. Produces p2_11 ... p2_14 and p2_16 CSV files.
Syllabus: Unit 5.1 (SciPy stats), helper functions from pmkisan_utils (Unit 2.2).
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

import pmkisan_utils as pu          # our own module (keep it in the same folder)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


# =============================================================================
# SHARP DROPS
# =============================================================================
def sharp_drops(pivot, prev, change_masked):
    long = change_masked.reset_index().melt(id_vars="State", var_name="Instalment",
                                            value_name="Pct_Change")
    lp = prev.reset_index().melt(id_vars="State", var_name="Instalment", value_name="Previous")
    long = long.merge(lp, on=["State", "Instalment"])
    long["Instalment"] = long["Instalment"].astype(int)
    drops = (long[long["Pct_Change"] <= pu.DROP_LIMIT].sort_values("Pct_Change")
             .reset_index(drop=True).round(2))
    drops.to_csv("p2_13_sharp_drops.csv", index=False)
    print("Sharp drops (<= -10%, base >= 10,000):", len(drops))
    print(drops.head(8))
    print("Drops per instalment:", drops.groupby("Instalment").size().to_dict())
    return long, drops


# =============================================================================
# UNIT 5.1  SCIPY STATS: descriptive, normality, correlation, t-tests, ANOVA, non-parametric
# =============================================================================
def scipy_section(pivot, long, ok):
    pu.heading("Unit 5.1 SciPy stats")
    state_mean = ok.groupby("State")["Rs"].mean()
    size = pivot[18].reindex(state_mean.index)
    large = state_mean[size >= 1_000_000]
    small = state_mean[size < 1_000_000]

    # --- basic & descriptive statistics, rank
    d = stats.describe(state_mean)
    print("DESCRIBE Rs/beneficiary by state: n=%d mean=%.0f var=%.0f skew=%.2f"
          % (d.nobs, d.mean, d.variance, d.skewness))
    ranks = pd.Series(stats.rankdata(-state_mean), index=state_mean.index).sort_values()
    print("RANK (1 = highest Rs/beneficiary):", ranks.head(3).to_dict())

    # --- normality
    changes = long["Pct_Change"].dropna()
    sh = stats.shapiro(changes.sample(min(len(changes), 200), random_state=1))
    ks1 = stats.kstest(stats.zscore(changes), "norm")
    print("\nNORMALITY of %% changes: Shapiro p=%.2e | Kolmogorov-Smirnov (one sample) p=%.2e" % (sh.pvalue, ks1.pvalue))
    print("  -> p < 0.05 means NOT normal, so non-parametric tests are justified.")

    # --- homogeneity of variances
    groups = [g["Rs"].values for _, g in ok.groupby("FY")]
    lev = stats.levene(*groups)
    print("\nLEVENE (equal variances across financial years): p=%.2e" % lev.pvalue)

    # --- correlation
    both = pd.concat([size, state_mean], axis=1, keys=["Size", "Rs"]).dropna()
    pr, sp = stats.pearsonr(both["Size"], both["Rs"]), stats.spearmanr(both["Size"], both["Rs"])
    print("CORRELATION state size vs Rs/beneficiary: Pearson r=%.2f (p=%.2f) | Spearman rho=%.2f (p=%.2f)"
          % (pr[0], pr[1], sp[0], sp[1]))

    # --- chi-square: large vs small state x had a sharp drop at instalment 12
    x12 = long[long["Instalment"] == 12].dropna(subset=["Pct_Change"]).copy()
    x12["Size"] = np.where(pivot[18].reindex(x12["State"]).values >= 1_000_000, "Large", "Small")
    x12["Dropped"] = np.where(x12["Pct_Change"] <= pu.DROP_LIMIT, "Drop", "No drop")
    table = pd.crosstab(x12["Size"], x12["Dropped"])
    chi = stats.chi2_contingency(table)
    print("\nCHI-SQUARE state size vs sharp drop at instalment 12:\n", table)
    print("  chi2=%.2f p=%.3f" % (chi[0], chi[1]))

    # --- parametric: one-sample, independent, dependent t-tests, one-way ANOVA
    t1 = stats.ttest_1samp(state_mean, pu.NOMINAL_RS)
    print("\nONE-SAMPLE t-test (mean Rs/beneficiary vs Rs 2,000): mean=%.0f t=%.2f p=%.2e"
          % (state_mean.mean(), t1.statistic, t1.pvalue))
    t2 = stats.ttest_ind(large, small, equal_var=False)
    print("INDEPENDENT t-test (large vs small states): means %.0f vs %.0f, p=%.3f"
          % (large.mean(), small.mean(), t2.pvalue))
    before = ok[ok["Instalment"] < 12].groupby("State")["Rs"].mean()
    after = ok[ok["Instalment"] >= 12].groupby("State")["Rs"].mean()
    pair = pd.concat([before, after], axis=1, keys=["Before", "After"]).dropna()
    t3 = stats.ttest_rel(pair["After"], pair["Before"])
    print("DEPENDENT (paired) t-test before vs after instalment 12 (n=%d): %.0f -> %.0f, p=%.2e"
          % (len(pair), pair["Before"].mean(), pair["After"].mean(), t3.pvalue))
    an = stats.f_oneway(*groups)
    print("ONE-WAY ANOVA across financial years: F=%.1f p=%.2e" % (an.statistic, an.pvalue))
    lo, hi = pu.bootstrap_ci(pair["After"].tolist())
    print("Bootstrap 95%% CI of mean Rs after instalment 12: %.0f to %.0f" % (lo, hi))

    # --- non-parametric
    ks2 = stats.ks_2samp(ok[ok["Instalment"] < 12]["Rs"], ok[ok["Instalment"] >= 12]["Rs"])
    mw = stats.mannwhitneyu(large, small)
    w1 = stats.wilcoxon(pair["After"], pair["Before"])
    kw = stats.kruskal(*groups)
    x = x12["Pct_Change"]
    w12 = stats.wilcoxon(x, alternative="less")
    print("\nKOLMOGOROV-SMIRNOV two-sample (Rs before vs after 12): D=%.2f p=%.2e" % (ks2.statistic, ks2.pvalue))
    print("MANN-WHITNEY (large vs small states): p=%.3f" % mw.pvalue)
    print("WILCOXON dependent (before vs after 12): p=%.2e" % w1.pvalue)
    print("KRUSKAL-WALLIS across financial years: H=%.1f p=%.2e" % (kw.statistic, kw.pvalue))
    print("WILCOXON one-sided, %% change at instalment 12 (n=%d, median %.1f%%): p=%.2e"
          % (len(x), x.median(), w12.pvalue))


# =============================================================================
# CONSISTENCY OF STATES WITH THE NATIONAL TREND
# =============================================================================
def consistency(change_masked, nat):
    pu.heading("Consistency of states with the national trend")
    rows = []
    for s in change_masked.index:
        both = pd.DataFrame({"state": change_masked.loc[s], "nat": nat["Pct_Change"]}).dropna()
        both = both[both.index >= 6]
        if len(both) < 6:
            continue
        gaps = (both["state"] - both["nat"]).tolist()
        r = stats.pearsonr(both["state"], both["nat"])[0]
        rows.append({"State": s, "Correlation": round(r, 2), "RMSE_Gap_pp": round(pu.rmse(gaps), 2)})
    cons = pd.DataFrame(rows).sort_values("RMSE_Gap_pp").reset_index(drop=True)
    cons["Consistency"] = cons["RMSE_Gap_pp"].apply(pu.classify_consistency)
    cons.to_csv("p2_14_consistency_vs_national.csv", index=False)
    print(cons["Consistency"].value_counts().to_dict())
    print("Most consistent :", cons.head(4)["State"].tolist())
    print("Least consistent:", cons.tail(4)["State"].tolist())


# =============================================================================
# MAIN
# =============================================================================
def main():
    state = pu.load_csv(pu.STATE_FILE)
    nat = pu.one_row_per_instalment(pu.load_csv(pu.NATIONAL_FILE))
    nat.to_csv("p2_16_national_one_row_per_instalment.csv")

    pivot = pu.build_pivot(state)
    prev, change_masked, comp_change = pu.change_tables(pivot)
    pu.heading("State % change table")
    print("Largest 5 states, last instalment:", pu.top_states(pivot))
    print("Fall at instalment 12 -> complete states %.2f%% | national total %.2f%%"
          % (comp_change[12], nat.loc[12, "Pct_Change"]))

    long, drops = sharp_drops(pivot, prev, change_masked)

    money, ok = pu.build_money(state)
    pu.heading("Disbursement outliers")
    print("Outlier rows (amount and count do not match):", int(money["Outlier"].sum()))

    scipy_section(pivot, long, ok)
    consistency(change_masked, nat)


if __name__ == "__main__":
    main()
