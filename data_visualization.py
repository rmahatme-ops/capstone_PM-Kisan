
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def finish(filename):
    """Tidy layout, save the picture and show it."""
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


pct = pd.read_csv("p2_06_pct_change_national.csv")
pivot = pd.read_csv("p2_05_pivot_state_by_instalment.csv", index_col="State")
pivot.columns = pivot.columns.astype(int)               # column names become numbers

change = pd.read_csv("p2_11_state_pct_change_by_instalment.csv", index_col="State")
change.columns = change.columns.astype(int)
nat_complete = pd.read_csv("p2_12_national_pct_change_complete_states.csv", index_col="Instalment")
drops = pd.read_csv("p2_13_sharp_drops.csv")
cons = pd.read_csv("p2_14_consistency_vs_national.csv")
fc = pd.read_csv("p2_15_forecast_arima.csv")
nat_one = pd.read_csv("p2_16_national_one_row_per_instalment.csv", index_col="Instalment")
rs_state = pd.read_csv("p2_09_disbursement_per_beneficiary_by_state.csv")
sel = pd.read_csv("p2_01_selected_columns.csv")

# GRAPH 1: Line graph - total beneficiaries across instalments
plt.figure(figsize=(11, 5.5))
plt.plot(pct["Instalment"], pct["Total_All_States"] / 1e6,
         marker="o", linewidth=2, color="#2a78d6", label="All states (as reported)")
plt.plot(pct["Instalment"], pct["Total_Comparable_States"] / 1e6,
         marker="s", linewidth=2, linestyle="--", color="#eb6834", label="30 states with complete data")
plt.title("PM-KISAN: Total Beneficiaries per Instalment", fontsize=14, fontweight="bold")
plt.xlabel("Instalment number")
plt.ylabel("Beneficiaries (millions)")
plt.xticks(range(1, 19))
plt.ylim(0, 120)
plt.grid(True, alpha=0.3)
plt.legend()
finish("graph1_national_trend.png")

# GRAPH 2: Heat map - % change vs previous instalment, every state
hm = change.copy()
hm.loc["NATIONAL (30 complete states)"] = nat_complete["Pct_Change"]
order = pivot[18].sort_values(ascending=False).index.tolist()      # largest state first
hm = hm.loc[["NATIONAL (30 complete states)"] + order]
hm = hm.drop(columns=1).dropna(how="all")                           # instalment 1 has no previous

plt.figure(figsize=(12, 11))
ax = sns.heatmap(hm, cmap="RdYlGn", center=0, vmin=-40, vmax=40,
                 linewidths=0.3, linecolor="white",
                 cbar_kws={"label": "% change vs previous instalment (capped at +/-40)"})
ax.axvline(10, color="black", linewidth=2)       # the two lines frame Instalment 12
ax.axvline(11, color="black", linewidth=2)
ax.set_title("Change in Beneficiaries Between Instalments, by State\n"
             "Red = fall, green = rise, blank = missing data or under 10,000 farmers",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Instalment number")
ax.set_ylabel("")
finish("graph2_heatmap_pct_change.png")

# GRAPH 3: Bar chart - sharpest drops (state and instalment)
top_drops = drops.head(14).iloc[::-1]
labels = [f"{s} (I{i})" for s, i in zip(top_drops["State"], top_drops["Instalment"])]
cols = ["#eb6834" if i == 12 else "#c0392b" for i in top_drops["Instalment"]]

plt.figure(figsize=(10, 6.5))
plt.barh(labels, top_drops["Pct_Change"], color=cols, height=0.7)
for y, v in enumerate(top_drops["Pct_Change"]):
    plt.text(v - 1, y, f"{v:.0f}%", va="center", ha="right", fontsize=9)
plt.xlim(top_drops["Pct_Change"].min() * 1.2, 0)
plt.title("Sharpest Drops in Beneficiaries (fall of 10%+ on a base of 10,000+)\n"
          "Orange = Instalment 12", fontsize=13, fontweight="bold")
plt.xlabel("% change vs previous instalment")
plt.grid(True, axis="x", alpha=0.3)
finish("graph3_sharpest_drops.png")

# GRAPH 4: Six largest states vs the national % change
six = pivot[18].sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
for ax, st in zip(axes.flat, six):
    ax.plot(change.columns[5:], change.loc[st].iloc[5:], color="#1baf7a", linewidth=2, label="State")
    ax.plot(nat_one.index[5:], nat_one["Pct_Change"].iloc[5:], color="grey",
            linewidth=2, linestyle="--", label="National")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(st, fontsize=11)
    ax.set_ylim(-40, 40)
    ax.grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=8)
fig.supxlabel("Instalment number")
fig.supylabel("% change vs previous instalment")
fig.suptitle("Large States vs the National Trend", fontsize=14, fontweight="bold")
finish("graph4_states_vs_national.png")

# GRAPH 5: Consistency ranking of all states
c = cons.copy()
c["RMSE_plot"] = c["RMSE_Gap_pp"].clip(upper=40)
c = c.sort_values("RMSE_plot", ascending=False)
palette = {"High": "#1baf7a", "Moderate": "#f5b800", "Low": "#c0392b"}

plt.figure(figsize=(9, 8))
plt.barh(c["State"], c["RMSE_plot"], color=[palette[x] for x in c["Consistency"]])
plt.axvline(5, color="grey", linestyle=":")
plt.axvline(12, color="grey", linestyle=":")
plt.title("How Closely Each State Follows the National Trend\n(lower = more consistent)",
          fontsize=13, fontweight="bold")
plt.xlabel("Typical gap from national % change (percentage points, capped at 40)")
plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in palette.values()],
           labels=list(palette.keys()), loc="upper right")
plt.tick_params(axis="y", labelsize=8)
finish("graph5_consistency.png")

# GRAPH 6: Rs per beneficiary by state (top 15, median)
top = rs_state.head(15).iloc[::-1]

plt.figure(figsize=(11, 7))
plt.barh(top["State"], top["Avg_Rs_per_Beneficiary"], color="#2a78d6", height=0.65)
plt.axvline(2000, color="#eb6834", linestyle="--", linewidth=2, label="Rs 2,000 (one instalment)")
for i, v in enumerate(top["Avg_Rs_per_Beneficiary"]):
    plt.text(v + 30, i, f"{v:,.0f}", va="center", fontsize=9)
plt.title("Median Rs per Beneficiary per Instalment (top 15 states, Instalments 5-18)",
          fontsize=13, fontweight="bold")
plt.xlabel("Rupees per beneficiary")
plt.ylabel("State")
plt.xlim(0, 3600)
plt.grid(True, axis="x", alpha=0.3)
plt.legend(loc="lower right")
finish("graph6_rs_per_beneficiary_by_state.png")

# GRAPH 7: Box plot - Rs per beneficiary by financial year
box = sel.dropna(subset=["Avg_Rs_Per_Beneficiary"]).copy()
box = box[(box["Beneficiaries"] >= 10000) & box["Avg_Rs_Per_Beneficiary"].between(1000, 3000)]
box["Financial_Year"] = box["Instalment"].apply(
    lambda n: f"{2020 + (n - 5) // 3}-{str(2021 + (n - 5) // 3)[-2:]}")

plt.figure(figsize=(10, 5.5))
sns.boxplot(data=box, x="Financial_Year", y="Avg_Rs_Per_Beneficiary", color="#9ecbf5")
plt.axhline(2000, color="#eb6834", linestyle="--", linewidth=2, label="Rs 2,000 (one instalment)")
plt.title("Rs per Beneficiary per Instalment by Financial Year (all states)",
          fontsize=13, fontweight="bold")
plt.xlabel("Financial year")
plt.ylabel("Rupees per beneficiary")
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
finish("graph7_rs_by_financial_year.png")

# GRAPH 8: National series with ARIMA forecast
hist = nat_one["Beneficiaries"] / 1e6
plt.figure(figsize=(11, 5.5))
plt.plot(hist.index, hist.values, marker="o", linewidth=2, color="#2a78d6", label="Reported")
plt.plot([18] + fc["Instalment"].tolist(), [hist.iloc[-1]] + fc["Forecast_million"].tolist(),
         marker="s", linestyle="--", color="#eb6834", label="ARIMA forecast")
plt.fill_between(fc["Instalment"], fc["Low_95"], fc["High_95"], color="#eb6834", alpha=0.15,
                 label="95% range")
plt.axvline(12, color="grey", linestyle=":")
plt.title("National Beneficiaries with 3-Instalment Forecast (indicative: only 18 data points)",
          fontsize=13, fontweight="bold")
plt.xlabel("Instalment number")
plt.ylabel("Beneficiaries (millions)")
plt.xticks(range(1, 22))
plt.grid(True, alpha=0.3)
plt.legend()
finish("graph8_forecast.png")
