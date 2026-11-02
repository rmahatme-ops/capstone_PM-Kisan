# PM-KISAN: State-wise, Instalment-wise Analysis (2018-2024)
Capstone Mini Project, CSBS (Lab CA)

## How to run
```
pip install pandas numpy scipy statsmodels matplotlib seaborn
python run_all.py            # runs everything in order, saves tables (p2_*.csv) and graphs (graph*.png)
```
Or run the scripts one by one, in this order (needed for the viva demonstrations):

| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `data_cleaning.py` | Load, check problems, clean, before/after table | `pmkisan_cleaned_state_records.csv`, `pmkisan_cleaned_national_records.csv` |
| 2 | `data_analysis.py` | groupby, pivot_table, % change, rupees per beneficiary, key findings, pandas basics | `p2_01` ... `p2_10` |
| 3 | `data_statistics.py` | State % change table, sharp drops, SciPy tests, consistency with national trend | `p2_11` ... `p2_14`, `p2_16` |
| 4 | `data_timeseries.py` | NumPy on the national series, stationarity, smoothing, decomposition, ARIMA forecast | `p2_15` |
| 5 | `data_visualization.py` | 8 graphs | `graph1` ... `graph8` |
| - | `pmkisan_utils.py` | Our own module: helper functions and constants used by the scripts | - |
| - | `run_all.py` | Runs steps 1-5 in order | logs |

Input: `pmkisan_master_merged.csv` (four parliamentary-reply tables stacked together). Add the four source links here before submission: [links].

## Project objectives and where they are answered
| Objective | Code | Graph |
|---|---|---|
| Reach across states and time | `data_analysis.py` | Graph 1, Graph 2 |
| States with sharp drops between instalments | `data_statistics.py` (sharp_drops) | Graph 3 |
| Average disbursement per beneficiary | `data_analysis.py`, `data_statistics.py` | Graph 6, Graph 7 |
| Benchmark against the national trend | `data_statistics.py` (consistency) | Graph 4, Graph 5 |
| Forecast | `data_timeseries.py` | Graph 8 |

## Rubric mapping
| Rubric item | Where |
|---|---|
| Problem definition and dataset | this README, `data_cleaning.py` |
| Code structure and modularity | separate scripts, functions, `pmkisan_utils.py` (user-defined module), `run_all.py` |
| Cleaning and manipulation (NumPy/Pandas) | `data_cleaning.py`, `data_analysis.py`, `data_timeseries.py` (NumPy) |
| Visualization (Matplotlib/Seaborn) | `data_visualization.py` |
| Statistics and forecasting (SciPy/Statsmodels) | `data_statistics.py`, `data_timeseries.py` |

## Syllabus mapping
| Unit | Topics used | Where |
|---|---|---|
| 1 Python | variables, dictionaries, loops, if/elif/else, break/continue/pass, exceptions, functions, recursion | `pmkisan_utils.py` |
| 2 Modules | math, random, statistics, array, string functions, re, datetime, os, own module | `pmkisan_utils.py` |
| 3 NumPy and Pandas | arrays, groupby, pivot_table, loc/iloc, missing values | `data_timeseries.py`, `data_analysis.py` |
| 4 Matplotlib and Seaborn | line, bar, box plot, heat map, subplots | `data_visualization.py` |
| 5 SciPy and Statsmodels | descriptive stats, normality, t-tests, ANOVA, chi-square, Wilcoxon, Kruskal-Wallis, ADF, ARIMA | `data_statistics.py`, `data_timeseries.py` |

## Key decisions (be ready to explain)
- Instalment number comes from the ordinal in the label, otherwise from the season and year.
- Overlapping sources: the newest and most complete table wins (AU1297, then AU321, then AU795, then AU520).
- Missing values are left blank, not filled with 0. Amounts exist only for instalments 5-18.
- Sharp drop = fall of 10% or more on a base of at least 10,000 beneficiaries.
- "Comparable states" = the 30 states with data in all 18 instalments.
- Rupees per beneficiary: rows outside Rs 1,000-3,000 are treated as outliers (count and amount do not match) and excluded from tests; state ranking uses the median.

## Limitations
- No amounts for instalments 1-4.
- 45 rows where beneficiary count and amount do not match.
- Sources disagree for a few small states around instalments 11-13.
- Forecast uses only 18 points with a break at instalment 12, so it is indicative.
- The analysis shows that the drop at instalment 12 happened and was broad; it does not establish the cause.
