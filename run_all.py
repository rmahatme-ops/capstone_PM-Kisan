"""
Runs the whole PM-KISAN project in the correct order.
    python run_all.py
Graphs are saved as PNG files (no pop-up windows). To SEE the graphs in windows,
run   python data_visualization.py   on its own.
"""
import os
import subprocess
import sys

STEPS = [
    ("Part 1  Data cleaning .............. data_cleaning.py",     "data_cleaning.py"),
    ("Part 2  Data analysis (pandas) ..... data_analysis.py",     "data_analysis.py"),
    ("Part 3  Statistics (SciPy) ......... data_statistics.py",   "data_statistics.py"),
    ("Part 4  NumPy + time series ........ data_timeseries.py",   "data_timeseries.py"),
    ("Part 5  Visualization (8 graphs) ... data_visualization.py", "data_visualization.py"),
]

env = dict(os.environ, MPLBACKEND="Agg")       # save graphs without opening windows

for title, script in STEPS:
    print("Running", title)
    result = subprocess.run([sys.executable, script], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open("log_" + script.replace(".py", ".txt"), "w") as f:
        f.write(result.stdout)
    if result.returncode != 0:
        print("  FAILED - see", "log_" + script.replace(".py", ".txt"))
        print(result.stdout[-1500:])
        sys.exit(1)
    print("  done")

print("\nAll parts finished. Result tables are the p2_*.csv files; graphs are graph*.png.")
