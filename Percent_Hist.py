import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("average_tally_110.csv")

percent_col = "Average Percentage of State 0 Cells"
count_col = "Count"

plt.figure()
plt.bar(df[percent_col], df[count_col])

# Log scale on Y axis
plt.yscale('log')

plt.xlabel("Percentage")
plt.ylabel("Count (log scale)")
plt.title("Counts vs Percentage (Log Scale)")
plt.show()