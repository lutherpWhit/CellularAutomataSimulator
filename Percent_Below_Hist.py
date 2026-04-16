import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("average_tally_110.csv")

percent_col = "Average Percentage of State 0 Cells"
count_col = "Count"

# Sort by percentage (ascending)
df = df.sort_values(by=percent_col)

# Compute reverse cumulative sum (≥ x)
df["ccdf_count"] = df[count_col][::-1].cumsum()[::-1]

# Plot
plt.figure()
plt.plot(df[percent_col], df["ccdf_count"], marker='o')

# Log scale on Y axis
plt.yscale('log')

plt.xlabel("Percentage")
plt.ylabel("Count ≥ x (log scale)")
plt.title("Complementary Cumulative Distribution (CCDF)")

plt.show()