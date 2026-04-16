import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_csv("total_feature_size_distribution 110.csv")
df = pd.read_csv("average_tally_110.csv")

#sizes = df["Feature Size"].to_numpy()
percents = df["Average Percentage of State 0 Cells"].to_numpy(dtype=np.float64)
counts = df["Count"].to_numpy(dtype=np.int64)

ccdf = np.cumsum(counts[::-1])[::-1]

plt.figure()
#plt.scatter(sizes, ccdf)
#plt.plot(sizes, ccdf)
plt.scatter(percents, ccdf)
plt.plot(percents, ccdf)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Feature Size")
plt.ylabel("Number of Features ≥ Size")
plt.title("CCDF")

plt.grid(True, which="both")
plt.show()