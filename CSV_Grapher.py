import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("total_feature_size_distribution 110.csv")

sizes = df["Feature Size"].to_numpy()
counts = df["Count"].to_numpy(dtype=np.int64)

ccdf = np.cumsum(counts[::-1])[::-1]

plt.figure()
plt.scatter(sizes, ccdf)
plt.plot(sizes, ccdf)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Feature Size")
plt.ylabel("Number of Features ≥ Size")
plt.title("CCDF")

plt.grid(True, which="both")
plt.show()