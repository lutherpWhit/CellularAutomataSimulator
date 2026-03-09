import numpy as np
import matplotlib.pyplot as plt

sizes = np.array([1,3,6,10,15,21,28,36,45,55,66,78])

counts = np.array([
    1747909344,
    1531822,
    1845065935,
    806772,
    672096,
    812867,
    62521,
    9415,
    6,
    9380,
    4,
    1
], dtype=np.int64)

# CCDF calculation
ccdf = np.cumsum(counts[::-1])[::-1]

for s, c in zip(sizes, ccdf):
    print(f"Size ≥ {s}: {c} features")

plt.figure()

plt.scatter(sizes, ccdf)
plt.plot(sizes, ccdf)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Feature Size")
plt.ylabel("Number of Features ≥ Size")
plt.title("Complementary Cumulative Distribution (CCDF)")

plt.grid(True, which="both")
plt.show()