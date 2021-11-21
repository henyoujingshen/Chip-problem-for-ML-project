# Simple program for Latin Hypervolume Sampling (LHS)
# Use "pip install lhsmdu" to install the package.
import lhsmdu
import numpy as np

k = np.array(lhsmdu.sample(numDimensions=5,
                           numSamples=45))
scale = (5 - 1) * np.array([30, 5, 25, 100, 1]).reshape(-1, 1)
offset = np.array([200, 20, 200, 550, 8]).reshape(-1, 1)
k = np.multiply(scale, k) + offset
k = np.rint(k)
print(k)
