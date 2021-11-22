# Simple program for Latin Hypervolume Sampling (LHS)
# Use "pip install lhsmdu" to install the package.
import lhsmdu
import numpy as np

num_dimensions = 5
num_samples = 45
k = np.array(lhsmdu.sample(numDimensions=num_dimensions,
                           numSamples=num_samples))
scale = np.array([30, 5, 25, 100, 1]).reshape(-1, 1)
offset = np.array([200, 20, 200, 550, 8]).reshape(-1, 1)
k = np.multiply(scale, np.rint((num_dimensions - 1) * k)) + offset
print(k)
