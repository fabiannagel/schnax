import numpy as np

indices = np.empty(shape=(96, 48), dtype=np.int)
for i, row in enumerate(indices):
    indices[i] = np.arange(1, 48+1)

offset = np.ones(shape=(96, 48, 3), dtype=np.int)
new_offsets = np.empty(shape=(96, 48, 3), dtype=np.int)

for idx in indices:




