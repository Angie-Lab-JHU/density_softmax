import numpy as np

with open('out.npy', 'rb') as f:
	cNLL = np.load(f)
	cAcc = np.load(f)
	cECE = np.load(f)
print(len(cNLL))
print(np.mean(cNLL))
print(np.mean(cAcc))
print(np.mean(cECE))