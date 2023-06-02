import pandas as pd
from matplotlib import pyplot as plt
 
fig, axs = plt.subplots(1, 2, figsize = (24, 8), constrained_layout=True)
 
axs[0].bar([0, 1, 2, 3, 4, 5, 6, 7], [36.50, 36.50, 72.96, 36.65, 44.39, 36.58, 145.99, 36.58], align='center', color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "blue"])
axs[0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7], ("ERM", "MC Dropout", "MFVI BNN", "Rank-1 BNN", "SNGP", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"))
axs[0].set_ylabel('Model weights (Million)', fontsize=20)
axs[0].set_title('Storage comparision', fontsize=20)

axs[1].bar([0, 1, 2, 3, 4, 5, 6, 7], [518.12, 1551.34, 2564.58, 2388.08, 1107.68, 1498.01, 1520.34, 520.53], yerr=[2.2, 3.3, 4.1, 4.2, 2.2, 2.9, 3.1, 2.2], align='center', ecolor='black', capsize=10, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "blue"])
axs[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7], ("ERM", "MC Dropout", "MFVI BNN", "Rank-1 BNN", "SNGP", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"))
axs[1].set_ylabel('Inference speed (milliseconds/sample)', fontsize=20)
axs[1].set_title('Inference speed comparision', fontsize=20)

# Show Plot
plt.savefig("computational_barchart.png")