import pandas as pd
from matplotlib import pyplot as plt
 
fig, axs = plt.subplots(1, 2, figsize = (39, 13), constrained_layout=True)
 
axs[0].bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.8*100/60, 1.8*100/60, 6*100/60, 10.95*100/60, 3.08*100/60, 3.61*100/60, 3.9*100/60, 7.5*100/60, 8.1*100/60, 1.8*4*100/60, 1.8*3*100/60], yerr=[0.1, 0.11, 0.15, 0.16, 0.10, 0.11, 0.11, 0.16, 0.16, 0.15, 0.15], align='center', color=["tab:blue", "tab:orange", "tab:green", "tab:red", "#bcbd22", "#17becf", "tab:purple", "#7f7f7f", "tab:brown", "tab:pink", "blue"])
axs[0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ("ERM", "MC Dropout", "MFVI BNN", "Rank-1 BNN", "Posterior Net", "Heteroscedastic", "SNGP", "MIMO", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"))
axs[0].set_ylabel('Training cost (hours)', fontsize=39)
axs[0].set_title('Training cost comparision', fontsize=39)

axs[1].bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [518.12, 4319.01, 809.01, 2027.67, 1162.48, 560.43, 916.26, 701.66, 1498.01, 1520.34, 520.34], yerr=[10, 41, 17, 32, 18, 11, 16, 12, 25, 26, 11], align='center', color=["tab:blue", "tab:orange", "tab:green", "tab:red", "#bcbd22", "#17becf", "tab:purple", "#7f7f7f", "tab:brown", "tab:pink", "blue"])
axs[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ("ERM", "MC Dropout", "MFVI BNN", "Rank-1 BNN", "Posterior Net", "Heteroscedastic", "SNGP", "MIMO", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"))
axs[1].set_ylabel('Inference cost (milliseconds/sample)', fontsize=39)
axs[1].set_title('Inference cost comparision', fontsize=39)

# Show Plot
plt.savefig("train_test_barchart.pdf")

# axs[0].bar([0, 1, 2, 3, 4, 5, 6, 7, 8], [25.61, 25.61, 26.35, 58.39, 26.60, 27.67, 25.82, 102.44, 25.88], align='center', color=["tab:blue", "tab:orange", "tab:red", "#17becf", "tab:purple", "#7f7f7f", "tab:brown", "tab:pink", "blue"])
# axs[0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ("ERM", "MC Dropout", "Rank-1 BNN", "Heteroscedastic", "SNGP", "MIMO", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"))
# axs[0].set_ylabel('Model weights (Million)', fontsize=39)
# axs[0].set_title('Storage comparision', fontsize=39)

# axs[1].bar([0, 1, 2, 3, 4, 5, 6, 7, 8], [299.81, 601.15, 990.14, 337.50, 606.11, 367.17, 696.81, 701.34, 299.90], yerr=[5, 6, 18, 5, 10, 6, 13, 14, 5], align='center', ecolor='black', capsize=10, color=["tab:blue", "tab:orange", "tab:red", "#17becf", "tab:purple", "#7f7f7f", "tab:brown", "tab:pink", "blue"])
# axs[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ("ERM", "MC Dropout", "Rank-1 BNN", "Heteroscedastic", "SNGP", "MIMO", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"))
# axs[1].set_ylabel('Inference cost (milliseconds/sample)', fontsize=39)
# axs[1].set_title('Inference cost comparision', fontsize=39)

# # Show Plot
# plt.savefig("computational_barchart.pdf")