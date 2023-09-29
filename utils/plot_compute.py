import matplotlib.pyplot as plt
x = ["NVIDIA Tesla K80", "NVIDIA RTX A5000", "NVIDIA A100"]

plt.errorbar(x, [3130/5, 299.81, 260.37], yerr=[20, 10, 8], capsize=5, label="ERM", ecolor = "tab:blue", color = "tab:blue")
plt.errorbar(x, [10069/5, 601.15, 580.40], yerr=[20, 10, 8], capsize=5, label="MC Dropout", ecolor = "tab:orange", color = "tab:orange")
plt.errorbar(x, [12520/5, 990.14, 920.68], yerr=[20, 10, 8], capsize=5, label="Rank-1 BNN", ecolor = "tab:red", color = "tab:red")
plt.errorbar(x, [4023/5, 337.50, 300.07], yerr=[20, 10, 8], capsize=5, label="Heteroscedastic", ecolor = "#17becf", color = "#17becf")
plt.errorbar(x, [10306/5, 606.11, 590.10], yerr=[20, 10, 8], capsize=5, label="SNGP", ecolor = "tab:purple", color = "tab:purple")
plt.errorbar(x, [4533/5, 367.17, 340.12], yerr=[20, 10, 8], capsize=5, label="MIMO", ecolor = "#7f7f7f", color = "#7f7f7f")
plt.errorbar(x, [11219/5, 696.81, 670.04], yerr=[20, 10, 8], capsize=5, label="BatchEnsemble", ecolor = "tab:brown", color = "tab:brown")
plt.errorbar(x, [12485/5, 701.34, 680.76], yerr=[20, 10, 8], capsize=5, label="Deep Ensembles", ecolor = "tab:pink", color = "tab:pink")
plt.errorbar(x, [3321/5, 299.90, 270.22], yerr=[20, 10, 8], capsize=5, label="Density-Softmax", ecolor = "blue", color = "blue")
plt.legend()
plt.ylabel('Inference cost (milliseconds/sample)')
# plt.title('Inference cost comparision', fontsize = 15)
plt.tight_layout()
plt.savefig("ebar_speed.pdf")