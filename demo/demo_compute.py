import pandas as pd
from matplotlib import pyplot as plt
 
fig, axs = plt.subplots(1, 2, figsize = (24, 8), constrained_layout=True)
 
axs[0].bar([0, 1, 2, 3, 4, 5], [25610152, 49604521, 46433192, 28781481, 46703192, 25880152], align='center')
axs[0].set_xticks([0, 1, 2, 3, 4, 5], ('ERM', 'SNGP', 'DNN-SN', 'DNN-GP', 'SN-Density-Softmax', 'Density-Softmax'))
axs[0].set_ylabel('Model weights (Million)', fontsize=20)
axs[0].set_title('Storage comparision with Resnet-50 on ImageNet', fontsize=20)

axs[1].bar([0, 1, 2, 3, 4, 5], [299.81, 545.70, 468.12, 400.30, 377.47, 299.90], yerr=[1.2, 3.3, 3.1, 3.2, 1.5, 1.2], align='center', ecolor='black', capsize=10)
axs[1].set_xticks([0, 1, 2, 3, 4, 5], ('ERM', 'SNGP', 'DNN-SN', 'DNN-GP', 'SN-Density-Softmax', 'Density-Softmax'))
axs[1].set_ylabel('Inference speed (milliseconds/sample)', fontsize=20)
axs[1].set_title('Inference speed comparision with Resnet-50 on ImageNet', fontsize=20)

plt.savefig("demo_scalability.pdf")