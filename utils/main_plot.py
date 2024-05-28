import numpy as np
import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 2, figsize = (24, 8), constrained_layout=True)

ERM_10 = [0.159, 96.0, 0.023, 1.05, 76.1, 0.153, 0.40, 89.9, 0.064, 0.781, 0.835, 36.50, 518.12]
Dropout_10 = [0.148, 95.9, 0.020, 1.05, 75.6, 0.150, 0.39, 89.9, 0.058, 0.971, 0.832, 36.50, 1551.34]
MFVI_10 = [0.208, 95.0, 0.027, 1.58, 71.0, 0.183, 0.49, 88.1, 0.070, 0.780, 0.828, 72.96, 2564.58]
Rank_1_10 = [0.128, 96.3, 0.008, 0.84, 76.7, 0.080, 0.32, 90.4, 0.033, 0.963, 0.885, 36.65, 2388.08]
ENN_10 = [0.288, 91.5, 0.071, 1.10, 73.4, 0.149, 0.43, 87.6, 0.062, 0.784, 0.830, 36.51, 958.59]
Posterior_10 = [0.360, 93.1, 0.112, 1.06, 75.2, 0.139, 0.42, 87.9, 0.053, 0.893, 0.812, 36.51, 1162.48]
DUQ_10 = [0.239, 94.7, 0.034, 1.35, 71.6, 0.183, 0.49, 87.9, 0.068, 0.973, 0.854, 40.61, 1538.35]
DDU_10 = [0.159, 96.0, 0.024, 1.06, 76.0, 0.153, 0.39, 89.8, 0.063, 0.986, 0.887, 40.61, 1354.31]
NUQ_10 = [0.301, 92.0, 0.106, 1.72, 73.2, 0.188, 0.50, 87.6, 0.068, 0.702, 0.810, 68.50, 1614.67]
DNN_GP_10 = [0.221, 95.9, 0.029, 1.38, 71.7, 0.175, 0.56, 89.8, 0.081, 0.976, 0.887, 39.25, 988.94]
SNGP_10 = [0.138, 95.9, 0.018, 0.86, 75.6, 0.090, 0.43, 89.7, 0.064, 0.990, 0.905, 44.39, 1107.68]
BatchEnsemble_10 = [0.136, 96.3, 0.018, 0.97, 77.8, 0.124, 0.35, 90.6, 0.048, 0.897, 0.801, 36.58, 1498.01]
Ensembles_10 = [0.114, 96.6, 0.010, 0.81, 77.9, 0.087, 0.28, 92.2, 0.025, 0.964, 0.888, 145.99, 1520.34]
Density_10 = [0.140, 96.2, 0.015, 0.79, 77.0, 0.086, 0.33, 90.2, 0.015, 0.972, 0.890, 36.58, 520.53]

def get_NLL_10():
    def m(matrix):
        return np.mean([matrix[0], matrix[3], matrix[6]])
    return ([m(ERM_10), m(Dropout_10), m(MFVI_10), m(Rank_1_10),  m(ENN_10),
        m(Posterior_10), m(DUQ_10), m(DDU_10), m(NUQ_10), m(DNN_GP_10), 
        m(SNGP_10), m(BatchEnsemble_10), m(Ensembles_10), m(Density_10)])

def get_Acc10():
    def m(matrix):
        return np.mean([matrix[1], matrix[4], matrix[7]])
    return ([m(ERM_10), m(Dropout_10), m(MFVI_10), m(Rank_1_10),  m(ENN_10),
        m(Posterior_10), m(DUQ_10), m(DDU_10), m(NUQ_10), m(DNN_GP_10), 
        m(SNGP_10), m(BatchEnsemble_10), m(Ensembles_10), m(Density_10)])

def get_ECE10():
    def m(matrix):
        return np.mean([matrix[2], matrix[5], matrix[8]])
    return ([m(ERM_10), m(Dropout_10), m(MFVI_10), m(Rank_1_10),  m(ENN_10),
        m(Posterior_10), m(DUQ_10), m(DDU_10), m(NUQ_10), m(DNN_GP_10), 
        m(SNGP_10), m(BatchEnsemble_10), m(Ensembles_10), m(Density_10)])

def get_AUPR10():
    def m(matrix):
        return np.mean([matrix[9], matrix[10]])
    return ([m(ERM_10), m(Dropout_10), m(MFVI_10), m(Rank_1_10),  m(ENN_10),
        m(Posterior_10), m(DUQ_10), m(DDU_10), m(NUQ_10), m(DNN_GP_10), 
        m(SNGP_10), m(BatchEnsemble_10), m(Ensembles_10), m(Density_10)])

def get_Par10():
    def m(matrix):
        return matrix[11]
    return ([m(ERM_10), m(Dropout_10), m(MFVI_10), m(Rank_1_10),  m(ENN_10),
        m(Posterior_10), m(DUQ_10), m(DDU_10), m(NUQ_10), m(DNN_GP_10), 
        m(SNGP_10), m(BatchEnsemble_10), m(Ensembles_10), m(Density_10)])

def get_Lat10():
    def m(matrix):
        return matrix[12]
    return ([m(ERM_10), m(Dropout_10), m(MFVI_10), m(Rank_1_10),  m(ENN_10),
        m(Posterior_10), m(DUQ_10), m(DDU_10), m(NUQ_10), m(DNN_GP_10), 
        m(SNGP_10), m(BatchEnsemble_10), m(Ensembles_10), m(Density_10)])

ERM_100 = [0.875, 79.8, 0.085, 2.70, 51.3, 0.239, 0.882, 0.745, 36.55, 521.15]
Dropout_100 = [0.797, 79.6, 0.050, 2.43, 51.5, 0.188, 0.832, 0.757, 36.55, 1562.39]
MFVI_100 = [0.933, 77.3, 0.094, 3.15, 48.0, 0.283, 0.882, 0.748, 73.07, 2588.58]
Rank_1_100 = [0.692, 81.3, 0.018, 2.24, 53.8, 0.117, 0.884, 0.797, 36.71, 2402.04]
Posterior_100 = [2.021, 77.3, 0.391, 3.12, 48.3, 0.281, 0.880, 0.760, 36.56, 1190.87]
DUQ_100 = [0.980, 78.5, 0.119, 2.84, 50.4, 0.281, 0.878, 0.732, 77.58, 1547.35]
DDU_100 = [0.877, 79.7, 0.086, 2.70, 51.3, 0.240, 0.890, 0.797, 77.58, 1359.25]
DNN_GP_100 = [0.885, 79.2, 0.064, 2.63, 47.7, 0.166, 0.876, 0.746, 39.35, 997.42]
SNGP_100 = [0.847, 79.9, 0.025, 2.53, 50.0, 0.117, 0.923, 0.801, 44.48, 1141.17]
BatchEnsemble_100 = [0.690, 81.9, 0.027, 2.56, 51.3, 0.149, 0.870, 0.757, 36.63, 1568.77]
Ensembles_100 = [0.666, 82.7, 0.021, 2.27, 54.1, 0.138, 0.888, 0.780, 146.22, 1569.23]
Density_100 = [0.780, 81.2, 0.038, 2.20, 52.4, 0.102, 0.894, 0.801, 36.64, 522.94]

def get_NLL_100():
    def m(matrix):
        return np.mean([matrix[0], matrix[3]])
    return ([m(ERM_100), m(Dropout_100), m(MFVI_100), m(Rank_1_100),  
        m(Posterior_100), m(DUQ_100), m(DDU_100),  m(DNN_GP_100), 
        m(SNGP_100), m(BatchEnsemble_100), m(Ensembles_100), m(Density_100)])

def get_Acc100():
    def m(matrix):
        return np.mean([matrix[1], matrix[4]])
    return ([m(ERM_100), m(Dropout_100), m(MFVI_100), m(Rank_1_100),  
        m(Posterior_100), m(DUQ_100), m(DDU_100),  m(DNN_GP_100), 
        m(SNGP_100), m(BatchEnsemble_100), m(Ensembles_100), m(Density_100)])

def get_ECE100():
    def m(matrix):
        return np.mean([matrix[2], matrix[5]])
    return ([m(ERM_100), m(Dropout_100), m(MFVI_100), m(Rank_1_100),  
        m(Posterior_100), m(DUQ_100), m(DDU_100),  m(DNN_GP_100), 
        m(SNGP_100), m(BatchEnsemble_100), m(Ensembles_100), m(Density_100)])

def get_AUPR100():
    def m(matrix):
        return np.mean([matrix[6], matrix[7]])
    return ([m(ERM_100), m(Dropout_100), m(MFVI_100), m(Rank_1_100),  
        m(Posterior_100), m(DUQ_100), m(DDU_100),  m(DNN_GP_100), 
        m(SNGP_100), m(BatchEnsemble_100), m(Ensembles_100), m(Density_100)])

def get_Par100():
    def m(matrix):
        return matrix[8]
    return ([m(ERM_100), m(Dropout_100), m(MFVI_100), m(Rank_1_100),  
        m(Posterior_100), m(DUQ_100), m(DDU_100),  m(DNN_GP_100), 
        m(SNGP_100), m(BatchEnsemble_100), m(Ensembles_100), m(Density_100)])

def get_Lat100():
    def m(matrix):
        return matrix[9]
    return ([m(ERM_100), m(Dropout_100), m(MFVI_100), m(Rank_1_100),  
        m(Posterior_100), m(DUQ_100), m(DDU_100),  m(DNN_GP_100), 
        m(SNGP_100), m(BatchEnsemble_100), m(Ensembles_100), m(Density_100)])

fig = plt.figure(figsize = (28, 26))
plt.subplots_adjust(left=-0.1,
                    bottom=-0.1,
                    right=0.98,
                    top=1.15,
                    wspace=0.05,
                    hspace=-0.5)

plt.rcParams['font.size'] = '24'

def plot_scatter(ax, x, y, z, colors, labels, zlabel, title, arrow, starx, stary):
    for i in range(len(labels)):
        ax.scatter(x[i], y[i], z[i], s=300, c = colors[i], label = labels[i])
    ax.set_ylabel('Latency'+ "($\downarrow$)", fontsize = 26, labelpad=22, fontweight='bold')
    ax.set_xlabel('Model weights'+ "($\downarrow$)", fontsize = 26, labelpad=20, fontweight='bold')
    # ax.set_zlabel(zlabel, rotation=90, fontsize = 18)
    if arrow == 1:
        ax.set_title(title + "\n" + zlabel + "($\\uparrow$)", y=1.0, fontsize = 28, fontweight='bold')
        ax.scatter(min(x) - starx, min(y)- stary, max(z), s=1000, c = "red", marker = "*", alpha = 1)
    else:
        ax.set_title(title + "\n" + zlabel + "($\downarrow$)", y=1.0, fontsize = 28, fontweight='bold')
        ax.scatter(min(x)- starx, min(y)- stary, min(z), s=1000, c = "red", marker = "*", alpha = 1)
    # ax.set_xscale('log')

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "black", "orange", "green", "tab:gray", "tab:olive", "tab:cyan", "tab:purple", "tab:brown", "tab:pink", "blue"]
labels = ["ERM", "MC Dropout", "MFVI BNN", "Rank-1 BNN", "ENN", "Posterior Net", "DUQ", "DDU", "NUQ", "DNN-GP", "SNGP", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"]

ax = fig.add_subplot(3, 4, 1, projection='3d')
plot_scatter(ax, get_Par10(), get_Lat10(), get_NLL_10(), colors, labels, "Negative log-likelihood", "WideResNet28-10 on CIFAR-10", 2, 10, 10)

ax = fig.add_subplot(3, 4, 2, projection='3d')
plot_scatter(ax, get_Par10(), get_Lat10(), get_ECE10(), colors, labels, "Expected calibration error", "WideResNet28-10 on CIFAR-10", 2, 10, 10)

ax = fig.add_subplot(3, 4, 3, projection='3d')
plot_scatter(ax, get_Par10(), get_Lat10(), get_Acc10(), colors, labels, "Accuracy", "WideResNet28-10 on CIFAR-10", 1, 0, 0)

ax = fig.add_subplot(3, 4, 4, projection='3d')
plot_scatter(ax, get_Par10(), get_Lat10(), get_AUPR10(), colors, labels, "AUPR", "WideResNet28-10 on CIFAR-10",  1, 5, 5)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "orange", "green", "tab:gray", "tab:cyan", "tab:purple", "tab:brown", "tab:pink", "blue"]
labels = ["ERM", "MC Dropout", "MFVI BNN", "Rank-1 BNN", "Posterior Net", "DUQ", "DDU", "DNN-GP", "SNGP", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"]

ax = fig.add_subplot(3, 4, 5, projection='3d')
plot_scatter(ax, get_Par100(), get_Lat100(), get_NLL_100(), colors, labels, "Negative log-likelihood", "WideResNet28-10 on CIFAR-100", 2, 10, 10)

ax = fig.add_subplot(3, 4, 6, projection='3d')
plot_scatter(ax, get_Par100(), get_Lat100(), get_ECE100(), colors, labels, "Expected calibration error", "WideResNet28-10 on CIFAR-100", 2, 10, 10)

ax = fig.add_subplot(3, 4, 7, projection='3d')
plot_scatter(ax, get_Par100(), get_Lat100(), get_Acc100(), colors, labels, "Accuracy", "WideResNet28-10 on CIFAR-100", 1, 0, 0)

ax = fig.add_subplot(3, 4, 8, projection='3d')
plot_scatter(ax, get_Par100(), get_Lat100(), get_AUPR100(), colors, labels, "AUPR", "WideResNet28-10 on CIFAR-100", 1, 0, 0)

ERM_ImgN = [0.939, 76.2, 0.032, 3.21, 40.5, 0.103, 25.61, 299.81]
Rank_1_ImgN = [0.886, 77.3, 0.017, 2.95, 42.9, 0.054, 26.35, 1383.65]
SNGP_ImgN = [0.932 , 76.1 , 0.015, 3.03, 41.1, 0.047, 49.60, 545.70]
BatchEnsemble_ImgN = [0.922, 76.8, 0.037, 3.09, 41.9, 0.089, 25.82, 696.81]
Ensembles_ImgN = [0.857, 77.9, 0.017, 2.82, 44.9, 0.047, 102.44, 701.34]
Density_ImgN = [0.950, 76.3, 0.024, 2.99, 41.0, 0.050, 25.88, 299.90]
colors = ["tab:blue", "tab:red", "tab:purple", "tab:brown", "tab:pink", "blue"]
labels = ["ERM", "Rank-1 BNN", "SNGP", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"]

def get_NLLImgN():
    def m(matrix):
        return np.mean([matrix[0], matrix[3]])
    return ([m(ERM_ImgN), m(Rank_1_ImgN),  
        m(SNGP_ImgN), m(BatchEnsemble_ImgN), m(Ensembles_ImgN), m(Density_ImgN)])

def get_AccImgN():
    def m(matrix):
        return np.mean([matrix[1], matrix[4]])
    return ([m(ERM_ImgN), m(Rank_1_ImgN),  
        m(SNGP_ImgN), m(BatchEnsemble_ImgN), m(Ensembles_ImgN), m(Density_ImgN)])

def get_ECEImgN():
    def m(matrix):
        return np.mean([matrix[2], matrix[5]])
    return ([m(ERM_ImgN), m(Rank_1_ImgN),  
        m(SNGP_ImgN), m(BatchEnsemble_ImgN), m(Ensembles_ImgN), m(Density_ImgN)])

def get_ParImgN():
    def m(matrix):
        return matrix[6]
    return ([m(ERM_ImgN), m(Rank_1_ImgN),  
        m(SNGP_ImgN), m(BatchEnsemble_ImgN), m(Ensembles_ImgN), m(Density_ImgN)])

def get_LatImgN():
    def m(matrix):
        return matrix[7]
    return ([m(ERM_ImgN), m(Rank_1_ImgN),  
        m(SNGP_ImgN), m(BatchEnsemble_ImgN), m(Ensembles_ImgN), m(Density_ImgN)])

ax = fig.add_subplot(3, 4, 9, projection='3d')
plot_scatter(ax, get_ParImgN(), get_LatImgN(), get_NLLImgN(), colors, labels, "Negative log-likelihood", "Resnet-50 on ImageNet", 2, 0, 0)

ax = fig.add_subplot(3, 4, 10, projection='3d')
plot_scatter(ax, get_ParImgN(), get_LatImgN(), get_ECEImgN(), colors, labels, "Expected calibration error", "Resnet-50 on ImageNet", 2, 0, 0)

ax = fig.add_subplot(3, 4, 11, projection='3d')
plot_scatter(ax, get_ParImgN(), get_LatImgN(), get_AccImgN(), colors, labels, "Accuracy", "Resnet-50 on ImageNet", 1, 0, 0)

y = [1071.88, 3880.63, 1296.12, 1279.91, 1485.95, 3875.74, 1087.49]
z = [(0.897+0.757)/2, (0.938+0.799)/2, (0.917+0.806)/2, (0.941+0.831)/2, (0.969+0.880)/2, (0.964+0.862)/2, (0.938+0.840)/2]
x = [108.43, 108.43, 197.02, 112.07, 118.70, 433.72, 108.53]
colors = ["tab:blue", "tab:orange", "green", "tab:cyan", "tab:purple", "tab:pink", "blue"]
labels = ["ERM", "MC Dropout", "DUQ", "DNN-GP", "SNGP", "Deep Ensembles", "Density-Softmax"]

ax = fig.add_subplot(3, 4, 12, projection='3d')
plot_scatter(ax, x, y, z, colors, labels, "AUPR/AUROC", "BERT on CLINC~OOS", 1, 0, 0)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
plt.figlegend(lines[:14], labels[:14], loc ='lower center', fancybox=True, shadow=True, ncol=7, prop={'size': 28})

plt.savefig("out.pdf",bbox_inches='tight')