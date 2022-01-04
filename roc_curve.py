#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Arjun Thangaraju'
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def pdf(x, std, mean):
    const = 1.0 / np.sqrt(2 * np.pi * (std ** 2))
    pdf_normal_dist = const * np.exp(-((x - mean) ** 2) / (2.0 * (std ** 2)))
    return pdf_normal_dist


def plot_pdf(good_pdf, bad_pdf, ax):
    ax.fill(x, good_pdf, "g", alpha=0.5)
    ax.fill(x, bad_pdf, "b", alpha=0.5)
    ax.set_xlim([-4, 4])
    ax.set_ylim([0, 10])
    ax.set_title("Probability Distribution", fontsize=14)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlabel('P(X)', fontsize=12)
    ax.legend(["Class Two", "Class Three"])


def plot_roc(good_pdf, bad_pdf, ax, color):
    # Total
    total_bad = np.sum(bad_pdf)
    total_good = np.sum(good_pdf)
    # Cummulative sum
    cum_TP = 0
    cum_FP = 0
    # TPR and FPR list initialization
    TPR_list = []
    FPR_list = []
    # Iterate through all values of x
    for i in range(len(x)):
        # We are only interested in non-zero values of bad
        if bad_pdf[i] > 0:
            cum_TP += bad_pdf[len(x) - 1 - i]
            cum_FP += good_pdf[len(x) - 1 - i]
        FPR = cum_FP / total_good
        TPR = cum_TP / total_bad
        TPR_list.append(TPR)
        FPR_list.append(FPR)
    # Calculating AUC, taking the 100 timesteps into account
    auc = metrics.auc(FPR_list, TPR_list)
    # auc = np.trapz(TPR_list, FPR_list, dx=0.000000000000000000001)
    plot_final_roc(x, FPR_list, TPR_list, auc, ax, color)
    return auc


def plot_final_roc(x, FPR_list, TPR_list, auc, ax, color):
    # Plotting final ROC curve
    ax.plot(FPR_list, TPR_list, color)
    ax.plot(x, x, "--r")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("ROC Cruve", fontsize=14)
    ax.set_ylabel('TPR', fontsize=12)
    ax.set_xlabel('FPR', fontsize=12)
    ax.grid()
    # print(auc)
    ax.legend(["AUC=%.3f" % auc])

#Plot ROC Function
fig, ax = plt.subplots(1 ,1, figsize=(10, 5))
plot_roc(good_pdf, bad_pdf, ax)
plt.show()


# Individual Plots

x = np.linspace(-4, 4, 100000)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# plot pdf curve

# Class 1 vs Class 2
plot_pdf(pdf(x, 0.05, -0.62), pdf(x, 0.13, 0.12), ax[0])
plot_pdf(pdf(x, 0.18, -1.14), pdf(x, 0.07, -0.13), ax[0])
plot_pdf(pdf(x, 0.08, 0.01), pdf(x, 0.10, 0.61), ax[0])
plot_pdf(pdf(x, 0.13, 0.40), pdf(x, 0.11, 1.54), ax[0])
# plot roc curve
auc_pca = plot_roc(pdf(x, 0.05, -0.62), pdf(x, 0.13, 0.12), ax[1], "orange")
auc_linear_ae = plot_roc(pdf(x, 0.18, -1.14), pdf(x, 0.07, -0.13), ax[1], "green")
auc_non_linear_sig_ae = plot_roc(pdf(x, 0.08, 0.01), pdf(x, 0.10, 0.61), ax[1], "blue")
auc_non_linear_relu_ae = plot_roc(pdf(x, 0.13, 0.40), pdf(x, 0.11, 1.54), ax[1], "black")



# Class 2 vs Class 3
plot_pdf(pdf(x, 0.13, 0.12), pdf(x, 0.15, 0.50), ax[0])
plot_pdf(pdf(x, 0.21, -1.65), pdf(x, 0.18, -1.14), ax[0]) # Order Switched
plot_pdf(pdf(x, 0.10, 0.61), pdf(x, 0.06, 0.85), ax[0])
plot_pdf(pdf(x, 0.03, 0.05), pdf(x, 0.13, 0.40), ax[0]) # Order Switched

# plot roc curve
# auc_pca = plot_roc(pdf(x, 0.1, 0.5), pdf(x, 0.1, 0.5), ax[1], "orange")
auc_pca = plot_roc(pdf(x, 0.13, 0.12), pdf(x, 0.15, 0.50), ax[1], "orange")
auc_linear_ae = plot_roc(pdf(x, 0.21, -1.65), pdf(x, 0.18, -1.14), ax[1], "green") # Order Switched
auc_non_linear_sig_ae = plot_roc(pdf(x, 0.10, 0.61), pdf(x, 0.06, 0.85), ax[1], "blue")
auc_non_linear_relu_ae = plot_roc(pdf(x, 0.03, 0.05), pdf(x, 0.13, 0.40), ax[1], "black") # Order Switched



# Test
plot_pdf(pdf(x, 0.1, 0.1), pdf(x, 0.1, 0.9), ax[0])
auc_pca = plot_roc(pdf(x, 0.1, 0.1), pdf(x, 0.1, 0.9), ax[1], "orange")
plt.legend(["PCA; AUC=%.4f" % auc_pca])



plt.legend(["PCA; AUC=%.4f" % auc_pca, "", "Linear AE; AUC=%.4f" % auc_linear_ae, "",
            "Non-Linear sigmoid-based AE; AUC=%.4f" % auc_non_linear_sig_ae, "",
            "Non-Linear relu-based; AUC=%.4f" % auc_non_linear_relu_ae], loc="best")

plt.tight_layout()
plt.show()


# Print Statements
# print("PCA AUC = %.4f" %auc_pca)
# print("Linear AE AUC = %.4f"%auc_linear_ae)
# print("Non-Linear sigmoid-based AE AUC = %.4f"%auc_non_linear_sig_ae)
# print("Non-Linear relu-based AE AUC = %.4f"%auc_non_linear_relu_ae)
