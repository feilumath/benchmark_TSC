import numpy as np
import matplotlib.pyplot as plt


def plot_my_roc_version1(logging_save_path, *, addition_str, fpr_NNpred, tpr_NNpred, fpr_simpleapprox, tpr_simpleapprox, fpr_low_frequency, tpr_low_frequency, fpr_hiddentrue, tpr_hiddentrue):
    plt.figure()
    plt.plot(fpr_NNpred, tpr_NNpred, label="Model approximation; ACC* = %.4f" % np.max((-fpr_NNpred + tpr_NNpred + 1.0) / 2.0))
    plt.plot(fpr_simpleapprox, tpr_simpleapprox, label="Numerical approximation; ACC* = %.4f" % np.max((-fpr_simpleapprox + tpr_simpleapprox + 1.0) / 2.0))
    if not (fpr_low_frequency is None):
        plt.plot(fpr_low_frequency, tpr_low_frequency, label="Discrete theoretical; ACC* = %.4f" % np.max((-fpr_low_frequency + tpr_low_frequency + 1.0) / 2.0))
    plt.plot(fpr_hiddentrue, tpr_hiddentrue, label="Continuous hidden truth; ACC* = %.4f" % np.max((-fpr_hiddentrue + tpr_hiddentrue + 1.0) / 2.0))
    plt.plot([0.0, 1.0], [0.0, 1.0], 'black')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title(addition_str + "Part: ROC Curve")

    new_path = logging_save_path[:-4] + "_" + addition_str + "_ROC.png"
    plt.savefig(new_path)
    plt.close()


def plot_my_roc_version2(logging_save_path, *, addition_str,
                         fpr_test_NN, tpr_test_NN,
                         fpr_test_RF, tpr_test_RF, fpr_test_ROCKRT, tpr_test_ROCKRT,
                         fpr_simpleapprox, tpr_simpleapprox, fpr_low_frequency, tpr_low_frequency, fpr_hiddentrue, tpr_hiddentrue):
    plt.figure()
    plt.plot(fpr_test_NN, tpr_test_NN, label="ResNet; ACC* = %.4f" % np.max((-fpr_test_NN + tpr_test_NN + 1.0) / 2.0))
    plt.plot(fpr_test_RF, tpr_test_RF, label="RF; ACC* = %.4f" % np.max((-fpr_test_RF + tpr_test_RF + 1.0) / 2.0))
    plt.plot(fpr_test_ROCKRT, tpr_test_ROCKRT, label="ROCKRT; ACC* = %.4f" % np.max((-fpr_test_ROCKRT + tpr_test_ROCKRT + 1.0) / 2.0))

    plt.plot(fpr_simpleapprox, tpr_simpleapprox,label="Numerical approximation; ACC* = %.4f" % np.max((-fpr_simpleapprox + tpr_simpleapprox + 1.0) / 2.0))
    if not (fpr_low_frequency is None):
        plt.plot(fpr_low_frequency, tpr_low_frequency, label="Discrete theoretical; ACC* = %.4f" % np.max((-fpr_low_frequency + tpr_low_frequency + 1.0) / 2.0))
    plt.plot(fpr_hiddentrue, tpr_hiddentrue, label="Continuous hidden truth; ACC* = %.4f" % np.max((-fpr_hiddentrue + tpr_hiddentrue + 1.0) / 2.0))
    plt.plot([0.0, 1.0], [0.0, 1.0], 'black')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title(addition_str + "Part: ROC Curve")

    new_path = logging_save_path[:-4] + "_" + addition_str + "_ROC.png"
    plt.savefig(new_path)
    plt.close()


def plot_loss_freq_small(logging_save_path, *, history_val_loss_freq_small, history_train_loss_freq_small, history_lr):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(history_val_loss_freq_small, label="val")
    ax.plot(history_train_loss_freq_small, label="train")
    ax.legend()
    ax.set_title("Loss")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(history_lr, label="lr")
    ax.legend()
    ax.set_title("Learning rate")

    new_path = logging_save_path + "_loss.png"
    plt.savefig(new_path)
    plt.close()


def plot_loss(logging_save_path, *, history_loss, history_lr):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(history_loss, label="ori")
    ax.legend()
    ax.set_title("Loss")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(history_lr, label="lr")
    ax.legend()
    ax.set_title("Learning rate")

    new_path = logging_save_path[:-4] + "_loss_when_training.png"
    plt.savefig(new_path)
    plt.close()