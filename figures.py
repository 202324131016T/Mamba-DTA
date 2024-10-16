import os
import time
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # 在pyplot之前使用，控制pycharm是否显示绘图结果
import matplotlib.pyplot as plt
from sklearn import metrics

datatime = time.strftime('%Y%m%d-%H%M%S', time.localtime())
# Davis and drug_setting
path_dt_our = "20240811-121253"  # our
path_dt_CoVAE = "20240811-121105"  # CoVAE
# affinity_ls and pre_affinity_ls
path_aff = "fold_2_affinity_ls.txt"
path_pre = "fold_2_pre_affinity_ls.txt"
# epoch-CI
path_CI_txt = "fold_0_CI.txt"

def save_path(p):
    save_path = "./figures/" + datatime + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path + p

def fig_CI(save_flag=False, type='.pdf'):
    path_CI_our = "./result/" + path_dt_our + "/" + path_CI_txt
    CI_ls_our = np.loadtxt(path_CI_our)
    y_our = CI_ls_our  # our

    path_CI_CoVAE = "./result/" + path_dt_CoVAE + "/" + path_CI_txt
    CI_ls_CoVAE = np.loadtxt(path_CI_CoVAE)
    y_CoVAE = CI_ls_CoVAE  # CoVAE

    x = [i for i in range(len(CI_ls_our))]  # x

    y_our_max_idx = np.argmax(y_our)
    y_CoVAE_max_idx = np.argmax(y_CoVAE)
    # add new line
    plt.plot(x, y_our, color="red", linestyle="-", alpha=0.5, linewidth=1)
    plt.plot(x, y_CoVAE, color="blue", linestyle="-", alpha=0.5, linewidth=1)
    # max value
    plt.plot(y_our_max_idx, y_our[y_our_max_idx], color="y", marker='*', markeredgecolor='m', markersize='8')
    plt.plot(y_CoVAE_max_idx, y_CoVAE[y_CoVAE_max_idx], color="y", marker='*', markeredgecolor='m', markersize='8')
    # linestyle:线形状, alpha:透明度, linewidth:线条宽度, label:单条线的标签
    plt.legend(['Ours', 'CoVAE'], loc='lower right')
    # plt.title("Epoch-CI")  # 显示上面的label
    plt.xlabel("Epoch")  # x_label
    plt.ylabel("CI")  # y_label
    if save_flag:
        path = save_path("Epoch-CI" + type)
        plt.savefig(path)
    plt.show()

# figure AUC
def fig_AUC(save_flag=False, type='.pdf'):
    # plt setting figsize:size, dpi:分辨率, facecolor:背景颜色, edgecolor:边框颜色, frameon:是否显示边框
    # plt.figure(figsize=(6, 6), facecolor='#F0FFFF', edgecolor='#00FFFF')

    # our
    path_affinity = "./result/" + path_dt_our + "/" + path_aff
    affinity_ls = np.loadtxt(path_affinity)
    path_pre_affinity = "./result/" + path_dt_our + "/" + path_pre
    pre_affinity_ls = np.loadtxt(path_pre_affinity)
    # 计算FPR TPR
    FPR, TPR, threshold = metrics.roc_curve(affinity_ls > 7, pre_affinity_ls)
    roc_auc = metrics.auc(FPR, TPR)
    # 画图
    plt.plot(FPR, TPR, 'r', label='Ours=%0.3f' % roc_auc)
    # plt.plot(FPR, TPR, linestyle='-.', color='black', label='Ours = %0.3f' % roc_auc)

    # CoVAE
    path_affinity = "./result/" + path_dt_CoVAE + "/" + path_aff
    affinity_ls = np.loadtxt(path_affinity)
    path_pre_affinity = "./result/" + path_dt_CoVAE + "/" + path_pre
    pre_affinity_ls = np.loadtxt(path_pre_affinity)
    # 计算FPR TPR
    FPR, TPR, threshold = metrics.roc_curve(affinity_ls > 7, pre_affinity_ls)
    roc_auc = metrics.auc(FPR, TPR)
    # 画图
    plt.plot(FPR, TPR, 'b', label='CoVAE=%0.3f' % roc_auc)
    # plt.plot(FPR, TPR, linestyle='-', color='black', label='CoVAE = %0.3f' % roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1]) # 设定x轴范围
    plt.ylim([0, 1]) # 设定y轴范围
    # plt.title('ROC_AUC')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if save_flag:
        path = save_path("roc_auc" + type)
        plt.savefig(path)
    plt.show()

# figure AUC—cutoff_values
def fig_AUC_cutoff(save_flag=False, type='.pdf'):
    # plt setting figsize:size, dpi:分辨率, facecolor:背景颜色, edgecolor:边框颜色, frameon:是否显示边框
    # plt.figure(figsize=(6, 6), facecolor='#F0FFFF', edgecolor='#00FFFF')

    # 计算 cutoff_values 变化后的AUC
    cutoff_values = [float(i) / 10 for i in range(50, 90, 1)] # x
    AUC_values = [0 for i in range(len(cutoff_values))] # y

    # our
    path_affinity = "./result/" + path_dt_our + "/" + path_aff
    affinity_ls = np.loadtxt(path_affinity)
    path_pre_affinity = "./result/" + path_dt_our + "/" + path_pre
    pre_affinity_ls = np.loadtxt(path_pre_affinity)
    for cutoff_i in range(len(cutoff_values)):
        # 计算FPR TPR
        FPR, TPR, threshold = metrics.roc_curve(affinity_ls > cutoff_values[cutoff_i], pre_affinity_ls)
        AUC_values[cutoff_i] = metrics.auc(FPR, TPR)
    x = cutoff_values
    y = AUC_values
    # 画图
    plt.plot(x, y, 'r')
    # plt.plot(x, y, linestyle='-.', color='black')

    # CoVAE
    path_affinity = "./result/" + path_dt_CoVAE + "/" + path_aff
    affinity_ls = np.loadtxt(path_affinity)
    path_pre_affinity = "./result/" + path_dt_CoVAE + "/" + path_pre
    pre_affinity_ls = np.loadtxt(path_pre_affinity)
    for cutoff_i in range(len(cutoff_values)):
        # 计算FPR TPR
        FPR, TPR, threshold = metrics.roc_curve(affinity_ls > cutoff_values[cutoff_i], pre_affinity_ls)
        AUC_values[cutoff_i] = metrics.auc(FPR, TPR)
    x = cutoff_values
    y = AUC_values
    # 画图
    plt.plot(x, y, 'b')
    # plt.plot(x, y, linestyle='-', color='black')

    plt.legend(['Ours', 'CoVAE'], loc='upper left')
    # plt.xlim([5, 9]) # 设定x轴范围
    # plt.ylim([0.75, 1]) # 设定y轴范围
    # plt.title('Threshold-AUC')
    plt.ylabel('AUC')
    plt.xlabel('Threshold')
    if save_flag:
        path = save_path("Threshold-AUC" + type)
        plt.savefig(path)
    plt.show()

# figure affinity and pre_affinity for CoVAE
def fig_aff_CoVAE(save_flag=False, type='.pdf'):
    # plt setting figsize:size, dpi:分辨率, facecolor:背景颜色, edgecolor:边框颜色, frameon:是否显示边框
    # plt.figure(figsize=(6, 6), facecolor='#F0FFFF', edgecolor='#00FFFF')

    path_affinity = "./result/" + path_dt_CoVAE + "/" + path_aff
    affinity_ls = np.loadtxt(path_affinity)
    path_pre_affinity = "./result/" + path_dt_CoVAE + "/" + path_pre
    pre_affinity_ls = np.loadtxt(path_pre_affinity)

    x = affinity_ls
    y = pre_affinity_ls
    # 散点图
    plt.scatter(x, y, s=6, c='b', marker='o',alpha=0.65)

    # 参照线
    line = [i for i in range(4, 11)]
    plt.plot(line, line, color="k", linestyle="--", alpha=1, linewidth=1)
    # plt.title("CoVAE")  # 显示上面的label
    plt.xlabel("Ground Truth Affinity")  # x_label
    plt.ylabel("Predicted Affinity")  # y_label
    if save_flag:
        path = save_path("aff_CoVAE" + type)
        plt.savefig(path)
    plt.show()

# figure affinity and pre_affinity for our
def fig_aff_our(save_flag=False, type='.pdf'):
    # plt setting figsize:size, dpi:分辨率, facecolor:背景颜色, edgecolor:边框颜色, frameon:是否显示边框
    # plt.figure(figsize=(6, 6), facecolor='#F0FFFF', edgecolor='#00FFFF')

    path_affinity = "./result/" + path_dt_our + "/" + path_aff
    affinity_ls = np.loadtxt(path_affinity)
    path_pre_affinity = "./result/" + path_dt_our + "/" + path_pre
    pre_affinity_ls = np.loadtxt(path_pre_affinity)

    x = affinity_ls
    y = pre_affinity_ls
    # 散点图
    plt.scatter(x, y, s=6, c='r', marker='o',alpha=0.65)

    # 参照线
    line = [i for i in range(4, 11)]
    plt.plot(line, line, color="k", linestyle="--", alpha=1, linewidth=1)
    # plt.title("our")  # 显示上面的label
    plt.xlabel("Ground Truth Affinity")  # x_label
    plt.ylabel("Predicted Affinity")  # y_label
    if save_flag:
        path = save_path("aff_our" + type)
        plt.savefig(path)
    plt.show()

# figure affinity and pre_affinity for our
def fig_aff(save_flag=False, type='.pdf'):
    # plt setting figsize:size, dpi:分辨率, facecolor:背景颜色, edgecolor:边框颜色, frameon:是否显示边框
    # plt.figure(figsize=(6, 6), facecolor='#F0FFFF', edgecolor='#00FFFF')

    path_affinity = "./result/" + path_dt_our + "/" + path_aff
    affinity_ls = np.loadtxt(path_affinity)
    path_pre_affinity = "./result/" + path_dt_our + "/" + path_pre
    pre_affinity_ls = np.loadtxt(path_pre_affinity)
    x = affinity_ls
    y = pre_affinity_ls
    # 散点图 our
    plt.scatter(x, y, s=2, c='r', marker='o',alpha=0.4)

    path_affinity = "./result/" + path_dt_CoVAE + "/" + path_aff
    affinity_ls = np.loadtxt(path_affinity)
    path_pre_affinity = "./result/" + path_dt_CoVAE + "/" + path_pre
    pre_affinity_ls = np.loadtxt(path_pre_affinity)
    x = affinity_ls
    y = pre_affinity_ls
    # 散点图 CoVAE
    plt.scatter(x, y, s=2, c='b', marker='o', alpha=0.4)

    # 参照线
    line = [i for i in range(9, 16)]
    plt.plot(line, line, color="k", linestyle="--", alpha=1, linewidth=1)
    # plt.title("our")  # 显示上面的label
    plt.xlabel("Ground Truth Affinity")  # x_label
    plt.ylabel("Predicted Affinity")  # y_label
    plt.legend(['ours', 'CoVAE'], loc='lower right')
    if save_flag:
        path = save_path("aff" + type)
        plt.savefig(path)
    plt.show()

# MSE drug setting
def fig_MSE_drug(save_flag=False, type='.pdf'):
    # figure MSE_drug
    MSE_Davis_2 = [0.796, 0.726, 0.955, 0.840, 0.724, 0.796]
    methods = ["KronRLS", "DeepDTA", "DeepAffinity", "GraphDTA", "Co-VAE", "our"]
    colors = ["#696969", "#FFC0CB", "#00FFFF", "#FFFF00", "#90EE90", "#FF0000"]
    x = methods
    y = MSE_Davis_2
    plt.bar(x, y, width=0.5, bottom=None, color=colors)
    plt.title("Davis and drug_setting")  # 显示上面的label
    plt.xlabel("methods")  # x_label
    plt.ylabel("MSE")  # y_label
    if save_flag:
        path = save_path("MSE_drug" + type)
        plt.savefig(path)
    plt.show()

# std_MSE drug setting
def fig_std_MSE_drug(save_flag=False, type='.pdf'):
    # figure std_MSE_drug
    std_MSE_Davis_2 = [0.231, 0.091, 0.140, 0.058, 0.096, 0.139]
    methods = ["KronRLS", "DeepDTA", "DeepAffinity", "GraphDTA", "Co-VAE", "our"]
    colors = ["#696969", "#FFC0CB", "#00FFFF", "#FFFF00", "#90EE90", "#FF0000"]
    x = methods
    y = std_MSE_Davis_2
    plt.bar(x, y, width=0.5, bottom=None, color=colors)
    plt.title("Davis and drug_setting")  # 显示上面的label
    plt.xlabel("methods")  # x_label
    plt.ylabel("std_MSE")  # y_label
    if save_flag:
        path = save_path("std_MSE_drug" + type)
        plt.savefig(path)
    plt.show()

# MSE tgt setting
def fig_MSE_tgt(save_flag=False, type='.pdf'):
    # figure MSE_tgt
    MSE_Davis_3 = [0.429, 0.490, 0.477, 0.457, 0.419, 0.400]
    methods = ["KronRLS", "DeepDTA", "DeepAffinity", "GraphDTA", "Co-VAE", "our"]
    colors = ["#696969", "#FFC0CB", "#00FFFF", "#FFFF00", "#90EE90", "#FF0000"]
    x = methods
    y = MSE_Davis_3
    plt.bar(x, y, width=0.5, bottom=None, color=colors)
    plt.title("Davis and target_setting")  # 显示上面的label
    plt.xlabel("methods")  # x_label
    plt.ylabel("MSE")  # y_label
    if save_flag:
        path = save_path("MSE_tgt" + type)
        plt.savefig(path)
    plt.show()

# std_MSE tgt setting
def fig_std_MSE_tgt(save_flag=False, type='.pdf'):
    # figure std_MSE_tgt
    std_MSE_Davis_3 = [0.055, 0.095, 0.019, 0.009, 0.016, 0.033]
    methods = ["KronRLS", "DeepDTA", "DeepAffinity", "GraphDTA", "Co-VAE", "our"]
    colors = ["#696969", "#FFC0CB", "#00FFFF", "#FFFF00", "#90EE90", "#FF0000"]
    x = methods
    y = std_MSE_Davis_3
    plt.bar(x, y, width=0.5, bottom=None, color=colors)
    plt.title("Davis and target_setting")  # 显示上面的label
    plt.xlabel("methods")  # x_label
    plt.ylabel("MSE")  # y_label
    if save_flag:
        path = save_path("std_MSE_tgt" + type)
        plt.savefig(path)
    plt.show()

# rm2 drug setting
def fig_rm2_drug(save_flag=False, type='.pdf'):
    # figure rm2_drug
    rm2_drug = [0.143, 0.122, 0.114, 0.176, 0.107, 0.221]
    methods = ["KronRLS", "DeepDTA", "DeepAffinity", "GraphDTA", "Co-VAE", "our"]
    colors = ["#696969", "#FFC0CB", "#00FFFF", "#FFFF00", "#90EE90", "#FF0000"]
    x = methods
    y = rm2_drug
    plt.bar(x, y, width=0.5, bottom=None, color=colors)
    plt.title("Davis and drug_setting")  # 显示上面的label
    plt.xlabel("methods")  # x_label
    plt.ylabel("rm2")  # y_label
    if save_flag:
        path = save_path("rm2_drug" + type)
        plt.savefig(path)
    plt.show()

# rm2 tgt setting
def fig_rm2_tgt(save_flag=False, type='.pdf'):
    # figure rm2_tgt
    rm2_tgt = [0.448, 0.433, 0.444, 0.429, 0.477, 0.525]
    methods = ["KronRLS", "DeepDTA", "DeepAffinity", "GraphDTA", "Co-VAE", "our"]
    colors = ["#696969", "#FFC0CB", "#00FFFF", "#FFFF00", "#90EE90", "#FF0000"]
    x = methods
    y = rm2_tgt
    plt.bar(x, y, width=0.5, bottom=None, color=colors)
    plt.title("Davis and target_setting")  # 显示上面的label
    plt.xlabel("methods")  # x_label
    plt.ylabel("rm2")  # y_label
    if save_flag:
        path = save_path("rm2_tgt" + type)
        plt.savefig(path)
    plt.show()

def test():
    ls_3 = [
        "setting=3 i=0 (CI=@0.740822@ std_CI=0.012047) (MSE=@0.376927@ std_MSE=0.023374) (MAE=@0.416399@ std_MAE=0.015427) (rm2=@0.489961@ std_rm2=0.019152) (auc=@0.827165@ aupr=@0.654364@)",
        "setting=3 i=1 (CI=@0.779944@ std_CI=0.008261) (MSE=@0.347692@ std_MSE=0.030019) (MAE=@0.381487@ std_MAE=0.008941) (rm2=@0.524444@ std_rm2=0.025443) (auc=@0.841537@ aupr=@0.671721@)",
        "setting=3 i=2 (CI=@0.735758@ std_CI=0.019938) (MSE=@0.439970@ std_MSE=0.025661) (MAE=@0.439444@ std_MAE=0.018744) (rm2=@0.380514@ std_rm2=0.040212) (auc=@0.775507@ aupr=@0.563398@)",
        "setting=3 i=3 (CI=@0.772834@ std_CI=0.006622) (MSE=@0.602954@ std_MSE=0.173237) (MAE=@0.517168@ std_MAE=0.100286) (rm2=@0.379772@ std_rm2=0.027095) (auc=@0.771817@ aupr=@0.578261@)",
        "setting=3 i=4 (CI=@0.767926@ std_CI=0.011989) (MSE=@0.370397@ std_MSE=0.010653) (MAE=@0.426760@ std_MAE=0.009532) (rm2=@0.441468@ std_rm2=0.015570) (auc=@0.833231@ aupr=@0.634294@)",
        "setting=3 i=5 (CI=@0.779934@ std_CI=0.014779) (MSE=@0.499772@ std_MSE=0.096488) (MAE=@0.480164@ std_MAE=0.044493) (rm2=@0.447685@ std_rm2=0.061930) (auc=@0.800741@ aupr=@0.642708@)",
        "setting=3 i=6 (CI=@0.783995@ std_CI=0.014364) (MSE=@0.491283@ std_MSE=0.070332) (MAE=@0.462043@ std_MAE=0.038666) (rm2=@0.409084@ std_rm2=0.027688) (auc=@0.785045@ aupr=@0.576759@)",
        "setting=3 i=7 (CI=@0.772918@ std_CI=0.021094) (MSE=@0.371324@ std_MSE=0.038905) (MAE=@0.412736@ std_MAE=0.026868) (rm2=@0.461622@ std_rm2=0.039331) (auc=@0.822138@ aupr=@0.625676@)",
        "setting=3 i=8 (CI=@0.744718@ std_CI=0.005140) (MSE=@0.574551@ std_MSE=0.176441) (MAE=@0.542381@ std_MAE=0.106637) (rm2=@0.411010@ std_rm2=0.061762) (auc=@0.805445@ aupr=@0.630040@)",
        "setting=3 i=9 (CI=@0.820736@ std_CI=0.010044) (MSE=@0.875595@ std_MSE=0.099312) (MAE=@0.690607@ std_MAE=0.054921) (rm2=@0.365877@ std_rm2=0.029848) (auc=@0.784315@ aupr=@0.647536@)"]
    ls_2 = [
        "setting=2 i=0 (CI=@0.762190@ std_CI=0.004577) (MSE=@0.415533@ std_MSE=0.010848) (MAE=@0.405477@ std_MAE=0.009484) (rm2=@0.418034@ std_rm2=0.005897) (auc=@0.800446@ aupr=@0.626057@)",
        "setting=2 i=1 (CI=@0.747199@ std_CI=0.003735) (MSE=@0.380290@ std_MSE=0.007743) (MAE=@0.395178@ std_MAE=0.007905) (rm2=@0.406965@ std_rm2=0.014518) (auc=@0.805140@ aupr=@0.616658@)",
        "setting=2 i=2 (CI=@0.773647@ std_CI=0.001736) (MSE=@0.387161@ std_MSE=0.008650) (MAE=@0.392317@ std_MAE=0.008441) (rm2=@0.452778@ std_rm2=0.010940) (auc=@0.809196@ aupr=@0.646134@)",
        "setting=2 i=3 (CI=@0.757554@ std_CI=0.004887) (MSE=@0.393740@ std_MSE=0.011001) (MAE=@0.373009@ std_MAE=0.009033) (rm2=@0.388696@ std_rm2=0.013094) (auc=@0.793541@ aupr=@0.593768@)",
        "setting=2 i=4 (CI=@0.767067@ std_CI=0.004469) (MSE=@0.426862@ std_MSE=0.008458) (MAE=@0.409846@ std_MAE=0.002404) (rm2=@0.410433@ std_rm2=0.009762) (auc=@0.799997@ aupr=@0.630031@)",
        "setting=2 i=5 (CI=@0.754425@ std_CI=0.006200) (MSE=@0.389848@ std_MSE=0.014799) (MAE=@0.400287@ std_MAE=0.009841) (rm2=@0.402761@ std_rm2=0.010711) (auc=@0.793344@ aupr=@0.605062@)",
        "setting=2 i=6 (CI=@0.742611@ std_CI=0.007139) (MSE=@0.422275@ std_MSE=0.027646) (MAE=@0.423435@ std_MAE=0.015563) (rm2=@0.391316@ std_rm2=0.023045) (auc=@0.783810@ aupr=@0.589287@)",
        "setting=2 i=7 (CI=@0.773558@ std_CI=0.006734) (MSE=@0.375463@ std_MSE=0.024008) (MAE=@0.386946@ std_MAE=0.015342) (rm2=@0.408690@ std_rm2=0.018926) (auc=@0.800952@ aupr=@0.612375@)",
        "setting=2 i=8 (CI=@0.748292@ std_CI=0.007156) (MSE=@0.365496@ std_MSE=0.010560) (MAE=@0.378905@ std_MAE=0.009939) (rm2=@0.418484@ std_rm2=0.016617) (auc=@0.796914@ aupr=@0.620513@)",
        "setting=2 i=9 (CI=@0.747774@ std_CI=0.004819) (MSE=@0.446595@ std_MSE=0.025952) (MAE=@0.423930@ std_MAE=0.010992) (rm2=@0.357235@ std_rm2=0.023712) (auc=@0.788396@ aupr=@0.601752@)"]
    CI =  []
    MSE = []
    MAE = []
    rm2 = []
    AUC = []
    AUPR= []
    for i in range(10):
        s = ls_3[i].split("@")
        CI.append(float(s[1]))
        MSE.append(float(s[3]))
        MAE.append(float(s[5]))
        rm2.append(float(s[7]))
        AUC.append(float(s[9]))
        AUPR.append(float(s[11]))

    result_CI = np.mean(CI)
    result_MSE = np.mean(MSE)
    result_MAE = np.mean(MAE)
    result_rm2 = np.mean(rm2)
    result_auc = np.mean(AUC)
    result_aupr = np.mean(AUPR)
    std_CI = np.std(CI)
    std_MSE = np.std(MSE)
    std_MAE = np.std(MAE)
    std_rm2 = np.std(rm2)
    print("%.3f(%.3f)%.3f(%.3f)%.3f(%.3f)%.3f(%.3f) %.3f %.3f" % (result_CI, std_CI, result_MSE, std_MSE, result_MAE, std_MAE, result_rm2, std_rm2, result_auc, result_aupr))

    print()

def att(save_flag=False, type='.pdf'):
    x = np.load("./figures/x1.npy")
    y = np.load("./figures/y1.npy")

    x_ = np.load("./figures/x_1.npy")
    y_ = np.load("./figures/y_1.npy")

    # (N, C, L) -> (N, L, C)
    x = np.transpose(x, (0, 2, 1))
    x = x[0:60, :]
    # x = np.true_divide(x, 7.1)

    y = np.transpose(y, (0, 2, 1))
    y = y[0:60, :]
    # y = np.true_divide(y, 7.2)

    x_ = np.transpose(x_, (0, 2, 1))
    x_ = x_[0:60, :]
    # x_ = np.true_divide(x_, 3.2)

    y_ = np.transpose(y_, (0, 2, 1))
    y_ = y_[0:60, :]
    # y_ = np.true_divide(y_, 2.6)

    # (N, L)
    x = np.max(x, axis=2)
    y = np.max(y, axis=2)
    x_ = np.max(x_, axis=2)
    y_ = np.max(y_, axis=2)

    plt.imshow(x, cmap=plt.cm.hot)
    plt.title("x")
    plt.colorbar()
    plt.show()

    plt.imshow(x_, cmap=plt.cm.hot)
    plt.title("x_att")
    plt.colorbar()
    plt.show()

    plt.imshow(y, cmap=plt.cm.hot)
    plt.title("y")
    plt.colorbar()
    plt.show()

    plt.imshow(y_, cmap=plt.cm.hot)
    plt.title("y_att")
    plt.colorbar()
    plt.show()

    print()

def check_DTA():
    XD = np.load("./figures/XD.npy")
    XT = np.load("./figures/XT.npy")
    Y = np.load("./figures/Y.npy")
    for di in range(len(XD)):
        for ti in range(len(XT)):
            A = Y[di][ti]
            if A >= 17:
                print(di, "and", ti, "and", A)
                print(XD[di])
                print(XT[ti])
                print("-----")

    print()
'''
17.200179498
CC1=C(C=CC(=C1)NC(=O)NC2=CC=C(C=C2)C3=CC=NC4=NNC(=C34)N)F
MVSYWDTGVLLCALLSCLLLTGSSSGSKLKDPELSLKGTQHIMQAGQTLHLQCRGEAAHKWSLPEMVSKESERLSITKSACGRNGKQFCSTLTLNTAQANHTGFYSCKYLAVPTSKKKETESAIYIFISDTGRPFVEourSEIPEIIHMTEGRELVIPCRVTSPNITVTLKKFPLDTLIPDGKRIIWDSRKGFIISNATYKEIGLLTCEATVNGHLYKTNYLTHRQTNTIIDVQISTPRPVKLLRGHTLVLNCTATTPLNTRVQMTWSYPDEKNKRASVRRRIDQSNSHANIFYSVLTIDKMQNKDKGLYTCRVRSGPSFKSVNTSVHIYDKAFITVKHRKQQVLETVAGKRSYRLSMKVKAFPSPEVVWLKDGLPATEKSARYLTRGYSLIIKDVTEEDAGNYTILLSIKQSNVFKNLTATLIVNVKPQIYEKAVSSFPDPALYPLGSRQILTCTAYGIPQPTIKWFWHPCNHNHSEARCDFCSNNEESFILDADSNMGNRIESITQRMAIIEGKNKMASTLVVADSRISGIYICIASNKVGTVGRNISFYITDVPNGFHVNLEKMPTEGEDLKLSCTVNKFLYRDVTWILLRTVNNRTMHYSISKQKMAITKEHSITLNLTIMNVSLQDSGTYACRARNVYTGEEILQKKEITIRDQEAPYLLRNLSDHTVAISSSTTLDCHANGVPEPQITWFKNNHKIQQEPGIILGPGSSTLFIERVTEEDEGVYHCKATNQKGSVESSAYLTVQGTSDKSNLELITLTCTCVAATLFWLLLTLFIRKMKRSSSEIKTDYLSIIMDPDEVPLDEQCERLPYDASKWEFARERLKLGKSLGRGAFGKVVQASAFGIKKSPTCRTVAVKMLKEGATASEYKALMTELKILTHIGHHLNVVNLLGACTKQGGPLMVIVEYCKYGNLSNYLKSKRDLFFLNKDAALHMEPKKEKMEPGLEQGKKPRLDSVTSSESFASSGFQEDKSLSDVEEEEDSDGFYKEPITMEDLISYSFQVARGMEFLSSRKCIHRDLAARNILLSENNVVKICDFGLARDIYKNPDYVRKGDTRLPLKWMAPESIFDKIYSTKSDVWSYGVLLWEIFSLGGSPYPGVQMDEDFCSRLREGMRMRAPEYSTPEIYQIMLDCWHRDPKERPRFAELVEKLGDLLQANVQQDGKDYIPINAILTGNSGFTYSTPAFSEDFFKESISAPKFNSGSSDDVRYVNAFKFMSLERIKTFEELLPNATSMFDDYQGDSSTLLASPMLKRFTWTDSKPKASLKIDLRVTSKSKESGLSDVSRPSFCHSSCGHVSEGKRRFTYDHAELERKIACCSPPPDYNSVVLYSTPPI
'''

def fig_CI_bar(save_flag=False, type='.pdf'):
    # figure MSE_drug
    KIBA_D_CI = [0.729, 0.728, 0.707, 0.728, 0.742, 0.763]
    methods = ["KronRLS", "DeepDTA", "DeepAffinity", "GraphDTA", "Co-VAE", "BDTA(ours)"]
    colors = ["#A2B5CD", "#A2B5CD", "#A2B5CD", "#A2B5CD", "#A2B5CD", "#CD919E"]
    x = methods
    y = KIBA_D_CI
    plt.bar(x, y, width=0.5, bottom=None, color=colors)
    # plt.title("KIBA and drug_setting")  # 显示上面的label
    plt.ylim([0.6, 0.8])  # 设定y轴范围
    plt.xlabel("methods")  # x_label
    plt.ylabel("CI")  # y_label
    if save_flag:
        path = save_path("KIBA_drug_CI" + type)
        plt.savefig(path)
    plt.show()

if __name__ == '__main__':
    save_flags = [False, True]
    save_flag = save_flags[0]
    types = ['.pdf', '.png']
    type = types[0]
    # fig_CI(save_flag, type)
    fig_AUC(save_flag, type)
    fig_AUC_cutoff(save_flag, type)
    fig_aff_CoVAE(save_flag, type)
    fig_aff_our(save_flag, type)
    # fig_aff(save_flag, type)  # one fig two points
    # fig_MSE_drug(save_flag, type)
    # fig_std_MSE_drug(save_flag, type)
    # fig_MSE_tgt(save_flag, type)
    # fig_std_MSE_tgt(save_flag, type)
    # fig_rm2_drug(save_flag, type)
    # fig_rm2_tgt(save_flag, type)
    # test()
    # att(save_flag, type)
    # check_DTA()
    # fig_CI_bar(save_flag, type)

    print()