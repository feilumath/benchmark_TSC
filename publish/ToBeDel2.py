# %%
import matplotlib.pyplot as plt
import numpy as np
import ml_collections
from utils.add_paths import add_paths
import pandas as pd
import copy

# %%
def find_data(*, root, date, Majorind):
    if Majorind == 1:
        keysforsave_str = root + f'/constant_drift_BM/res_all_keys_{date[0]}.npy'
        valuesforsave_str = root + f'/constant_drift_BM/res_all_values_{date[0]}.npy'
        SubPathName = 'constant_drift_BM/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 1
    elif Majorind == 2:
        keysforsave_str = root + f'/moredim_OU/res_all_keys_{date[1]}.npy'
        valuesforsave_str = root + f'/moredim_OU/res_all_values_{date[1]}.npy'
        SubPathName = 'moredim_OU/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 2
    elif Majorind == 3:
        keysforsave_str = root + f'/data_ts/linear v.s. nonlinear/res_all_keys_{date[2]}.npy'
        valuesforsave_str = root + f'/data_ts/linear v.s. nonlinear/res_all_values_{date[2]}.npy'
        SubPathName = 'data_ts/linear v.s. nonlinear/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 3
    elif Majorind == 4:
        keysforsave_str = root + f'/Different_potentials/res_all_keys_{date[3]}.npy'
        valuesforsave_str = root + f'/Different_potentials/res_all_values_{date[3]}.npy'
        SubPathName = 'Different_potentials/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 4
    elif Majorind == 5:
        keysforsave_str = root + f'/Different_potentials/res_all_keys_{date[4]}.npy'
        valuesforsave_str = root + f'/Different_potentials/res_all_values_{date[4]}.npy'
        SubPathName = 'Different_potentials/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 5
    elif Majorind == 6:
        keysforsave_str = root + f'/moredim_Opinion/res_all_keys_{date[5]}.npy'
        valuesforsave_str = root + f'/moredim_Opinion/res_all_values_{date[5]}.npy'
        SubPathName = 'moredim_Opinion/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 6
    elif Majorind == 7:
        keysforsave_str = root + f'/moredim_Opinion/res_all_keys_{date[6]}.npy'
        valuesforsave_str = root + f'/moredim_Opinion/res_all_values_{date[6]}.npy'
        SubPathName = 'moredim_Opinion/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 7
    res_all_keys = np.load(keysforsave_str, allow_pickle=True)
    res_all_values = np.load(valuesforsave_str, allow_pickle=True)
    res_all_len = res_all_keys.__len__()
    res_load = ml_collections.ConfigDict()
    for i in range(res_all_len):
        setattr(res_load, res_all_keys[i], res_all_values[i])

    num_cases = res_load.description.__len__()
    num_dims_all_np = np.zeros(num_cases)
    my_description = res_load.description
    if (SubPathName == 'constant_drift_BM/') and (tableline == 1):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(
                float(mid_description[mid_description.index('_T_') + 3: mid_description.index('_ir_')]))
    elif (SubPathName == 'moredim_OU/') and (tableline == 2):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(
                float(mid_description[mid_description.index('_m_') + 3: mid_description.index('_p_')]))
    elif (SubPathName == 'data_ts/linear v.s. nonlinear/') and (tableline == 3):
        for i in range(num_cases):
            mid_description = my_description[i]
            mid_nst_m1 = int(
                float(mid_description[mid_description.index('_nst_') + 5: mid_description.index('_DS_')])) - 1
            num_dims_all_np[i] = mid_nst_m1
    elif (SubPathName == 'Different_potentials/') and (tableline == 4):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(
                float(mid_description[mid_description.index('_T_') + 3: mid_description.index('_ir_')]))
    elif (SubPathName == 'Different_potentials/') and (tableline == 5):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(
                float(mid_description[mid_description.index('_T_') + 3: mid_description.index('_ir_')]))
    elif (SubPathName == 'moredim_Opinion/') and (tableline == 6):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(
                float(mid_description[mid_description.index('_m_') + 3: mid_description.index('_p_')]))
    elif (SubPathName == 'moredim_Opinion/') and (tableline == 7):
        for i in range(num_cases):
            mid_description = my_description[i]
            mid_nst_m1 = int(
                float(mid_description[mid_description.index('_nst_') + 5: mid_description.index('_DS_')])) - 1
            num_dims_all_np[i] = mid_nst_m1

    num_dims_all_unique_np = np.sort(np.array(list(set(list(num_dims_all_np)))))
    num_x_toplot = num_dims_all_unique_np.shape[0]

    return res_load


# %%
# read data
root = "/Users/esthersida/Documents/Code/UC_ts_sig/marcc/sigUCts/results"
# date = ["2022-12-24-3-1-48","2022-12-24-2-45-45","2022-12-24-3-5-2",\
#     "2022-12-24-4-53-8", "2022-12-24-15-36-42", "2022-12-24-3-42-3",\
#         "2022-12-24-22-10-38"]
date = ["2023-01-01-19-24-20","2023-01-02-1-35-48","2023-01-02-2-17-21",\
    "2023-01-02-3-18-15", "2023-01-02-1-10-0", "2023-01-02-1-56-25",\
        "2023-01-01-19-57-57"]


# %%
def get_dataframe(whichtoplot):
    df = pd.DataFrame({"X": [None], "Y": [None], "Type": [None], "Case": None})
    for caseNumber in range(1, 8):
        res_load = find_data(root=root, date=date, Majorind=caseNumber)

        x_toplot = getattr(res_load, 'fpr_hiddentrue_tuple')[whichtoplot]
        y_toplot = getattr(res_load, 'tpr_hiddentrue_tuple')[whichtoplot]
        df_temp = pd.DataFrame({"X": x_toplot, "Y": y_toplot, "Type": ["ROC hidden truth"] * len(x_toplot),
                                "Case": [caseNumber] * len(x_toplot)})
        df = pd.concat([df, df_temp])

        x_toplot = getattr(res_load, 'fpr_simpleapprox_tuple')[whichtoplot]
        y_toplot = getattr(res_load, 'tpr_simpleapprox_tuple')[whichtoplot]
        df_temp = pd.DataFrame({"X": x_toplot, "Y": y_toplot, "Type": ["ROC numerical"] * len(x_toplot),
                                "Case": [caseNumber] * len(x_toplot)})
        df = pd.concat([df, df_temp])

        x_toplot = getattr(res_load, 'fpr_test_ResNet_tuple')[whichtoplot]
        y_toplot = getattr(res_load, 'tpr_test_ResNet_tuple')[whichtoplot]
        df_temp = pd.DataFrame({"X": x_toplot, "Y": y_toplot, "Type": ["ResNet"] * len(x_toplot),
                                "Case": [caseNumber] * len(x_toplot)})
        df = pd.concat([df, df_temp])

        x_toplot = getattr(res_load, 'fpr_test_RF_tuple')[whichtoplot]
        y_toplot = getattr(res_load, 'tpr_test_RF_tuple')[whichtoplot]
        df_temp = pd.DataFrame({"X": x_toplot, "Y": y_toplot, "Type": ["RF"] * len(x_toplot),
                                "Case": [caseNumber] * len(x_toplot)})
        df = pd.concat([df, df_temp])

        x_toplot = getattr(res_load, 'fpr_test_ROCKRT_tuple')[whichtoplot]
        y_toplot = getattr(res_load, 'tpr_test_ROCKRT_tuple')[whichtoplot]
        df_temp = pd.DataFrame({"X": x_toplot, "Y": y_toplot, "Type": ["ROCKET"] * len(x_toplot),
                                "Case": [caseNumber] * len(x_toplot)})
        df = pd.concat([df, df_temp])
    return df.iloc[1:]


# %%
whichtoplot = 4
df_roc = get_dataframe(whichtoplot)

# %%
# Figure 567 combined
fig,axs = plt.subplots(2,3,figsize = (16,8))
# sns.set_theme(style="ticks", palette="pastel")
# My_ls_list = [(0, (5, 3)),
#               (0, (5, 3)),
#               (0, (5, 3)),
#               '-',
#               (0, (5, 3)),
#               (0, (5, 3))]  # # For ResNet,RF, ROCKET, hidden truth, numerical, middle line
My_ls_list = ["--",
              "-.",
              ":",
              '-',
              (0, (5, 3)),
              (0, (5, 3))]  # # For ResNet,RF, ROCKET, hidden truth, numerical, middle line

# a)
subset = df_roc[df_roc["Case"] == 1]
axs[0, 0].plot(subset[subset['Type'] == 'ROC hidden truth']["X"], subset[subset['Type'] == 'ROC hidden truth']["Y"],
               ls=My_ls_list[3], lw=3, color="black", label="LRT hidden truth")
axs[0, 0].plot(subset[subset['Type'] == 'ROC numerical']["X"], subset[subset['Type'] == 'ROC numerical']["Y"],
               ls=My_ls_list[4], lw=2, color="red", label="LRT numerical")

axs[0, 0].plot(subset[subset['Type'] == 'ResNet']["X"], subset[subset['Type'] == 'ResNet']["Y"], ls=My_ls_list[0], lw=1.5,
               label="ResNet")
axs[0, 0].plot(subset[subset['Type'] == 'RF']["X"], subset[subset['Type'] == 'RF']["Y"], ls=My_ls_list[1], lw=1.5,
               label="RF")
axs[0, 0].plot(subset[subset['Type'] == 'ROCKET']["X"], subset[subset['Type'] == 'ROCKET']["Y"], ls=My_ls_list[2],
               lw=1.5, label="ROCKET")
axs[0, 0].plot([0.0, 1.0], [0.0, 1.0], ls=My_ls_list[5], lw=1, color="black")
axs[0, 0].set_xlabel("FPR",fontsize = 16)
axs[0, 0].set_ylabel("TPR",fontsize = 16)
axs[0, 0].set_title("Constant drifts", fontsize=20)
axs[0, 0].set_xlim([0.0, 1.0])
axs[0, 0].set_ylim([0.0, 1.0])
axs[0, 0].legend([], [], frameon=False)

# b)
subset = df_roc[df_roc["Case"] == 4]
axs[1,0].plot(subset[subset['Type'] == 'ROC hidden truth']["X"], subset[subset['Type'] == 'ROC hidden truth']["Y"],
               ls=My_ls_list[3], lw=3, color="black", label="LRT hidden truth")
axs[1,0].plot(subset[subset['Type'] == 'ROC numerical']["X"], subset[subset['Type'] == 'ROC numerical']["Y"],
               ls=My_ls_list[4], lw=2, color="red", label="LRT numerical")

axs[1,0].plot(subset[subset['Type'] == 'ResNet']["X"], subset[subset['Type'] == 'ResNet']["Y"], ls=My_ls_list[0], lw=1.5,
               label="ResNet")
axs[1,0].plot(subset[subset['Type'] == 'RF']["X"], subset[subset['Type'] == 'RF']["Y"], ls=My_ls_list[1], lw=1.5,
               label="RF")
axs[1,0].plot(subset[subset['Type'] == 'ROCKET']["X"], subset[subset['Type'] == 'ROCKET']["Y"], ls=My_ls_list[2],
               lw=1.5, label="ROCKET")
axs[1,0].plot([0.0, 1.0], [0.0, 1.0], ls=My_ls_list[5], lw=1, color="black")
axs[1,0].set_xlabel("FPR",fontsize = 16)
axs[1,0].set_ylabel("TPR",fontsize = 16)
axs[1,0].set_title("Different potentials", fontsize=20)
axs[1,0].set_xlim([0.0, 1.0])
axs[1,0].set_ylim([0.0, 1.0])
axs[1,0].legend([], [], frameon=False)

# c)
subset = df_roc[df_roc["Case"] == 2]
axs[0,1].plot(subset[subset['Type'] == 'ROC hidden truth']["X"], subset[subset['Type'] == 'ROC hidden truth']["Y"],
               ls=My_ls_list[3], lw=3, color="black", label="LRT hidden truth")
axs[0,1].plot(subset[subset['Type'] == 'ROC numerical']["X"], subset[subset['Type'] == 'ROC numerical']["Y"],
               ls=My_ls_list[4], lw=2, color="red", label="LRT numerical")
axs[0,1].plot(subset[subset['Type'] == 'ResNet']["X"], subset[subset['Type'] == 'ResNet']["Y"], ls=My_ls_list[0], lw=1.5,
               label="ResNet")
axs[0,1].plot(subset[subset['Type'] == 'RF']["X"], subset[subset['Type'] == 'RF']["Y"], ls=My_ls_list[1], lw=1.5,
               label="RF")
axs[0,1].plot(subset[subset['Type'] == 'ROCKET']["X"], subset[subset['Type'] == 'ROCKET']["Y"], ls=My_ls_list[2],
               lw=1.5, label="ROCKET")
axs[0,1].plot([0.0, 1.0], [0.0, 1.0], ls=My_ls_list[5], lw=1, color="black")
axs[0,1].set_xlabel("FPR",fontsize = 16)
axs[0,1].set_ylabel("TPR",fontsize = 16)
axs[0,1].set_title("OU processes", fontsize=20)
axs[0,1].set_xlim([0.0, 1.0])
axs[0,1].set_ylim([0.0, 1.0])
axs[0,1].legend([], [], frameon=False)

# d)
subset = df_roc[df_roc["Case"] == 6]
axs[1, 1].plot(subset[subset['Type'] == 'ROC hidden truth']["X"], subset[subset['Type'] == 'ROC hidden truth']["Y"],
               ls=My_ls_list[3], lw=3, color="black", label="LRT hidden truth")
axs[1, 1].plot(subset[subset['Type'] == 'ROC numerical']["X"], subset[subset['Type'] == 'ROC numerical']["Y"],
               ls=My_ls_list[4], lw=2, color="red", label="LRT numerical")

axs[1, 1].plot(subset[subset['Type'] == 'ResNet']["X"], subset[subset['Type'] == 'ResNet']["Y"], ls=My_ls_list[0], lw=1.5,
               label="ResNet")
axs[1, 1].plot(subset[subset['Type'] == 'RF']["X"], subset[subset['Type'] == 'RF']["Y"], ls=My_ls_list[1], lw=1.5,
               label="RF")
axs[1, 1].plot(subset[subset['Type'] == 'ROCKET']["X"], subset[subset['Type'] == 'ROCKET']["Y"], ls=My_ls_list[2],
               lw=1.5, label="ROCKET")
axs[1, 1].plot([0.0, 1.0], [0.0, 1.0], ls=My_ls_list[5], lw=1, color="black")
axs[1, 1].set_xlabel("FPR",fontsize = 16)
axs[1, 1].set_ylabel("TPR",fontsize = 16)
axs[1, 1].set_title("Interacting particles", fontsize=20)
axs[1, 1].set_xlim([0.0, 1.0])
axs[1, 1].set_ylim([0.0, 1.0])
axs[1, 1].legend([], [], frameon=False)

# e)
subset = df_roc[df_roc["Case"] == 3]
axs[0,2].plot(subset[subset['Type'] == 'ROC hidden truth']["X"], subset[subset['Type'] == 'ROC hidden truth']["Y"],
               ls=My_ls_list[3], lw=3, color="black", label="LRT hidden truth")
axs[0,2].plot(subset[subset['Type'] == 'ROC numerical']["X"], subset[subset['Type'] == 'ROC numerical']["Y"],
               ls=My_ls_list[4], lw=2, color="red", label="LRT numerical")

axs[0,2].plot(subset[subset['Type'] == 'ResNet']["X"], subset[subset['Type'] == 'ResNet']["Y"], ls=My_ls_list[0], lw=1.5,
               label="ResNet")
axs[0,2].plot(subset[subset['Type'] == 'RF']["X"], subset[subset['Type'] == 'RF']["Y"], ls=My_ls_list[1], lw=1.5,
               label="RF")
axs[0,2].plot(subset[subset['Type'] == 'ROCKET']["X"], subset[subset['Type'] == 'ROCKET']["Y"], ls=My_ls_list[2],
               lw=1.5, label="ROCKET")
axs[0,2].plot([0.0, 1.0], [0.0, 1.0], ls=My_ls_list[5], lw=1, color="black")
axs[0,2].set_xlabel("FPR",fontsize = 16)
axs[0,2].set_ylabel("TPR",fontsize = 16)
axs[0,2].set_title("Linear vs. nonlinear", fontsize=20)
axs[0,2].set_xlim([0.0, 1.0])
axs[0,2].set_ylim([0.0, 1.0])
axs[0,2].legend([], [], frameon=False)

whichtoplot = 1
df_roc = get_dataframe(whichtoplot)
# f)
subset = df_roc[df_roc["Case"] == 7]
axs[1,2].plot(subset[subset['Type'] == 'ROC hidden truth']["X"], subset[subset['Type'] == 'ROC hidden truth']["Y"],
               ls=My_ls_list[3], lw=3, color="black", label="LRT hidden truth")
axs[1,2].plot(subset[subset['Type'] == 'ROC numerical']["X"], subset[subset['Type'] == 'ROC numerical']["Y"],
               ls=My_ls_list[4], lw=2, color="red", label="LRT numerical")

axs[1,2].plot(subset[subset['Type'] == 'ResNet']["X"], subset[subset['Type'] == 'ResNet']["Y"], ls=My_ls_list[0], lw=1.5,
               label="ResNet")
axs[1,2].plot(subset[subset['Type'] == 'RF']["X"], subset[subset['Type'] == 'RF']["Y"], ls=My_ls_list[1], lw=1.5,
               label="RF")
axs[1,2].plot(subset[subset['Type'] == 'ROCKET']["X"], subset[subset['Type'] == 'ROCKET']["Y"], ls=My_ls_list[2],
               lw=1.5, label="ROCKET")
axs[1,2].plot([0.0, 1.0], [0.0, 1.0], ls=My_ls_list[5], lw=1, color="black")
axs[1,2].set_xlabel("FPR",fontsize = 16)
axs[1,2].set_ylabel("TPR",fontsize = 16)
axs[1,2].set_title("Interacting particles", fontsize=20)
axs[1,2].set_xlim([0.0, 1.0])
axs[1,2].set_ylim([0.0, 1.0])
axs[1,2].legend([], [], frameon=False)

axs[0,0].set_title("a)",loc = "left",fontsize = 16, weight='bold')
axs[0,1].set_title("c)",loc = "left",fontsize = 16, weight='bold')
axs[0,2].set_title("e)",loc = "left",fontsize = 16, weight='bold')
axs[1,0].set_title("b)",loc = "left",fontsize = 16, weight='bold')
axs[1,1].set_title("d)",loc = "left",fontsize = 16, weight='bold')
axs[1,2].set_title("f)",loc = "left",fontsize = 16, weight='bold')

axs[0,0].tick_params(axis='both', labelsize=14)
axs[0,1].tick_params(axis='both', labelsize=14)
axs[0,2].tick_params(axis='both', labelsize=14)
axs[1,0].tick_params(axis='both', labelsize=14)
axs[1,1].tick_params(axis='both', labelsize=14)
axs[1,2].tick_params(axis='both', labelsize=14)

handles, labels = fig.gca().get_legend_handles_labels()
hd,lb = copy.deepcopy(handles),copy.deepcopy(labels)
hd_p,lb_p = [],[]
hd_p.insert(0,hd.pop(0))
hd_p.insert(0,hd.pop(0))
lb_p.insert(0,lb.pop(0))
lb_p.insert(0,lb.pop(0))

axs[0, 1].legend(hd, lb, frameon=False,
                 loc="lower right",fontsize = 14)
axs[0, 0].legend(hd_p, lb_p, frameon=False,
                 loc="lower right",fontsize = 14)
# axs[1,0].legend(frameon = False, loc = "lower right")
fig.tight_layout()
fig.savefig(root + "/ROC_plot.pdf")

# %%
