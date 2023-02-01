# %%
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ml_collections
from utils.add_paths import add_paths
import datetime
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
# %%
def find_data(*, root, date, Majorind, type = "auc"):
    if Majorind == 1:
        keysforsave_str   = root+f'/constant_drift_BM/res_all_keys_{date[0]}.npy'
        valuesforsave_str = root+f'/constant_drift_BM/res_all_values_{date[0]}.npy'
        SubPathName = 'constant_drift_BM/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 1
    elif Majorind == 2:
        keysforsave_str   = root+f'/moredim_OU/res_all_keys_{date[1]}.npy'
        valuesforsave_str = root+f'/moredim_OU/res_all_values_{date[1]}.npy'
        SubPathName = 'moredim_OU/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 2
    elif Majorind == 3:
        keysforsave_str   = root+f'/data_ts/linear v.s. nonlinear/res_all_keys_{date[2]}.npy'
        valuesforsave_str = root+f'/data_ts/linear v.s. nonlinear/res_all_values_{date[2]}.npy'
        SubPathName = 'data_ts/linear v.s. nonlinear/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 3
    elif Majorind == 4:
        keysforsave_str   = root+f'/Different_potentials/res_all_keys_{date[3]}.npy'
        valuesforsave_str = root+f'/Different_potentials/res_all_values_{date[3]}.npy'
        SubPathName = 'Different_potentials/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 4
    elif Majorind == 5:
        keysforsave_str   = root+f'/Different_potentials/res_all_keys_{date[4]}.npy'
        valuesforsave_str = root+f'/Different_potentials/res_all_values_{date[4]}.npy'
        SubPathName = 'Different_potentials/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 5
    elif Majorind == 6:
        keysforsave_str   = root+f'/moredim_Opinion/res_all_keys_{date[5]}.npy'
        valuesforsave_str = root+f'/moredim_Opinion/res_all_values_{date[5]}.npy'
        SubPathName = 'moredim_Opinion/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 6
    elif Majorind == 7:
        keysforsave_str   = root+f'/moredim_Opinion/res_all_keys_{date[6]}.npy'
        valuesforsave_str = root+f'/moredim_Opinion/res_all_values_{date[6]}.npy'
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
            num_dims_all_np[i] = int(float(mid_description[mid_description.index('_T_') + 3: mid_description.index('_ir_')]))
    elif (SubPathName == 'moredim_OU/') and (tableline == 2):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(float(mid_description[mid_description.index('_m_') + 3: mid_description.index('_p_')]))
    elif (SubPathName == 'data_ts/linear v.s. nonlinear/') and (tableline == 3):
        for i in range(num_cases):
            mid_description = my_description[i]
            mid_nst_m1      = int(float(mid_description[mid_description.index('_nst_') + 5: mid_description.index('_DS_')])) - 1
            num_dims_all_np[i] = mid_nst_m1
    elif (SubPathName == 'Different_potentials/') and (tableline == 4):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(float(mid_description[mid_description.index('_T_') + 3: mid_description.index('_ir_')]))
    elif (SubPathName == 'Different_potentials/') and (tableline == 5):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(float(mid_description[mid_description.index('_T_') + 3: mid_description.index('_ir_')]))
    elif (SubPathName == 'moredim_Opinion/') and (tableline == 6):
        for i in range(num_cases):
            mid_description = my_description[i]
            num_dims_all_np[i] = int(float(mid_description[mid_description.index('_m_') + 3: mid_description.index('_p_')]))
    elif (SubPathName == 'moredim_Opinion/') and (tableline == 7):
        for i in range(num_cases):
            mid_description = my_description[i]
            mid_nst_m1      = int(float(mid_description[mid_description.index('_nst_') + 5: mid_description.index('_DS_')])) - 1
            num_dims_all_np[i] = mid_nst_m1

    num_dims_all_unique_np = np.sort(np.array(list(set(list(num_dims_all_np)))))
    num_x_toplot = num_dims_all_unique_np.shape[0]

    name_app_list = ['_test_ResNet',
                     '_test_RF', '_test_ROCKRT',
                     '_simpleapprox', '_hiddentrue']
    for i in range(name_app_list.__len__()):
        mid_auc_np = np.array(getattr(res_load, 'auc' + name_app_list[i] + '_tuple'), dtype=float)
        mid_auc_mean_np = np.zeros(num_x_toplot)
        mid_auc_errorbar_np = np.zeros((2, num_x_toplot))
        for j in range(num_x_toplot):
            mid_mid_auc_np = mid_auc_np[np.where(num_dims_all_np == num_dims_all_unique_np[j])]
            mid_auc_mean_np[j] = np.mean(mid_mid_auc_np)

            mid_auc_errorbar_np[0, j] = mid_auc_mean_np[j] - np.sort(mid_mid_auc_np)[2]
            mid_auc_errorbar_np[1, j] = -mid_auc_mean_np[j] + np.sort(mid_mid_auc_np)[-3]
        setattr(res_load, type + '_mean' + name_app_list[i] + '_np', mid_auc_mean_np)
        setattr(res_load, type + '_errorbar' + name_app_list[i] + '_np', mid_auc_errorbar_np)
    return num_dims_all_unique_np,res_load,Sub_Results_path,SubPathName

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
def get_dataframe(typeNow):
    df = pd.DataFrame({"X": [None], "Y": [None], "Type":[None], "Case": None})
    for caseNumber in range(1,8):
        num_dims_all_unique_np,res_load,Sub_Results_path,SubPathName = find_data(root = root ,date = date ,Majorind = caseNumber,type = typeNow)
        x_toplot = np.log2(num_dims_all_unique_np)
        LRT_hidden_truth = getattr(res_load, typeNow +'_mean' + '_hiddentrue' + '_np')
        df_temp = pd.DataFrame({"X": x_toplot, "Y": LRT_hidden_truth, "Type":["LRT hidden truth"]*len(x_toplot), "Case": [caseNumber]*len(x_toplot)})
        df = pd.concat([df,df_temp])
        LRT_numerical = getattr(res_load, typeNow+'_mean' + '_simpleapprox' + '_np')
        df_temp = pd.DataFrame({"X": x_toplot, "Y": LRT_numerical, "Type":["LRT numerical"]*len(x_toplot), "Case": [caseNumber]*len(x_toplot)})
        df = pd.concat([df,df_temp])
        x_toplot_rep = np.repeat(x_toplot,40)
        ResNet = getattr(res_load,typeNow+"_test_ResNet_tuple")
        df_temp = pd.DataFrame({"X": x_toplot_rep, "Y": ResNet, "Type":["ResNet"]*len(x_toplot)*40, "Case": [caseNumber]*len(x_toplot)*40})
        df = pd.concat([df,df_temp])
        RF = getattr(res_load,typeNow+"_test_RF_tuple")
        df_temp = pd.DataFrame({"X": x_toplot_rep, "Y": RF, "Type":["RF"]*len(x_toplot)*40, "Case": [caseNumber]*len(x_toplot)*40})
        df = pd.concat([df,df_temp])
        ResNet = getattr(res_load,typeNow+"_test_ROCKRT_tuple")
        df_temp = pd.DataFrame({"X": x_toplot_rep, "Y": ResNet, "Type":["ROCKET"]*len(x_toplot)*40, "Case": [caseNumber]*len(x_toplot)*40})
        df = pd.concat([df,df_temp])
    return df.iloc[1:]
# %%   
df_auc = get_dataframe("auc")
df_acc = get_dataframe("acc_star")
#%%
fig,axs = plt.subplots(2,3,figsize = (16,8))
sns.set_theme(style="ticks", palette="pastel")
# a)
subset = df_auc[df_auc["Case"] == 1]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,0],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,0].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[0,0].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[0,0].set_xlabel (r"End Time $t_L$",fontsize = 16)
axs[0,0].set_ylabel ("AUC",fontsize = 16)
axs[0,0].set_title("Constant drifts",fontsize = 20)
axs[0,0].set_xticks(np.arange(4), (np.rint(2 ** subset.iloc[:4]["X"])).astype(int),fontsize = 14)
axs[0,0].set_ylim([0.6,1])
axs[0,0].legend([],[],frameon= False)
# b)s
subset = df_auc[df_auc["Case"] == 4]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,0],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,0].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[1,0].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[1,0].set_xlabel(r"End Time $t_L$",fontsize = 16)
axs[1,0].set_ylabel ("AUC",fontsize = 16)
axs[1,0].set_title("Different potentials",fontsize = 20)
axs[1,0].set_xticks( np.arange(4),(np.rint(2 ** subset.iloc[:4]["X"])).astype(int),fontsize = 14)
axs[1,0].legend([],[],frameon= False)
axs[1,0].set_ylim([0.6,1])
# c)
subset = df_auc[df_auc["Case"] == 2]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,1],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,1].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[0,1].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[0,1].set_xlabel (r"Dimension $d$",fontsize = 16)
axs[0,1].set_ylabel ("")
axs[0,1].set_title("OU processes",fontsize = 20)
axs[0,1].set_xticks(np.arange(4), (np.rint(2 ** subset.iloc[:4]["X"])).astype(int),fontsize = 14)
axs[0,1].legend([],[],frameon= False)
axs[0,1].set_ylim([0.3,1])
# d)
subset = df_auc[df_auc["Case"] == 6]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,1],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,1].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[1,1].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[1,1].set_xlabel(r"Dimension $d$",fontsize = 16)
axs[1,1].set_ylabel ("")
axs[1,1].set_title("Interacting particles",fontsize = 20)
axs[1,1].set_xticks( np.arange(4),(np.rint(2 ** subset.iloc[:4]["X"])).astype(int),fontsize = 14)
axs[1,1].legend([],[],frameon= False)
axs[1,1].set_ylim([0.3,1])
# e)
subset = df_auc[df_auc["Case"] == 3]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,2],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,2].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[0,2].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[0,2].set_xlabel (r"Time interval $\Delta t$",fontsize = 16)
axs[0,2].set_ylabel ("")
axs[0,2].set_title("Linear vs. nonlinear",fontsize = 20)
axs[0,2].set_xticks(np.arange(4), (1./np.rint(2 ** subset.iloc[:4]["X"])).astype(float),fontsize = 14)
axs[0,2].legend([],[],frameon= False)
axs[0,2].set_ylim([0.9,1])
# f)
subset = df_auc[df_auc["Case"] == 7]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,2],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,2].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[1,2].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[1,2].set_xlabel(r"Time interval $\Delta t$",fontsize = 16)
axs[1,2].set_ylabel ("")
axs[1,2].set_title("Interacting particles",fontsize = 20)
axs[1,2].set_xticks( np.arange(4),(4./np.rint(2 ** subset.iloc[:4]["X"])).astype(float),fontsize = 14)
axs[1,2].legend([],[],frameon= False)
axs[1,2].set_ylim([0.5,1])

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
handles.insert(1, plt.Line2D([],[], alpha=0))
labels.insert(1,'')
order = [1,4,5,0,2,3]
axs[0,1].legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol = 2,frameon = False, loc = "lower right",fontsize = 14)
# axs[1,0].legend(frameon = False, loc = "lower right")
fig.tight_layout()
fig.savefig(root+"/AUC_plot.pdf")
# %%
# Figure 8 combined
fig,axs = plt.subplots(2,3,figsize = (16,8))
sns.set_theme(style="ticks", palette="pastel")
# a)
subset = df_acc[df_acc["Case"] == 1]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,0],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,0].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[0,0].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[0,0].set_xlabel (r"End Time $t_L$",fontsize = 16)
axs[0,0].set_ylabel (r"ACC$_*$",fontsize = 16)
axs[0,0].set_title("Constant drifts",fontsize = 20)
axs[0,0].set_xticks(np.arange(4), (np.rint(2 ** subset.iloc[:4]["X"])).astype(int),fontsize = 14)
axs[0,0].set_ylim([0.6,1])
axs[0,0].legend([],[],frameon= False)
# b)s
subset = df_acc[df_acc["Case"] == 4]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,0],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,0].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[1,0].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[1,0].set_xlabel(r"End Time $t_L$",fontsize = 16)
axs[1,0].set_ylabel (r"ACC$_*$",fontsize = 16)
axs[1,0].set_title("Different potentials",fontsize = 20)
axs[1,0].set_xticks( np.arange(4),(np.rint(2 ** subset.iloc[:4]["X"])).astype(int),fontsize = 14)
axs[1,0].legend([],[],frameon= False)
axs[1,0].set_ylim([0.6,1])
# c)
subset = df_acc[df_acc["Case"] == 2]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,1],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,1].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[0,1].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[0,1].set_xlabel (r"Dimension $d$",fontsize = 16)
axs[0,1].set_ylabel ("")
axs[0,1].set_title("OU processes",fontsize = 20)
axs[0,1].set_xticks(np.arange(4), (np.rint(2 ** subset.iloc[:4]["X"])).astype(int),fontsize = 14)
axs[0,1].legend([],[],frameon= False)
axs[0,1].set_ylim([0.4,1])
# d)
subset = df_acc[df_acc["Case"] == 6]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,1],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,1].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[1,1].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[1,1].set_xlabel(r"Dimension $d$",fontsize = 16)
axs[1,1].set_ylabel ("")
axs[1,1].set_title("Interacting particles",fontsize = 20)
axs[1,1].set_xticks( np.arange(4),(np.rint(2 ** subset.iloc[:4]["X"])).astype(int),fontsize = 14)
axs[1,1].legend([],[],frameon= False)
axs[1,1].set_ylim([0.4,1])
# e)
subset = df_acc[df_acc["Case"] == 3]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,2],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,2].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[0,2].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[0,2].set_xlabel (r"Time interval $\Delta t$",fontsize = 16)
axs[0,2].set_ylabel ("")
axs[0,2].set_title("Linear vs. nonlinear",fontsize = 20)
axs[0,2].set_xticks(np.arange(4), (1./np.rint(2 ** subset.iloc[:4]["X"])).astype(float),fontsize = 14)
axs[0,2].legend([],[],frameon= False)
axs[0,2].set_ylim([0.8,1])
# f)
subset = df_acc[df_acc["Case"] == 7]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,2],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,2].plot(subset.iloc[:4]["Y"],"v-",lw = 2,color = "black",label = "LRT hidden truth")
axs[1,2].plot(subset.iloc[4:8]["Y"],"*:",lw = 2,color = "red",label = "LRT numerical")
axs[1,2].set_xlabel(r"Time interval $\Delta t$",fontsize = 16)
axs[1,2].set_ylabel ("")
axs[1,2].set_title("Interacting particles",fontsize = 20)
axs[1,2].set_xticks( np.arange(4),(4./np.rint(2 ** subset.iloc[:4]["X"])).astype(float),fontsize = 14)
axs[1,2].legend([],[],frameon= False)
axs[1,2].set_ylim([0.5,1])

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
handles.insert(1, plt.Line2D([],[], alpha=0))
labels.insert(1,'')
order = [1,4,5,0,2,3]
axs[0,1].legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol = 2,frameon = False, loc = "lower right",fontsize = 14)
# axs[1,0].legend(frameon = False, loc = "lower right")
fig.tight_layout()
fig.savefig(root+"/ACC_plot.pdf")


# %%
typeNow = "duration"

def get_dataframe_dur(typeNow):
    df = pd.DataFrame({"X": [None], "Y": [None], "Type":[None], "Case": None})
    for caseNumber in range(1,8):
        num_dims_all_unique_np,res_load,Sub_Results_path,SubPathName = find_data(root = root ,date = date ,Majorind = caseNumber,type = typeNow)
        x_toplot = np.log2(num_dims_all_unique_np)
        all_duration = res_load.duration_all_tuple
        dur = np.zeros((len(all_duration),3))
        for i in range(len(all_duration)):
            dur[i] = all_duration[i]

        x_toplot_rep = np.repeat(x_toplot,40)
        ResNet = dur[:,0]
        df_temp = pd.DataFrame({"X": x_toplot_rep, "Y": ResNet, "Type":["ResNet"]*len(x_toplot)*40, "Case": [caseNumber]*len(x_toplot)*40})
        df = pd.concat([df,df_temp])
        RF = dur[:,1]
        df_temp = pd.DataFrame({"X": x_toplot_rep, "Y": RF, "Type":["RF"]*len(x_toplot)*40, "Case": [caseNumber]*len(x_toplot)*40})
        df = pd.concat([df,df_temp])
        ResNet = dur[:,2]
        df_temp = pd.DataFrame({"X": x_toplot_rep, "Y": ResNet, "Type":["ROCKET"]*len(x_toplot)*40, "Case": [caseNumber]*len(x_toplot)*40})
        df = pd.concat([df,df_temp])
    return df.iloc[1:]
df_t = get_dataframe_dur(typeNow)
# %%
# Figure 9 combined
fig,axs = plt.subplots(2,3,figsize = (16,8))
sns.set_theme(style="ticks", palette="pastel")
# a)
subset = df_t[df_t["Case"] == 1]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,0],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,0].set_xlabel (r"End Time $t_L$",fontsize = 16)
axs[0,0].set_ylabel ("Computation Time",fontsize = 16)
axs[0,0].set_title("Constant drifts",fontsize = 20)
axs[0,0].set_xticks(np.arange(4), (np.rint(2 ** subset["X"].unique())).astype(int),fontsize = 14)
axs[0,0].set_ylim([0,120])
axs[0,0].legend([],[],frameon= False)
# b)s
subset = df_t[df_t["Case"] == 4]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,0],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,0].set_xlabel(r"End Time $t_L$",fontsize = 16)
axs[1,0].set_ylabel ("Computation Time",fontsize = 16)
axs[1,0].set_title("Different potentials",fontsize = 20)
axs[1,0].set_xticks(np.arange(4),(np.rint(2 ** subset["X"].unique())).astype(int),fontsize = 14)
axs[1,0].legend([],[],frameon= False)
axs[1,0].set_ylim([0,120])
# c)
subset = df_t[df_t["Case"] == 2]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,1],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,1].set_xlabel (r"Dimension $d$",fontsize = 16)
axs[0,1].set_ylabel ("")
axs[0,1].set_title("OU processes",fontsize = 20)
axs[0,1].set_xticks(np.arange(4), (np.rint(2 ** subset["X"].unique())).astype(int),fontsize = 14)
axs[0,1].legend([],[],frameon= False)
axs[0,1].set_ylim([0,80])
# d)
subset = df_t[df_t["Case"] == 6]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,1],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,1].set_xlabel(r"Dimension $d$",fontsize = 16)
axs[1,1].set_ylabel ("")
axs[1,1].set_title("Interacting particles",fontsize = 20)
axs[1,1].set_xticks( np.arange(4),(np.rint(2 ** subset["X"].unique())).astype(int),fontsize = 14)
axs[1,1].legend([],[],frameon= False)
axs[1,1].set_ylim([0,80])
# e)
subset = df_t[df_t["Case"] == 3]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[0,2],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[0,2].set_xlabel (r"Time interval $\Delta t$",fontsize = 16)
axs[0,2].set_ylabel ("")
axs[0,2].set_title("Linear vs. nonlinear",fontsize = 20)
axs[0,2].set_xticks(np.arange(4), (1./np.rint(2 ** subset["X"].unique())).astype(float),fontsize = 14)
axs[0,2].legend([],[],frameon= False)
axs[0,2].set_ylim([0,100])
# f)
subset = df_t[df_t["Case"] == 7]
sns.boxplot(x="X", y="Y",
            hue="Type",ax = axs[1,2],fliersize = 2, linewidth = 1,
            data=subset.iloc[8:])
axs[1,2].set_xlabel(r"Time interval $\Delta t$",fontsize = 16)
axs[1,2].set_ylabel ("")
axs[1,2].set_title("Interacting particles",fontsize = 20)
axs[1,2].set_xticks( np.arange(4),(4./np.rint(2 ** subset["X"].unique())).astype(float),fontsize = 14)
axs[1,2].legend([],[],frameon= False)
axs[1,2].set_ylim([0,100])

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

axs[1,1].legend(ncol = 3,frameon = False, loc = "lower right",fontsize = 14)
fig.tight_layout()
fig.savefig(root+"/duration_plot.pdf")
# %%
