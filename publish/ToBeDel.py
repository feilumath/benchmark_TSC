import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ml_collections
from utils.add_paths import add_paths
import datetime
import matplotlib.colors as mcolors


# keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_keys_2022-12-15-11-19-58.npy'
# valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_values_2022-12-15-11-19-58.npy'
keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_keys_2022-12-15-12-38-53.npy'
valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_values_2022-12-15-12-38-53.npy'
SubPathName = 'Different_potentials/'
Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)

res_all_keys = np.load(keysforsave_str, allow_pickle=True)
res_all_values = np.load(valuesforsave_str, allow_pickle=True)
res_all_len = res_all_keys.__len__()
res_load = ml_collections.ConfigDict()
for i in range(res_all_len):
    setattr(res_load, res_all_keys[i], res_all_values[i])

keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_keys_2022-12-15-11-19-58.npy'
valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_values_2022-12-15-11-19-58.npy'
SubPathName = 'Different_potentials/'
Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)

res_all_keys = np.load(keysforsave_str, allow_pickle=True)
res_all_values = np.load(valuesforsave_str, allow_pickle=True)
res_all_len = res_all_keys.__len__()
for i in range(res_all_len):
    setattr( res_load, res_all_keys[i], np.concatenate((getattr(res_load, res_all_keys[i]),res_all_values[i])) )

num_cases       = res_load.description.__len__()
num_dims_all_np = np.zeros(num_cases)
num_runs        = 20
for i in range(int(num_cases/2)):
    num_dims_all_np[i] = int(100 * 2 ** int(np.floor(i / num_runs)))
for i in range(int(num_cases/2), num_cases):
    num_dims_all_np[i] = int(100 * 2 ** int(np.floor(i / num_runs) - 4))
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
        mid_auc_errorbar_np[0, j] = mid_auc_mean_np[j] - np.sort(mid_mid_auc_np)[3]
        mid_auc_errorbar_np[1, j] = -mid_auc_mean_np[j] + np.sort(mid_mid_auc_np)[-4]
    setattr(res_load, 'auc_mean' + name_app_list[i] + '_np', mid_auc_mean_np)
    setattr(res_load, 'auc_errorbar' + name_app_list[i] + '_np', mid_auc_errorbar_np)




def plot_here(res_load, x_toplot, *, name_app_list, name_app_list_saved, time_str, Sub_Results_path, str_additional):
    linestyle_tuple_list_saved = [
        ('dashed', (0, (5, 5))),
        ('dotted', (0, (1, 1))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),

        ('long dash with offset', (5, (10, 3))),
        ('loosely dashed', (0, (5, 10))),
        ('densely dashed', (0, (5, 1))),
        ('loosely dotted', (0, (1, 10)))]
    color_list_saved     = list(mcolors.TABLEAU_COLORS)
    mid_ind              = [name_app_list_saved.index(i) for i in name_app_list]
    linestyle_tuple_list = [linestyle_tuple_list_saved[i] for i in mid_ind]
    color_list           = [color_list_saved[i] for i in mid_ind]
    font = {'size': 22}
    matplotlib.rc('font', **font)
    fig1 = plt.figure(figsize=(2 * 5, 2 * 3))  # default: [6.4, 4.8]
    plt.plot(x_toplot, getattr(res_load, 'auc_mean' + '_hiddentrue' + '_np'), c='black', lw=3, ls='-', label='Continuous hidden truth')
    plt.plot(x_toplot, getattr(res_load, 'auc_mean' + '_simpleapprox' + '_np'), c=color_list_saved[3], lw=3.5,
             ls=linestyle_tuple_list_saved[4][1], label='Numerical approximation')
    for i in range(name_app_list.__len__()):
        plt.errorbar(x_toplot, getattr(res_load, 'auc_mean' + name_app_list[i] + '_np'), c=color_list[i],
                     ls=linestyle_tuple_list[i][1], lw=2.5, label=name_app_list[i][6:],
                     yerr=getattr(res_load, 'auc_errorbar' + name_app_list[i] + '_np'), capsize=10)
    plt.xlabel("Train size", labelpad=-4.0)
    plt.ylabel("AUC")
    plt.legend()
    plt.xticks(x_toplot, 100 * 2**x_toplot)
    # plotforsave_str = Sub_Results_path + str_additional + time_str[:-4] + '.png'
    plotforsave_str = 'C:/Users/zzh19/Desktop/Plots/Example_trainsize_' + str_additional[1:-1] + '.pdf'
    plt.savefig(plotforsave_str)
    plt.savefig(plotforsave_str)


# %%
# plot 1: To compare all
time_str = '{}-{}-{}-{}.npy'.format(datetime.date.today(), datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second)
x_toplot = np.log2(num_dims_all_unique_np/100)
name_app_list = ['_test_ResNet',
                 '_test_RF', '_test_ROCKRT']
name_app_list_saved = name_app_list

plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
          time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/auc_all_')

# # plot 2: To compare RF vs ROCKET
# name_app_list = ['_test_RF', '_test_ROCKRT']
# plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
#           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/auc_RFROCKET_')
#
# # plot 3: To compare ResNet vs RF
# name_app_list = ['_test_ResNet', '_test_RF']
# plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
#           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/auc_NNRF_')
#
# # plot 4: To compare ResNet vs ROCKET
# name_app_list = ['_test_ResNet', '_test_ROCKRT']
# plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
#           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/auc_NNROCKET_')

# plot 5: RF
name_app_list = ['_test_RF']
plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
          time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/auc_RF_')

# plot 6: ResNet
name_app_list = ['_test_ResNet']
plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
          time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/auc_NN_')

# plot 7: ROCKET
name_app_list = ['_test_ROCKRT']
plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
          time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/auc_ROCKET_')


print("Plot Finished")



