import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ml_collections
from utils.add_paths import add_paths
import datetime
import matplotlib.colors as mcolors


def main(*, Majorind, Toycase=False):

    if Majorind == 1:
        keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/constant_drift_BM/res_all_keys_2022-12-15-23-31-55.npy'
        valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/constant_drift_BM/res_all_values_2022-12-15-23-31-55.npy'
        SubPathName = 'constant_drift_BM/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 1
    elif Majorind == 2:
        keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/moredim_OU/res_all_keys_2022-12-16-2-17-14.npy'
        valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/moredim_OU/res_all_values_2022-12-16-2-17-14.npy'
        SubPathName = 'moredim_OU/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 2
    elif Majorind == 3:
        keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/data_ts/linear v.s. nonlinear/res_all_keys_2022-12-16-4-32-4.npy'
        valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/data_ts/linear v.s. nonlinear/res_all_values_2022-12-16-4-32-4.npy'
        SubPathName = 'data_ts/linear v.s. nonlinear/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 3
    elif Majorind == 4:
        keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_keys_2022-12-16-10-41-58.npy'
        valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_values_2022-12-16-10-41-58.npy'
        SubPathName = 'Different_potentials/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 4
    elif Majorind == 5:
        keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_keys_2022-12-16-11-31-55.npy'
        valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/Different_potentials/res_all_values_2022-12-16-11-31-55.npy'
        SubPathName = 'Different_potentials/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 5
    elif Majorind == 6:
        keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/moredim_Opinion/res_all_keys_2022-12-16-14-11-33.npy'
        valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/moredim_Opinion/res_all_values_2022-12-16-14-11-33.npy'
        SubPathName = 'moredim_Opinion/'
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(SubPathName)
        tableline = 6
    elif Majorind == 7:
        keysforsave_str   = 'C:/Users/zzh19/sigUCts/results/moredim_Opinion/res_all_keys_2022-12-17-0-24-45.npy'
        valuesforsave_str = 'C:/Users/zzh19/sigUCts/results/moredim_Opinion/res_all_values_2022-12-17-0-24-45.npy'
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
                     '_test_RF', '_test_ROCKRT']

    mid_duration_all_tuple = getattr(res_load, 'duration_all' + '_tuple')
    mid_all_cases = mid_duration_all_tuple.__len__()
    mid_duration_all = np.zeros((mid_all_cases, name_app_list.__len__()))
    for i in range(mid_all_cases):
        for j in range(name_app_list.__len__()):
            mid_duration_all[i, j] = mid_duration_all_tuple[i][j]

    for i in range(name_app_list.__len__()):
        mid_duration_mean_np = np.zeros(num_x_toplot)
        mid_duration_errorbar_np = np.zeros((2, num_x_toplot))
        for j in range(num_x_toplot):
            mid_mid_duration_np = mid_duration_all[np.where(num_dims_all_np == num_dims_all_unique_np[j]), i].reshape((-1))
            mid_duration_mean_np[j] = np.mean(mid_mid_duration_np)
            if Toycase:
                mid_duration_errorbar_np[0, j] = mid_duration_mean_np[j] - np.sort(mid_mid_duration_np)[0]
                mid_duration_errorbar_np[1, j] = -mid_duration_mean_np[j] + np.sort(mid_mid_duration_np)[-1]
            else:
                mid_duration_errorbar_np[0, j] = mid_duration_mean_np[j] - np.sort(mid_mid_duration_np)[2]
                mid_duration_errorbar_np[1, j] = -mid_duration_mean_np[j] + np.sort(mid_mid_duration_np)[-3]
        setattr(res_load, 'duration_mean' + name_app_list[i] + '_np', mid_duration_mean_np)
        setattr(res_load, 'duration_errorbar' + name_app_list[i] + '_np', mid_duration_errorbar_np)




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
        # plt.plot(x_toplot, getattr(res_load, 'auc_mean' + '_hiddentrue' + '_np'), c='black', lw=3, ls='-', label='Continuous hidden truth')
        # plt.plot(x_toplot, getattr(res_load, 'auc_mean' + '_simpleapprox' + '_np'), c=color_list_saved[3], lw=3.5, ls=linestyle_tuple_list_saved[4][1], label='Numerical approximation')
        for i in range(name_app_list.__len__()):
            plt.errorbar(x_toplot, getattr(res_load, 'duration_mean' + name_app_list[i] + '_np'), c=color_list[i],
                         ls=linestyle_tuple_list[i][1], lw=2.5, label=name_app_list[i][6:],
                         yerr=getattr(res_load, 'duration_errorbar' + name_app_list[i] + '_np'), capsize=10)
        if (SubPathName == 'constant_drift_BM/') and (tableline == 1):
            plt.xlabel(r"End Time $t_L$", labelpad=-4.0)
            plt.xticks(x_toplot, (np.rint(2 ** x_toplot)).astype(int))
            case_name = 'Section5_Line1'
        elif (SubPathName == 'moredim_OU/') and (tableline == 2):
            plt.xlabel("Dimensions", labelpad=-4.0)
            plt.xticks(x_toplot, (np.rint(2 ** x_toplot)).astype(int))
            case_name = 'Section5_Line2'
        elif (SubPathName == 'data_ts/linear v.s. nonlinear/') and (tableline == 3):
            plt.xlabel("dt", labelpad=-4.0)
            plt.xticks(x_toplot, (1.0/np.rint(2 ** x_toplot)).astype(float))
            case_name = 'Section5_Line3'
        elif (SubPathName == 'Different_potentials/') and (tableline == 4):
            plt.xlabel(r"End Time $t_L$", labelpad=-4.0)
            plt.xticks(x_toplot, (np.rint(2 ** x_toplot)).astype(int))
            case_name = 'Section5_Line4'
        elif (SubPathName == 'Different_potentials/') and (tableline == 5):
            plt.xlabel(r"End Time $t_L$", labelpad=-4.0)
            plt.xticks(x_toplot, (np.rint(2 ** x_toplot)).astype(int))
            case_name = 'Section5_Line5'
        elif (SubPathName == 'moredim_Opinion/') and (tableline == 6):
            plt.xlabel("Dimensions", labelpad=-4.0)
            plt.xticks(x_toplot, (np.rint(2 ** x_toplot)).astype(int))
            case_name = 'Section5_Line6'
        elif (SubPathName == 'moredim_Opinion/') and (tableline == 7):
            plt.xlabel("dt", labelpad=-4.0)
            plt.xticks(x_toplot, (1.0 / np.rint(2 ** x_toplot)).astype(float))
            case_name = 'Section5_Line7'

        plt.ylabel("Duration")
        plt.legend()
        # plotforsave_str = Sub_Results_path + str_additional + time_str[:-4] + '.pdf'
        plotforsave_str = 'C:/Users/zzh19/Desktop/Plots/' + case_name + '_Duration.pdf'
        plt.savefig(plotforsave_str)
        plt.close()


    # %%
    # plot 1: To compare all
    time_str = '{}-{}-{}-{}.npy'.format(datetime.date.today(), datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second)
    x_toplot = np.log2(num_dims_all_unique_np)
    name_app_list = ['_test_ResNet',
                     '_test_RF', '_test_ROCKRT']
    name_app_list_saved = name_app_list

    plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
              time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/duration_all_')

    # # plot 2: To compare RF vs ROCKET
    # name_app_list = ['_test_RF', '_test_ROCKRT']
    # plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
    #           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/duration_RFROCKET_')
    #
    # # plot 3: To compare ResNet vs RF
    # name_app_list = ['_test_ResNet', '_test_RF']
    # plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
    #           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/duration_NNRF_')
    #
    # # plot 4: To compare ResNet vs ROCKET
    # name_app_list = ['_test_ResNet', '_test_ROCKRT']
    # plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
    #           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/duration_NNROCKET_')
    #
    # # plot 5: RF
    # name_app_list = ['_test_RF']
    # plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
    #           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/duration_RF_')
    #
    # # plot 6: ResNet
    # name_app_list = ['_test_ResNet']
    # plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
    #           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/duration_NN_')
    #
    # # plot 7: ROCKET
    # name_app_list = ['_test_ROCKRT']
    # plot_here(res_load, x_toplot, name_app_list=name_app_list, name_app_list_saved=name_app_list_saved,
    #           time_str=time_str, Sub_Results_path=Sub_Results_path, str_additional='/duration_ROCKET_')

    print("This Plot Finished")


if __name__ == '__main__':
    for i in range(7):
        Majorind = i + 1
        main(Majorind=Majorind, Toycase=True)
    print("All Plot Finished")



