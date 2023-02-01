import numpy as np
import logging
import os
import time
import datetime
import ml_collections  # For data saving
from sklearn.metrics import roc_curve, auc

from data.generateData import TrjsGen
from data.generateLRT import LRTClass

from models.models import NN_fit_predict, RF_fit_predict, ROCKET_fit_predict

from utils.logging_set import LoggingSet
from utils.load_para import ParaResnet, ParaMLP
from utils.plot import plot_my_roc_version2
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# system arguments:
# 1. case number
# 2. number of samples


def main(*, case_str, num_samples, para_dict_tuple, new_trainvaltest_tuple=None):
    """
    The main parts: generate sde -> get LRT -> model fit and predict -> get result acc, auc
    Input:
        case_str: from "CASE1", "moredim_case_OU", "Different_potentials", "OU_process", "constant_drift", "moredim_case_Opinion"
        para_dict_tuple: thple of two para_dict, each para_dict should contain info like default settings in
                         data.generateData.TrjsGen's initialization part
        new_trainvaltest_tuple: Set=None to use setting of train/val/test size from load_para.py.
                                Set=(train_size, val_size, test_size, train_size_withoutval), where train_size_withoutval=train_size+val_size
    Output:
        res: A ml_collections type var saving all info
        logging_save_path_pre: A path indicating where to save the data

    """
    """
    Generate Trjs
    """
    print("@@@@@@@@@@@@ Test for --------Example: %s -------- started: @@@@@@@@@@@@" % case_str)
    (para_dict1, para_dict2) = para_dict_tuple
    DS_Distri1 = 0  # set=0 to get same dt; set=1 or 2 will get time-grids with different dt, 2 will be more imbalance
    trjs_class = TrjsGen(case_str,num_samples, DS_Distri=DS_Distri1, para_dict=para_dict1)
    output1, time_grid1, data_prefix1, para_dict1 = trjs_class.get_final_trjs()
    output_ori1, time_grid_ori1 = trjs_class.get_ori_trjs()

    DS_Distri2 = DS_Distri1
    trjs_class = TrjsGen(case_str,num_samples, DS_Distri=DS_Distri2, para_dict=para_dict2)
    output2, time_grid2, data_prefix2, para_dict2 = trjs_class.get_final_trjs()
    output_ori2, time_grid_ori2 = trjs_class.get_ori_trjs()

    """
    Get LRT
    """
    # Get LRT
    LRT_class_hiddentrue = LRTClass((output_ori1, output_ori2), (time_grid_ori1, time_grid_ori2), (0, 1),
                                    (data_prefix1, data_prefix2),
                                    para_dict_tuple=(para_dict1, para_dict2), NameforSave_str='ori', MustGenNew=0)
    LRT_hiddentrue       = LRT_class_hiddentrue.get_all_LRT()
    label_true           = LRT_class_hiddentrue.get_label()

    LRT_class            = LRTClass((output1, output2), (time_grid1, time_grid2), (0, 1), (data_prefix1, data_prefix2),
                                    para_dict_tuple=(para_dict1, para_dict2), NameforSave_str='final', MustGenNew=0)
    LRT_simpleapprox     = LRT_class.get_all_LRT()
    if case_str[:10].upper() == "OU_process".upper():
        LRT_low_frequency = LRT_class.get_all_LRT_low_frequency()
    else:
        LRT_low_frequency = None

    """
    Model fit and predict
    """
    # Get feature
    featureOnlyTime_triple_np    = LRT_class.get_all_time_augment()

    # First model: ResNet
    duration_all = np.zeros(3)
    start_time = time.time()
    fpr_test_ResNet, tpr_test_ResNet = \
    NN_fit_predict(ParaResnet, para_dict_tuple=(para_dict1, para_dict2),
                   data_prefix_tuple=(data_prefix1, data_prefix2),
                   label_true=label_true, LRT_hiddentrue=LRT_hiddentrue, LRT_simpleapprox=LRT_simpleapprox,
                   name_prefix='ResNet_', feature_triple_np=featureOnlyTime_triple_np,
                   new_trainvaltest_tuple=new_trainvaltest_tuple)
    duration_all[0] = time.time() - start_time

    # Second model: RF
    start_time = time.time()
    fpr_test_RF, tpr_test_RF = \
    RF_fit_predict(ParaMLP, para_dict_tuple=(para_dict1, para_dict2),
                   data_prefix_tuple=(data_prefix1, data_prefix2),
                   label_true=label_true, LRT_hiddentrue=LRT_hiddentrue, LRT_simpleapprox=LRT_simpleapprox,
                   LRT_low_frequency=LRT_low_frequency,
                   feature_triple_np=featureOnlyTime_triple_np,
                   new_trainvaltest_tuple=new_trainvaltest_tuple)
    duration_all[1] = time.time() - start_time

    # Third model: ROCKET
    start_time = time.time()
    fpr_test_ROCKRT, tpr_test_ROCKRT = \
    ROCKET_fit_predict(ParaMLP, para_dict_tuple=(para_dict1, para_dict2),
                       data_prefix_tuple=(data_prefix1, data_prefix2),
                       label_true=label_true, LRT_hiddentrue=LRT_hiddentrue,
                       LRT_simpleapprox=LRT_simpleapprox, LRT_low_frequency=LRT_low_frequency,
                       feature_triple_np=featureOnlyTime_triple_np,
                       output1=output1, new_trainvaltest_tuple=new_trainvaltest_tuple)
    duration_all[2] = time.time() - start_time

    # logging info
    logging_save_path_pre, logging_save_path = LoggingSet(logging, para_dict_tuple=(para_dict1, para_dict2),
                                                          data_prefix_tuple=(data_prefix1, data_prefix2),
                                                          loggingname_prestr='All_')

    """
    Plot and save
    """
    # Get fpr and tpr for roc-curve plot
    fpr_simpleapprox, tpr_simpleapprox, thresholds_simpleapprox     = roc_curve(label_true, -LRT_simpleapprox, drop_intermediate=True)
    fpr_hiddentrue, tpr_hiddentrue, thresholds_hiddentrue           = roc_curve(label_true, -LRT_hiddentrue, drop_intermediate=True)
    if case_str[:10].upper() == "OU_process".upper():
        fpr_low_frequency, tpr_low_frequency, thresholds_low_frequency = roc_curve(label_true, -LRT_low_frequency, drop_intermediate=True)
    else:
        fpr_low_frequency = None
        tpr_low_frequency = None
        thresholds_low_frequency = None

    # Plot the roc-curve
    plot_my_roc_version2(logging_save_path, addition_str='CompareAll',
                         fpr_test_NN=fpr_test_ResNet, tpr_test_NN=tpr_test_ResNet,
                         fpr_test_RF=fpr_test_RF, tpr_test_RF=tpr_test_RF, fpr_test_ROCKRT=fpr_test_ROCKRT, tpr_test_ROCKRT=tpr_test_ROCKRT,
                         fpr_simpleapprox=fpr_simpleapprox, tpr_simpleapprox=tpr_simpleapprox,
                         fpr_low_frequency=fpr_low_frequency, tpr_low_frequency=tpr_low_frequency,
                         fpr_hiddentrue=fpr_hiddentrue, tpr_hiddentrue=tpr_hiddentrue)

    # Save res as ml_collections, not a dictionary; i.e. use res.auc_test_ResNet instead of res['auc_test_ResNet']
    res = ml_collections.ConfigDict()
    name_app_list = ['_test_ResNet',
                     '_test_RF', '_test_ROCKRT',
                     '_simpleapprox', '_hiddentrue']
    name_pre_list = ['fpr', 'tpr']
    for i in range(name_pre_list.__len__()):
        for j in range(name_app_list.__len__()):
            setattr(res, name_pre_list[i] + name_app_list[j], locals()[name_pre_list[i] + name_app_list[j]] )
    for j in range(name_app_list.__len__()):
        setattr(res, 'auc' + name_app_list[j], auc( locals()['fpr' + name_app_list[j]], locals()['tpr' + name_app_list[j]]) )
        setattr(res, 'acc' + name_app_list[j], ( - locals()['fpr' + name_app_list[j]] + locals()['tpr' + name_app_list[j]] + 1.0)/2.0 )
        setattr(res, 'acc_star' + name_app_list[j], np.max( getattr(res, 'acc' + name_app_list[j])) )
    setattr(res, 'duration_all', duration_all)

    # Log the time-consuming
    logging.info('ResNet total duration is %.3f' % duration_all[0])
    logging.info('RF total duration is %.3f' % duration_all[1])
    logging.info('ROCKET total duration is %.3f' % duration_all[2])

    print("@@@@@@@@@@@@ Test for --------Example: %s -------- finished: @@@@@@@@@@@@" % case_str)

    return res, logging_save_path_pre

# %%
if __name__ == '__main__':

    from utils.utils import get_para_dict_all_from_tableline, get_new_trainvaltest_tuple
    '''
    Set 1. case_str: if table_for_alg_design_ind is given, then this is useless. Do not need to specify here.
        2. use_different_trainvaltest_size: In general be False in general runs. Set as true if we want to use a 
            different setting of train/val/test size from load_para.py. 
        3. table_for_alg_design_ind: The ind of lines of the table for data generation, eg: line 1 should be:
            Constant drift, dim=1, nsp=10, 20, 40, 80 T=1, 2, 4, 8, dt=0.1
            Customer can modify get_para_dict_all_from_tableline by your own design, eg: set=8 and create setting for 
            this case.
    '''
    # case_str = "constant_drift"
    table_for_alg_design_ind = int(sys.argv[1])
    # Now, only line 10 needs to change train/val/test size
    use_different_trainvaltest_size = (table_for_alg_design_ind==10)
    res_all = ml_collections.ConfigDict()
    name_app_list = ['_test_ResNet',
                     '_test_RF', '_test_ROCKRT',
                     '_simpleapprox', '_hiddentrue']
    name_pre_list = ['fpr', 'tpr', 'auc', 'acc', 'acc_star']
    for i in range(name_pre_list.__len__()):
        for j in range(name_app_list.__len__()):
            setattr(res_all, name_pre_list[i] + name_app_list[j] + '_tuple', tuple() )
    setattr(res_all, 'duration_all' + '_tuple', tuple())
    res_all.description = tuple()

    num_runs     = 40  # For each sub-case, number of run; For simple test, set=1; For paper design, set=40
    num_subcases = 4  # The number of different setting from one line of the table in paper; Now paper set=4
    for outest_ind in range(num_runs * num_subcases):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ This is the %d run @@@@@@@@@@@@@@@@@@@@@@' % outest_ind)
        # Get all setting from ind of lines of the table for data generation in the paper
        para_dict1, para_dict2, case_str = get_para_dict_all_from_tableline(table_for_alg_design_ind,
                                                                            outest_ind=outest_ind, num_runs=num_runs)
        # If we want to use a different setting of train/val/test size from load_para.py. In general do not use this
        new_trainvaltest_tuple = get_new_trainvaltest_tuple(use_different_trainvaltest_size,
                                                            outest_ind=outest_ind, num_runs=num_runs)
        # The main parts: generate sde -> get LRT -> model fit and predict -> get result acc, auc
        res, logging_save_path_pre = main(case_str=case_str,num_samples= int(sys.argv[2]), para_dict_tuple=(para_dict1, para_dict2),
                                          new_trainvaltest_tuple=new_trainvaltest_tuple)
        # Save all res as res_all
        for i in range(name_pre_list.__len__()):
            for j in range(name_app_list.__len__()):
                setattr(res_all, name_pre_list[i] + name_app_list[j] + '_tuple',
                        getattr(res_all, name_pre_list[i] + name_app_list[j] + '_tuple') + (
                        getattr(res, name_pre_list[i] + name_app_list[j]),))
        setattr(res_all,  'duration_all' + '_tuple',
                getattr(res_all, 'duration_all' + '_tuple') + (getattr(res, 'duration_all'),) )
        res_all.description = res_all.description + (logging_save_path_pre[logging_save_path_pre.rindex('/') + 1:],)

    # Save res_all
    time_str          = '{}-{}-{}-{}.npy'.format(datetime.date.today(), datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second)
    keysforsave_str   = logging_save_path_pre[:logging_save_path_pre.rindex('/')] + '/res_all_keys_' + time_str
    valuesforsave_str = logging_save_path_pre[:logging_save_path_pre.rindex('/')] + '/res_all_values_' + time_str

    np.save(keysforsave_str, np.array(res_all.keys(), dtype=str))
    np.save(valuesforsave_str, np.array(res_all.values(), dtype=object))


    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  All finished!  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")








