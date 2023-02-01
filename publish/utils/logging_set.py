import os
from utils.add_paths import add_paths
import datetime


def LoggingSet(logging, *, para_dict_tuple, data_prefix_tuple, loggingname_prestr=str()):
    """
    Input:
        para_dict_tuple and data_prefix_tuple: MUST be tuple as (1, 2). CANNOT be tuple >=3
        Set logging paths according to cases by the paper.
        Then each logging file will be saved under folder named [paths of trjs1] __VS__ [paths of trjs2]
    Output:
        Case_Results_path: The path of logging file
        my_path: The whole path = Case_Results_path + name(some.txt) of logging file
    """
    # Set CaseName, para_dict1['NewFileName'] and para_dict2['NewFileName'] for the path of the logging file
    para_dict1, para_dict2     = para_dict_tuple
    data_prefix1, data_prefix2 = data_prefix_tuple
    if para_dict1['SubPathName']=='data_ts/':
        para_dict1['NewFileName'] = data_prefix1[data_prefix1.index(para_dict1['SubPathName']) + 9:]
        para_dict2['NewFileName'] = data_prefix2[data_prefix2.index(para_dict2['SubPathName']) + 9:-1]
        if para_dict1['nonlinear']==para_dict2['nonlinear'] and para_dict1['nonlinear']==0:
            CaseName = 'Markovian v.s. non-Markovian, linear/'
        elif para_dict1['nonlinear']==para_dict2['nonlinear'] and para_dict1['nonlinear']>=1:
            CaseName = 'Markovian v.s. non-Markovian, nonlinear/'
        elif not para_dict1['nonlinear']==para_dict2['nonlinear']:
            CaseName = 'linear v.s. nonlinear/'
    elif (para_dict1['SubPathName']=='OU_process/') or (para_dict1['SubPathName']=='Different_potentials/') \
            or (para_dict1['SubPathName']=='moredim_OU/') or (para_dict1['SubPathName']=='moredim_Opinion/')\
            or (para_dict1['SubPathName']=='constant_drift_BM/'):
        para_dict1['NewFileName'] = data_prefix1[data_prefix1.index(para_dict1['SubPathName']) + para_dict1['SubPathName'].__len__() + 1:]
        if '_p_' in data_prefix1:
            para_dict2['NewFileName'] = data_prefix2[data_prefix2.index('_p_'): data_prefix2.index('_nl_')]
        elif 'cd_' in data_prefix1:
            para_dict2['NewFileName'] = data_prefix2[data_prefix2.index('cd_'): data_prefix2.index('_nl_')]
        elif 'dn_' in data_prefix1:
            para_dict2['NewFileName'] = data_prefix2[data_prefix2.index('dn_'): data_prefix2.index('_nl_')]
        CaseName = str()
    else:
        raise ValueError(f"para_dict1['SubPathName'] is not 'data_ts/' or 'moredim_OU/' or 'constant_drift_BM/' or 'OU_process/' "
                         f"or 'Different_potentials/' or 'moredim_Opinion/', got {para_dict1['SubPathName']}")
    Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(para_dict1['SubPathName'])

    # The path of the logging file
    Case_Results_path = Sub_Results_path + CaseName + para_dict1['NewFileName'][:-1]  + "__VS__" + para_dict2['NewFileName']
    if not os.path.exists(Case_Results_path):
        os.makedirs(Case_Results_path)

    # Set logging, where time_str: name of the logging file
    handler1 = logging.StreamHandler()
    time_str = loggingname_prestr + 'MyLogging{}-{}-{}-{}.txt'.format(datetime.date.today(), datetime.datetime.now().hour,
                                                          datetime.datetime.now().minute, datetime.datetime.now().second)
    my_path = os.path.join(Case_Results_path, time_str)
    handler2 = logging.FileHandler(my_path, 'w')
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel('INFO')
    return Case_Results_path, my_path















