import numpy as np
import copy
from scipy.spatial.distance import cdist


class PreData:
    """
    Input allowed tuple, like (Trjs1, Trjs2, Trjs3), for unpacked inputs:
    Trjs       : if in [num_samples, num_steps, Num_dims] or [num_samples, num_steps], is trajectories of the same setting
    TimeGrids  : if in R, is End time
                 if in [num_steps, Num_dims] or [num_steps], which should time grid of x_paths when they share same time grid
                 if in [num_samples, num_steps, Num_dims] or [num_samples, num_steps], which should be same as x_paths's
    label      : in R or [num_samples]
    data_prefix: is str, the location(contains file name) of the trjs
    Then the shape of TimeGrids will be same as Trjs;
         the shape of label will be [num_samples]
    """
    def __init__(self, Trjs, TimeGrids, label, data_prefix):
        if isinstance(Trjs, tuple):
            self.Output_Tuple = True
            self.Trjs = Trjs
            self.TimeGrids = TimeGrids
            self.label = label
            self.data_prefix = data_prefix
        else:
            self.Output_Tuple = False
            self.Trjs = (Trjs,)
            self.TimeGrids = (TimeGrids,)
            self.label = (label,)
            self.data_prefix = (data_prefix,)
        self.tuple_len = Trjs.__len__()
        assert self.tuple_len == TimeGrids.__len__(), 'Number of cases is different from TimeGrids'
        assert self.tuple_len == label.__len__(), 'Number of cases is different from label'
        assert self.tuple_len == data_prefix.__len__(), 'Number of cases is different from data_prefix'
        self.TimeReshape()
        self.LabelReshape()

    def TimeReshape(self):
        """
        Make the shape of TimeGrids be the same as Trjs'
        """
        time_grids_tuple = tuple()
        for tuple_ind in range(self.tuple_len):
            x_paths = self.Trjs[tuple_ind]
            time_grids = self.TimeGrids[tuple_ind]
            num_samples = x_paths.shape[0]
            num_steps = x_paths.shape[1]
            if (isinstance(time_grids, np.ndarray) and time_grids.shape.__len__() > 0):
                if x_paths.shape[:2] == time_grids.shape[:2]:
                    T_paths = time_grids
                elif x_paths.shape[1] == time_grids.shape[0]:
                    T_paths = np.repeat(np.expand_dims(time_grids, 0), num_samples, axis=0)
            else:
                T_paths = np.repeat(np.expand_dims(np.linspace(0, time_grids, num_steps), 0), num_samples, axis=0)
            time_grids_tuple = time_grids_tuple + (T_paths,)
        self.TimeGrids = time_grids_tuple

    def LabelReshape(self):
        """
        Make the shape of label be [num_samples]
        """
        lable_re_tuple = tuple()
        for tuple_ind in range(self.tuple_len):
            x_paths   = self.Trjs[tuple_ind]
            lable_mid = self.label[tuple_ind]
            num_samples = x_paths.shape[0]
            if isinstance(lable_mid, np.ndarray):
                if x_paths.shape[0] == lable_mid.shape[0]:
                    lable_re = lable_mid
            else:
                lable_re = np.repeat(np.expand_dims(lable_mid, 0), num_samples, axis=0)
            lable_re_tuple = lable_re_tuple + (lable_re,)
        self.label = lable_re_tuple


def logging_best_model(logging, hist, duration):
    """
    Log the number of epoch that best model is obtained
    """
    hist_val_loss  = np.array(hist.history['val_loss'])
    ind_best_model = np.argmin(hist_val_loss)
    logging.info('The best val_loss is at epoch %d' % ind_best_model)
    logging.info('The training duration is %.4f' % duration)


def OD_influence(r, *, type_ind):
    """
    The kernel for opinion system
    """
    cutoff = 2.0 / np.sqrt(2.0)
    support = 2.0
    influence = 0.0
    if type_ind == 1:
        if (0 < r) and (r < cutoff):
            influence = 1.0 * 2
        elif (cutoff <= r) and (r < support):
            influence = 0.1 * 2
    elif type_ind == 2:
        if (0 < r) and (r < cutoff):
            influence = 1.0
        elif (cutoff <= r) and (r < support):
            influence = 1.0
    elif type_ind == 3:
        if (0 < r) and (r < cutoff):
            influence = 0.5
        elif (cutoff <= r) and (r < support):
            influence = 1.0
    elif type_ind == 4:
        if (0 < r) and (r < cutoff):
            influence = 0.1 * 2
        elif (cutoff <= r) and (r < support):
            influence = 1.0 * 2
    elif type_ind == 5:
        delta = 0.05
        if (0 < r) and (r < (cutoff - delta)):
            influence = 1
        elif ((cutoff - delta) <= r) and (r < (cutoff + delta)):
            y_1 = 1.0
            y_2 = 0.1
            influence = (y_2 - y_1) / (-2.0) * (np.cos(np.pi / (2.0 * delta) * (r - (cutoff - delta))) - 1.0) + y_1
        elif ((cutoff + delta) <= r) and (r < (support - delta)):
            influence = 0.1
        elif ((support - delta) <= r) and (r < (support + delta)):
            y_1 = 0.1
            y_2 = 0.0
            influence = (y_2 - y_1) / (-2.0) * (np.cos(np.pi / (2.0 * delta) * (r - (support - delta))) - 1.0) + y_1
    elif type_ind == 6:
        influence = r ** 3.0
    elif type_ind == 7:
        influence = np.sin(r)
    else:
        print('Select a type from 1-5 only! \n')
    return influence


def opinion_drift_function_pre(x0, t, p, num_agents, num_dims):
    """
    The drift function for opinion system
    """
    XX0         = x0.reshape((num_agents, num_dims))
    XX0_dis     = cdist(XX0, XX0)
    phi_XX0_dis = np.zeros_like(XX0_dis)
    for i in range(num_agents):
        for j in range(num_agents):
            phi_XX0_dis[i, j] = OD_influence(XX0_dis[i, j], type_ind=p[0])
    func_F = np.zeros((num_dims * num_agents))
    for i in range(num_agents):
        func_F[i * num_dims: (i + 1) * num_dims] = (1 / num_agents) * np.matmul(phi_XX0_dis[i, :], (XX0 - np.repeat(np.expand_dims(XX0[i, :], 0), num_agents, axis=0)))
    return func_F


def get_new_trainvaltest_tuple(use_different_trainvaltest_size, *, outest_ind=None, num_runs=None):
    """
    Use different setting of train/val/test size from load_para.py.
    """
    if use_different_trainvaltest_size is True:
        mew_train_size_withoutval = int(500 * 2 ** int(np.floor(outest_ind / num_runs)))
        new_train_size = int(np.floor(mew_train_size_withoutval * 2 / 3))
        new_val_size   = int(np.floor(mew_train_size_withoutval / 3))
        new_test_size  = int(500)
        new_trainvaltest_tuple = (new_train_size, new_val_size, new_test_size, mew_train_size_withoutval)
    else:
        new_trainvaltest_tuple = None
    return new_trainvaltest_tuple


def get_para_dict_all_from_tableline(table_for_alg_design_ind, *, outest_ind, num_runs,num_samples):
    """
    The code for the table in section 4 in paper
    Customer can create new setting for your own design, eg:
            elif table_for_alg_design_ind == 8:
            ......
    """
    if table_for_alg_design_ind == 1:
        case_str        = "constant_drift"
        maturity        = 1 * 2 ** int(np.floor(outest_ind / num_runs))
        dt              = 0.1
        num_steps_m1    = int(maturity/dt)
        Irre_time_level = int(100 * dt)
        num_dims        = int(1)
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims)
    elif table_for_alg_design_ind == 2:
        case_str        = "moredim_case_OU"
        maturity        = 2.0
        dt              = 0.1
        num_steps_m1    = int(maturity/dt)
        Irre_time_level = int(100 * dt)
        num_dims        = int(1 * 2 ** int(np.floor(outest_ind / num_runs)))
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims)
    elif table_for_alg_design_ind == 3:
        case_str        = "CASE1"
        maturity        = 1.0
        num_steps_m1    = int(10 * 2 ** int(np.floor(outest_ind / num_runs)))
        dt              = float(maturity/num_steps_m1)
        Irre_time_level = int(100 * dt)
        num_dims        = int(1)
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims)
    elif table_for_alg_design_ind == 4:
        case_str        = "Different_potentials"
        maturity        = 2 * 2 ** int(np.floor(outest_ind / num_runs))
        dt              = 0.1
        num_steps_m1    = int(maturity/dt)
        Irre_time_level = int(100 * dt)
        num_dims        = int(1)
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims)
    elif table_for_alg_design_ind == 5:
        case_str        = "Different_potentials"
        maturity        = 5 * 2 ** int(np.floor(outest_ind / num_runs))
        dt              = 1.0
        num_steps_m1    = int(maturity/dt)
        Irre_time_level = int(100 * dt)
        num_dims        = int(1)
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims)
    elif table_for_alg_design_ind == 6:
        case_str        = "moredim_case_Opinion"
        maturity        = 2.0
        dt              = 0.1
        num_steps_m1    = int(maturity/dt)
        Irre_time_level = int(100 * dt)
        num_dims        = 2
        num_agents      = int(3 * 2 ** int(np.floor(outest_ind / num_runs)))
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims, num_agents=num_agents)
    elif table_for_alg_design_ind == 7:
        case_str        = "moredim_case_Opinion"
        maturity        = 4.0
        num_steps_m1    = int(10 * 2 ** int(np.floor(outest_ind / num_runs)))
        dt              = float(maturity/num_steps_m1)
        Irre_time_level = int(100 * dt)
        num_dims        = 2
        num_agents      = 24
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims, num_agents=num_agents)
    elif table_for_alg_design_ind == 8:
        case_str = "moredim_case_Opinion"
        maturity        = 2 * 2 ** int(np.floor(outest_ind / num_runs))
        dt              = 0.1
        num_steps_m1    = int(maturity / dt)
        Irre_time_level = int(100 * dt)
        num_dims        = 2
        num_agents      = 6
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_samples=1000,num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims, num_agents=num_agents)
    elif table_for_alg_design_ind == 9:
        case_str        = "moredim_case_Opinion"
        maturity        = 2.0
        dt              = 0.1
        num_steps_m1    = int(maturity / dt)
        Irre_time_level = int(100 * dt)
        num_dims        = 2
        num_agents      = 6
        p1_for_opinion  = np.array([4, 0.8 * 2 ** int(-np.floor(outest_ind / num_runs))])
        p2_for_opinion  = np.array([1, 0.8 * 2 ** int(-np.floor(outest_ind / num_runs))])
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str,num_samples=1000,num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims,
                                                   num_agents=num_agents, p1_for_opinion=p1_for_opinion,
                                                   p2_for_opinion=p2_for_opinion)
    elif table_for_alg_design_ind == 10:
        case_str        = "moredim_case_Opinion"
        maturity        = 2.0
        dt              = 0.1
        num_steps_m1    = int(maturity / dt)
        Irre_time_level = int(100 * dt)
        num_dims        = 2
        num_agents      = 6
        p1_for_opinion  = np.array([4, 0.4])
        p2_for_opinion  = np.array([1, 0.4])
        para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                                   dt=dt, Irre_time_level=Irre_time_level, num_dims=num_dims,
                                                   num_agents=num_agents, p1_for_opinion=p1_for_opinion,
                                                   p2_for_opinion=p2_for_opinion, num_samples=2500)
    else:
        raise ValueError('Please set table_for_alg_design_ind in [1,2,3,4,5,6,7]')

    return para_dict1, para_dict2, case_str


def get_para_dict_all(*, case_str, num_steps_m1, maturity, dt, num_dims=2, num_agents=20, num_samples=1000,
                      Irre_time_level=100, p1_for_opinion=np.array([4, 1.0]), p2_for_opinion=np.array([1, 1.0])):
    """
    Get para_dict for sde generation
    """
    assert float(num_steps_m1) * dt == float(maturity), 'The number of steps, maturity and dt NOT MATCH!'
    if case_str == "constant_drift":
        para_dict1 = {
            "SubPathName": 'constant_drift_BM/',
            "num_samples": num_samples,  # Number of samples
            "num_steps": num_steps_m1 + 1,  # Number of time steps
            "maturity": float(maturity),  # The maturity of SDE, that is, the end-time of the time interval
            "Irre_time_level": Irre_time_level,  # Int: if 1, Equidis time grid; Larger may cause more irregularity (If not case irr, please set DS_Distri1 = 0)
            "ZeroIni": 0,  # Initial condition, if 1: Zero initial; not 1: standard Normal
            "hurst": 0.5,  # The Hurst parameter
            "nonlinear": 0,  # linear or nonlinear equation
            "constant_drift": 1.0
        }
        para_dict2 = copy.deepcopy(para_dict1)
        para_dict2["constant_drift"] = 2.0

    elif case_str == "Different_potentials":
        para_dict1 = {
            "SubPathName": 'Different_potentials/',
            "num_samples": num_samples,  # Number of samples
            "num_steps": num_steps_m1 + 1,  # Number of time steps
            "maturity": float(maturity),  # The maturity of SDE, that is, the end-time of the time interval
            "Irre_time_level": Irre_time_level,
            "ZeroIni": 0,  # Initial condition, if 1: Zero initial; not 1: standard Normal
            "hurst": 0.5,  # The Hurst parameter
            "nonlinear": 0,  # linear or nonlinear equation
            "constant_drift": 1.0,
            "drift_function": lambda y, t, p: - 2.0 * y * (y - 1) * (y + 1),
            "diffusion_function": lambda y, t, p: 1.0,
            "diffusion_function_dx": lambda y, t, p: 0.0,
            "drift_name": "2y(y-1)(y+1)_1.0"
        }
        para_dict2 = copy.deepcopy(para_dict1)
        para_dict2["drift_function"] = lambda y, t, p: - y * y * y
        para_dict2["diffusion_function"] = lambda y, t, p: 1.0
        para_dict2["drift_name"] = "yyy_1.0"

    elif case_str == "OU_process":
        para_dict1 = {
            "SubPathName": 'OU_process/',
            "num_samples": num_samples,  # Number of samples
            "num_steps": num_steps_m1 + 1,  # Number of time steps
            "maturity": float(maturity),  # The maturity of SDE, that is, the end-time of the time interval
            "Irre_time_level": Irre_time_level,
            "ZeroIni": 0,  # Initial condition, if 1: Zero initial; not 1: standard Normal
            "hurst": 0.5,  # The Hurst parameter
            "nonlinear": 0,  # linear or nonlinear equation
            "constant_drift": -1.0,
        }
        para_dict2 = copy.deepcopy(para_dict1)
        para_dict2["constant_drift"] = -2.0

    elif case_str == "moredim_case_OU":
        para_dict1 = {
            "SubPathName": 'moredim_OU/',
            "num_samples": num_samples,  # Number of samples
            "num_steps": num_steps_m1 + 1,  # Number of time steps
            "maturity": float(maturity),  # The maturity of SDE, that is, the end-time of the time interval
            "Irre_time_level": Irre_time_level,
            "ZeroIni": 0,  # Initial condition, if 1: Zero initial; not 1: standard Normal
            "hurst": 0.5,  # The Hurst parameter
            "nonlinear": 0,  # linear or nonlinear equation
            "num_dims": num_dims,  # Number of dims of y
            "num_dims_noise": num_dims,  # Number of dims of noise
            "drift_function": lambda y, t, p: p[0] * np.matmul(np.eye(num_dims), y),  # n-dim
            "diffusion_function": lambda y, t, p: p[1] * np.eye(num_dims),  # n*m-dim
            "diffusion_function_dx": lambda y, t, p: 0.0,
            "p": np.array([-1.0, 1.0]),
            "additional_name": "id_drift_diff_"
        }
        para_dict2 = copy.deepcopy(para_dict1)
        para_dict2["p"] = np.array([-2.0, 1.0])

    elif case_str == "moredim_case_Opinion":
        opinion_drift_function = lambda x0, t, p: opinion_drift_function_pre(x0, t, p, num_agents, num_dims)
        para_dict1 = {
            "SubPathName": 'moredim_Opinion/',
            "num_samples": num_samples,  # Number of samples
            "num_steps": num_steps_m1 + 1,  # Number of time steps
            "maturity": float(maturity),  # The maturity of SDE, that is, the end-time of the time interval
            "Irre_time_level": Irre_time_level,
            "ZeroIni": 0,  # Initial condition, if 1: Zero initial; not 1: standard Normal
            "hurst": 0.5,  # The Hurst parameter
            "nonlinear": 0,  # linear or nonlinear equation
            "num_dims": num_dims * num_agents,  # Number of dims of y
            "num_dims_noise": num_dims * num_agents,  # Number of dims of noise
            "drift_function": opinion_drift_function,  # n-dim
            "diffusion_function": lambda y, t, p: p[1] * np.eye(num_dims * num_agents),  # n*m-dim
            "diffusion_function_dx": lambda y, t, p: 0.0,
            "p": p1_for_opinion,
            "additional_name": "id_diff_"
        }
        para_dict2 = copy.deepcopy(para_dict1)
        para_dict2["p"] = p2_for_opinion

    elif case_str == "CASE1":
        para_dict1 = {
            "SubPathName": 'data_ts/',
            "num_samples": num_samples,  # Number of samples
            "num_steps": num_steps_m1 + 1,  # Number of time steps
            "maturity": float(maturity),  # The maturity of SDE, that is, the end-time of the time interval
            "Irre_time_level": Irre_time_level,
            "ZeroIni": 0,  # Initial condition, if 1: Zero initial; not 1: standard Normal
            "hurst": 0.5,  # The Hurst parameter
            "nonlinear": 0,  # linear or nonlinear equation
        }
        para_dict2 = copy.deepcopy(para_dict1)
        para_dict2["nonlinear"] = 1
    else:
        raise ValueError('Please set case_str in ["constant_drift", "Different_potentials", "OU_process", "CASE1", "moredim_case_Opinion", "moredim_case_OU"]')

    return para_dict1, para_dict2












