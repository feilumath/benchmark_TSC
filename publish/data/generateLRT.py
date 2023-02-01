import numpy as np
from utils.utils import PreData
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt
from utils.add_paths import add_paths
import os


class LRTClass(PreData):
    def __init__(self, Trjs, TimeGrids, label, data_prefix, *, para_dict_tuple, NameforSave_str='final', MustGenNew=1):
        super(LRTClass, self).__init__(Trjs, TimeGrids, label, data_prefix)
        if not (isinstance(para_dict_tuple, tuple) and para_dict_tuple.__len__() == 2):
            print("We now only can deal with 2 cases!")
        self.para_dict0 = para_dict_tuple[0]
        self.para_dict1 = para_dict_tuple[1]
        self.prepare_gen_savepaths()
        self.MustGenNew = MustGenNew
        self.NameforSave_str = NameforSave_str

        if Trjs[0].shape.__len__() == 3:
            self.num_dims = Trjs[0].shape[2]
            self.log_likelihood_prepare_moredim()
            self.get_one_LRT = self.get_one_LRT_moredim
            self.get_one_LRT_low_frequency = None
            # if self.para_dict0['case_str'][:10].upper() == "OU_process".upper():
                # self.prepare_OU_low_frequency_moredim()
        else:
            self.num_dims = 1
            self.log_likelihood_prepare_1dim()
            self.get_one_LRT = self.get_one_LRT_1dim
            self.get_one_LRT_low_frequency = self.get_one_LRT_low_frequency_1dim

    def get_new_TimeGrids(self):
        return self.TimeGrids

    def prepare_gen_savepaths(self):
        Main_Data_path, Sub_Data_path, Sub_Results_path = add_paths(self.para_dict0['SubPathName'])
        if '_p_' in self.data_prefix[0]:
            NewFileName0 = self.data_prefix[0][self.data_prefix[0].index(self.para_dict0['SubPathName']) + self.para_dict0['SubPathName'].__len__() + 1:]
            NewFileName1 = self.data_prefix[1][self.data_prefix[1].index('_p_'): self.data_prefix[1].index('_nl_')]
        elif 'cd_' in self.data_prefix[0]:
            NewFileName0 = self.data_prefix[0][self.data_prefix[0].index(self.para_dict0['SubPathName']) + self.para_dict0['SubPathName'].__len__() + 1:]
            NewFileName1 = self.data_prefix[1][self.data_prefix[1].index('cd_'): self.data_prefix[1].index('_nl_')]
        elif 'dn_' in self.data_prefix[0]:
            NewFileName0 = self.data_prefix[0][self.data_prefix[0].index(self.para_dict0['SubPathName']) + self.para_dict0['SubPathName'].__len__() + 1:]
            NewFileName1 = self.data_prefix[1][self.data_prefix[1].index('dn_'): self.data_prefix[1].index('_nl_')]
        elif 'nl_' in self.data_prefix[0]:
            NewFileName0 = self.data_prefix[0][self.data_prefix[0].index(self.para_dict0['SubPathName']) + self.para_dict0['SubPathName'].__len__() + 1:]
            NewFileName1 = self.data_prefix[1][self.data_prefix[1].index('nl_'): self.data_prefix[1].index('_fbm_')]
        self.data_prefix_new = Sub_Data_path + NewFileName0[:-1]  + "__VS__" + NewFileName1 + '/'
        if not os.path.exists(self.data_prefix_new):
            os.makedirs(self.data_prefix_new)

    def prepare_OU_low_frequency_1dim(self):
        self.Y_next_est0 = lambda y, dt: y * np.exp( self.para_dict0["p"] * dt )
        self.Y_next_est1 = lambda y, dt: y * np.exp( self.para_dict1["p"] * dt )
        self.Y_var_est0  = lambda y, t, dt: self.para_dict0["diffusion_function"](y, t, self.para_dict0["p"]) ** 2.0 \
                                               / (2 * self.para_dict0["p"]) \
                                               * (1 - np.exp( 2 * self.para_dict0["p"] * dt ))
        self.Y_var_est1  = lambda y, t, dt: self.para_dict1["diffusion_function"](y, t, self.para_dict1["p"]) ** 2.0 \
                                               / (2 * self.para_dict1["p"]) \
                                               * (1 - np.exp( 2 * self.para_dict1["p"] * dt ))

    def log_likelihood_prepare_1dim(self):
        self.theta0 = self.para_dict0["drift_function"]
        self.theta1 = self.para_dict1["drift_function"]
        self.Sigma0 = lambda y, t, p: self.para_dict0["diffusion_function"](y, t, p) * self.para_dict0["diffusion_function"](y, t, p)
        self.Sigma1 = lambda y, t, p: self.para_dict1["diffusion_function"](y, t, p) * self.para_dict1["diffusion_function"](y, t, p)
        self.Sigma0_inv = lambda y, t, p: np.nan_to_num(np.reciprocal(self.Sigma0(y, t, p)), nan=0.0, posinf=1e10, neginf=-1e10)
        self.Sigma1_inv = lambda y, t, p: np.nan_to_num(np.reciprocal(self.Sigma1(y, t, p)), nan=0.0, posinf=1e10, neginf=-1e10)
        self.p0 = self.para_dict0["p"]
        self.p1 = self.para_dict1["p"]

    def log_likelihood_prepare_moredim(self):
        self.theta0 = self.para_dict0["drift_function"]
        self.theta1 = self.para_dict1["drift_function"]
        self.Sigma0 = lambda y, t, p: np.matmul(self.para_dict0["diffusion_function"](y, t, p), self.para_dict0["diffusion_function"](y, t, p).transpose())
        self.Sigma1 = lambda y, t, p: np.matmul(self.para_dict1["diffusion_function"](y, t, p), self.para_dict1["diffusion_function"](y, t, p).transpose())
        self.Sigma0_inv = lambda y, t, p: np.linalg.inv(self.Sigma0(y, t, p))
        self.Sigma1_inv = lambda y, t, p: np.linalg.inv(self.Sigma1(y, t, p))
        self.p0     = self.para_dict0["p"]
        self.p1     = self.para_dict1["p"]

    def get_all_LRT_low_frequency(self):
        if not hasattr(self, "LRT_nparray_low_frequency"):
            self.LRT_nparray_low_frequency = self.get_one_LRT_low_frequency(0)
            for tuple_ind in range(1, self.tuple_len):
                self.LRT_nparray_low_frequency = np.concatenate((self.LRT_nparray_low_frequency, self.get_one_LRT_low_frequency(tuple_ind)))
            return self.LRT_nparray_low_frequency

    def get_all_LRT(self):
        if not hasattr(self, "LRT_nparray"):
            self.LRT_nparray = self.get_one_LRT(0)
            for tuple_ind in range(1, self.tuple_len):
                self.LRT_nparray = np.concatenate((self.LRT_nparray, self.get_one_LRT(tuple_ind)))
        return self.LRT_nparray

    def get_all_drift_time_augment(self):
        if not hasattr(self, "drift_time_augment_np"):
            self.drift_time_augment_np = self.get_one_drift_time_augment(0)
            for tuple_ind in range(1, self.tuple_len):
                self.drift_time_augment_np = np.concatenate(
                    (self.drift_time_augment_np, self.get_one_drift_time_augment(tuple_ind)), 0)
        return self.drift_time_augment_np

    def get_all_drift_time(self):
        if not hasattr(self, "drift_time_np"):
            self.drift_time_np = self.get_one_drift_time(0)
            for tuple_ind in range(1, self.tuple_len):
                self.drift_time_np = np.concatenate(
                    (self.drift_time_np, self.get_one_drift_time(tuple_ind)), 0)
        return self.drift_time_np

    def get_all_time_augment(self):
        if not hasattr(self, "time_augment_np"):
            self.time_augment_np = self.get_one_time_augment(0)
            for tuple_ind in range(1, self.tuple_len):
                self.time_augment_np = np.concatenate(
                    (self.time_augment_np, self.get_one_time_augment(tuple_ind)), 0)
        return self.time_augment_np

    def get_one_LRT_low_frequency_1dim(self, tuple_ind):
        DataFile    = self.data_prefix_new + self.NameforSave_str + '_low_frequency' + str(tuple_ind) + "__LRT.npy"
        if (os.path.exists(DataFile) and self.MustGenNew == 0):
            LRT_res = np.load(DataFile)
            print('Pre-Saved LRT Loaded')
        else:
            self.prepare_OU_low_frequency_1dim()
            Trjs        = self.Trjs[tuple_ind]
            TimeGrids   = self.TimeGrids[tuple_ind]
            num_samples = Trjs.shape[0]
            num_steps   = Trjs.shape[1]
            LRT_res     = np.zeros(num_samples)
            for i in tqdm(range(0, num_samples, 1), total=num_samples):
                for t in range(num_steps-1):
                    mid_dt         = TimeGrids[i, t + 1] - TimeGrids[i, t]
                    mid_dy0        = Trjs[i, t+1]   - self.Y_next_est0(Trjs[i, t], mid_dt)
                    mid_dy1        = Trjs[i, t + 1] - self.Y_next_est1(Trjs[i, t], mid_dt)
                    mid_var0       = self.Y_var_est0(Trjs[i, t], TimeGrids[i, t], mid_dt)
                    mid_var1       = self.Y_var_est1(Trjs[i, t], TimeGrids[i, t], mid_dt)

                    # LRT_res[i]    -= -num_steps/2.0 * np.log(mid_var0)
                    LRT_res[i]    -= -1.0/(2.0 * mid_var0) * mid_dy0 ** 2
                    # LRT_res[i]    += -num_steps/2.0 * np.log(mid_var1)
                    LRT_res[i]    += -1.0/(2.0 * mid_var1) * mid_dy1 ** 2
            np.save(DataFile, LRT_res)
        return LRT_res

    def get_one_LRT_1dim(self, tuple_ind):
        DataFile    = self.data_prefix_new + self.NameforSave_str + str(tuple_ind) + "__LRT.npy"
        if (os.path.exists(DataFile) and self.MustGenNew == 0):
            LRT_res = np.load(DataFile)
            print('Pre-Saved LRT Loaded')
        else:
            Trjs        = self.Trjs[tuple_ind]
            TimeGrids   = self.TimeGrids[tuple_ind]
            num_samples = Trjs.shape[0]
            num_steps   = Trjs.shape[1]
            LRT_res     = np.zeros(num_samples)
            for i in tqdm(range(0, num_samples, 1), total=num_samples):
                for t in range(num_steps-1):
                    mid_b0         = self.theta0(Trjs[i, t], TimeGrids[i, t], self.p0)
                    mid_b1         = self.theta1(Trjs[i, t], TimeGrids[i, t], self.p1)
                    mid_Sigma_inv0 = self.Sigma0_inv(Trjs[i, t], TimeGrids[i, t], self.p0)
                    mid_Sigma_inv1 = self.Sigma1_inv(Trjs[i, t], TimeGrids[i, t], self.p1)
                    mid_dy         = Trjs[i, t+1] - Trjs[i, t]
                    mid_dt         = TimeGrids[i, t+1] - TimeGrids[i, t]
                    LRT_res[i]    += mid_b0 * mid_Sigma_inv0 * mid_dy
                    LRT_res[i]    += -0.5 * mid_b0 * mid_Sigma_inv0 * mid_b0 * mid_dt
                    LRT_res[i]    -= mid_b1 * mid_Sigma_inv1 * mid_dy
                    LRT_res[i]    -= -0.5 * mid_b1 * mid_Sigma_inv1 * mid_b1 * mid_dt
            np.save(DataFile, LRT_res)
        return LRT_res

    def get_one_LRT_moredim(self, tuple_ind):
        DataFile    = self.data_prefix_new + self.NameforSave_str + str(tuple_ind) + "__LRT.npy"
        if (os.path.exists(DataFile) and self.MustGenNew == 0):
            LRT_res = np.load(DataFile)
            print('Pre-Saved LRT Loaded')
        else:
            Trjs        = self.Trjs[tuple_ind]
            TimeGrids   = self.TimeGrids[tuple_ind]
            num_samples = Trjs.shape[0]
            num_steps   = Trjs.shape[1]
            LRT_res     = np.zeros(num_samples)
            for i in tqdm(range(0, num_samples, 1), total=num_samples):
                for t in range(num_steps-1):
                    mid_b0         = self.theta0(Trjs[i, t, :], TimeGrids[i, t], self.p0)
                    mid_b1         = self.theta1(Trjs[i, t, :], TimeGrids[i, t], self.p1)
                    mid_Sigma_inv0 = self.Sigma0_inv(Trjs[i, t, :], TimeGrids[i, t], self.p0)
                    mid_Sigma_inv1 = self.Sigma1_inv(Trjs[i, t, :], TimeGrids[i, t], self.p1)
                    mid_dy         = Trjs[i, t+1, :] - Trjs[i, t, :]
                    mid_dt         = TimeGrids[i, t+1] - TimeGrids[i, t]
                    LRT_res[i]    += np.matmul( np.matmul(mid_b0.transpose(), mid_Sigma_inv0), mid_dy)
                    LRT_res[i]    += -0.5 * np.matmul( np.matmul(mid_b0.transpose(), mid_Sigma_inv0), mid_b0) * mid_dt
                    LRT_res[i]    -= np.matmul( np.matmul(mid_b1.transpose(), mid_Sigma_inv1), mid_dy)
                    LRT_res[i]    -= -0.5 * np.matmul( np.matmul(mid_b1.transpose(), mid_Sigma_inv1), mid_b1) * mid_dt
            np.save(DataFile, LRT_res)
        return LRT_res

    def get_label(self):
        if hasattr(self, "label_nparray"):
            return self.label_nparray
        else:
            self.label_nparray = self.label[0]
            for tuple_ind in range(1, self.tuple_len):
                self.label_nparray = np.concatenate((self.label_nparray, self.label[tuple_ind]))
            return self.label_nparray

    def get_best_ACC(self):
        label_true = self.get_label()
        LRT = self.get_all_LRT()
        self.fpr, self.tpr, self.thresholds = roc_curve(label_true, LRT, drop_intermediate=True)
        if self.tpr[int(self.thresholds.__len__() / 2)] < 0.5:
            self.fpr, self.tpr, self.thresholds = roc_curve(label_true, -LRT, drop_intermediate=True)
        self.accuracy_score_nparray = ((1-self.fpr) + self.tpr)/2.0
        # self.accuracy_score_nparray = np.zeros(self.thresholds.__len__())
        # for i in range(self.thresholds.__len__()):
        #     self.accuracy_score_nparray[i] = accuracy_score(label_true, LRT < self.thresholds[i])
        best_ind = np.argmax(self.accuracy_score_nparray)
        return self.accuracy_score_nparray[best_ind], self.thresholds[best_ind], self.fpr[best_ind], self.tpr[best_ind]

    def get_one_drift_time_augment(self, tuple_ind):
        Trjs        = self.Trjs[tuple_ind]
        time_grid   = self.TimeGrids[tuple_ind]
        num_samples = Trjs.shape[0]
        num_steps   = Trjs.shape[1]
        num_dims    = self.num_dims
        num_triple  = 3  # is constant 3
        num_new_steps = num_triple * num_steps * num_dims + num_steps
        if num_dims == 1:
            Trjs    = np.expand_dims(Trjs, -1)
        Trjs_aug = np.zeros((num_samples, num_new_steps))
        for i in range(num_samples):
            for t in range(num_steps):
                Trjs_aug[i, t * num_dims: (t + 1) * num_dims] = Trjs[i, t, :]
                Trjs_aug[i, t * num_dims + num_steps * num_dims: (t + 1) * num_dims + num_steps * num_dims] = self.para_dict0['drift_function'](Trjs[i, t, :], time_grid[i, t], self.para_dict0['p'])
                Trjs_aug[i, t * num_dims + 2 * num_steps * num_dims: (t + 1) * num_dims + 2 * num_steps * num_dims] = self.para_dict1['drift_function'](Trjs[i, t, :], time_grid[i, t], self.para_dict1['p'])
            Trjs_aug[i, num_triple * num_steps * num_dims:] = time_grid[i, :]
        return Trjs_aug

    def get_one_drift_time(self, tuple_ind):
        Trjs        = self.Trjs[tuple_ind]
        time_grid   = self.TimeGrids[tuple_ind]
        num_samples = Trjs.shape[0]
        num_steps   = Trjs.shape[1]
        num_dims    = self.num_dims
        num_triple  = 2  # is constant 3
        num_new_steps = num_triple * num_steps * num_dims + num_steps
        if num_dims == 1:
            Trjs    = np.expand_dims(Trjs, -1)
        Trjs_aug = np.zeros((num_samples, num_new_steps))
        for i in range(num_samples):
            for t in range(num_steps):
                Trjs_aug[i, t * num_dims: (t + 1) * num_dims] = self.para_dict0['drift_function'](Trjs[i, t, :], time_grid[i, t], self.para_dict0['p'])
                Trjs_aug[i, t * num_dims + num_steps * num_dims: (t + 1) * num_dims + num_steps * num_dims] = self.para_dict1['drift_function'](Trjs[i, t, :], time_grid[i, t], self.para_dict1['p'])
            Trjs_aug[i, num_triple * num_steps * num_dims:] = time_grid[i, :]
        return Trjs_aug

    def get_one_time_augment(self, tuple_ind):
        Trjs      = self.Trjs[tuple_ind]
        time_grid = self.TimeGrids[tuple_ind]
        num_samples = Trjs.shape[0]
        num_steps = Trjs.shape[1]
        num_dims  = self.num_dims
        num_new_steps = num_dims * num_steps + num_steps
        if num_dims == 1:
            Trjs = np.expand_dims(Trjs, -1)
        Trjs_aug = np.zeros((num_samples, num_new_steps))
        for i in range(num_samples):
            for t in range(num_steps):
                Trjs_aug[i, t * num_dims: (t + 1) * num_dims] = Trjs[i, t, :]
            Trjs_aug[i, num_steps * num_dims:] = time_grid[i, :]
        return Trjs_aug



















