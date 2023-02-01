import numpy as np
from tqdm import tqdm
from fbm import fbm    # fractional brownian motion package
import scipy.io
import os

from utils.add_paths import add_paths


class TrjsGen:
    """
    Generate time series data by SDE
    """
    def __init__(self, case_str, num_samples, *, DS_Distri=0, para_dict=None):
        self.case_str = case_str
        self.DS_Distri = DS_Distri  # Distribution of downsamples. 0: regular; 1: Unif on the whole interval; 2: Piecewise Unif
        if self.case_str[:9].upper() == "POTENTIAL":
            if para_dict is None:
                self.SubPathName = 'sde_data/'
                self.OriFileName = "SDE_potentials_nTraj500.mat"
                self.NewFileName = "LangLocalGradSys/beta1Mu2_tN2wSSBE/"  # potential_type1
                # self.NewFileName = "x4/beta1Mu1_tN2wSSBE/"  # potential_type2
                # self.NewFileName = "DoubleWell/beta1_tN2wSSBE_d1/"  # potential_type3
                # self.NewFileName = "DoubleWell/beta1_tN2wSSBE_d2/"  # potential_type3
                self.num_samples = num_samples  # Number of samples
                self.num_steps = 1001  # Number of time steps
                self.maturity = 20  # The maturity of SDE, that is, the end-time of the time interval
            else:
                self.SubPathName = para_dict['SubPathName']
                self.OriFileName = para_dict['OriFileName']
                self.NewFileName = para_dict['NewFileName']
                self.num_samples = para_dict['num_samples']
                self.num_steps = para_dict['num_steps']
                self.maturity = para_dict['maturity']
            self.Main_Data_path, self.Sub_Data_path, self.Sub_Results_path = add_paths(self.SubPathName)
            self.data_prefix = self.Sub_Data_path + self.NewFileName + '/T_%.2f_nsp_%d_nst_%d_DS_%d/' \
                               % (self.maturity, self.num_samples, self.num_steps, self.DS_Distri)

        elif self.case_str[:5].upper() == "CASE1":
            if para_dict is None:
                self.SubPathName = 'data_ts/'
                self.num_samples = num_samples   # Number of samples
                self.num_steps = 1001  # Number of time steps
                self.maturity = 1  # The maturity of SDE, that is, the end-time of the time interval
                self.Irre_time_level = 10  # Int: if 1, Equidis time grid; Larger may cause more irregularity
                self.ZeroIni = 0  # Initial condition, if 1: Zero initial; not 1: standard Normal
                self.hurst = 0.8  # The Hurst parameter
                self.nonlinear = 0  # linear or nonlinear equation
            else:
                self.SubPathName = para_dict['SubPathName']
                self.num_samples = para_dict['num_samples']
                self.num_steps = para_dict['num_steps']
                self.maturity = para_dict['maturity']
                self.Irre_time_level = para_dict['Irre_time_level']
                self.ZeroIni = para_dict['ZeroIni']
                self.hurst = para_dict['hurst']
                self.nonlinear = para_dict['nonlinear']
            if self.nonlinear == 0:
                self.drift_function = lambda y, t, p: (-np.pi * y + np.sin(np.pi * t))
            else:
                self.drift_function = lambda y, t, p: (-0.1 * y + np.cos(np.pi * y))
            self.diffusion_function = lambda y, t, p: y
            self.diffusion_function_dx = lambda y, t, p: 1.0
            self.p = 0.0
            self.para_dict = para_dict
            self.para_dict.update({
                "drift_function": self.drift_function,
                "diffusion_function": self.diffusion_function,
                "diffusion_function_dx": self.diffusion_function_dx,
                "p": self.p
            })
            self.Main_Data_path, self.Sub_Data_path, self.Sub_Results_path = add_paths(self.SubPathName)
            self.num_steps_ori = (self.num_steps - 1) * self.Irre_time_level + 1
            self.data_prefix = self.Sub_Data_path + '/nl_%d_fbm_%.2f_zi_%d_T_%.2f_ir_%d_nsp_%d_nst_%d_DS_%d/' \
                               % (self.nonlinear, self.hurst, self.ZeroIni, self.maturity, self.Irre_time_level,
                                  self.num_samples, self.num_steps, self.DS_Distri)  # The directory to save the data

        elif self.case_str[:14].upper() == "constant_drift".upper():
            if para_dict is None:
                self.SubPathName = 'constant_drift_BM/'
                self.num_samples = num_samples  # Number of samples
                self.num_steps = 1001  # Number of time steps
                self.maturity = 1  # The maturity of SDE, that is, the end-time of the time interval
                self.Irre_time_level = 1  # Int: if 1, Equidis time grid; Larger may cause more irregularity
                self.ZeroIni = 0  # Initial condition, if 1: Zero initial; not 1: standard Normal
                self.hurst = 0.5  # The Hurst parameter
                self.nonlinear = 0  # linear or nonlinear equation
                self.constant_drift = 1.0
            else:
                self.SubPathName = para_dict['SubPathName']
                self.num_samples = para_dict['num_samples']
                self.num_steps = para_dict['num_steps']
                self.maturity = para_dict['maturity']
                self.Irre_time_level = para_dict['Irre_time_level']
                self.ZeroIni = para_dict['ZeroIni']
                self.hurst = para_dict['hurst']
                self.nonlinear = para_dict['nonlinear']
                self.constant_drift = para_dict['constant_drift']
            self.drift_function = lambda y, t, p: p
            self.diffusion_function = lambda y, t, p: 1.0
            self.diffusion_function_dx = lambda y, t, p: 0.0
            self.p = self.constant_drift
            self.para_dict = para_dict
            self.para_dict.update({
                "drift_function": self.drift_function,
                "diffusion_function": self.diffusion_function,
                "diffusion_function_dx": self.diffusion_function_dx,
                "p": self.p
            })
            self.Main_Data_path, self.Sub_Data_path, self.Sub_Results_path = add_paths(self.SubPathName)
            self.num_steps_ori = (self.num_steps - 1) * self.Irre_time_level + 1
            self.data_prefix = self.Sub_Data_path + '/cd_%.2f_nl_%d_fbm_%.2f_zi_%d_T_%.2f_ir_%d_nsp_%d_nst_%d_DS_%d/' \
                               % (self.constant_drift, self.nonlinear, self.hurst, self.ZeroIni, self.maturity, self.Irre_time_level,
                                  self.num_samples, self.num_steps, self.DS_Distri)

        elif self.case_str[:10].upper() == "OU_process".upper():
            if para_dict is None:
                self.SubPathName = 'OU_process/'
                self.num_samples = num_samples   # Number of samples
                self.num_steps = 101  # Number of time steps
                self.maturity = 1  # The maturity of SDE, that is, the end-time of the time interval
                self.Irre_time_level = 10  # Int: if 1, Equidis time grid; Larger may cause more irregularity
                self.ZeroIni = 0  # Initial condition, if 1: Zero initial; not 1: standard Normal
                self.hurst = 0.5  # The Hurst parameter
                self.nonlinear = 0  # linear or nonlinear equation
                self.constant_drift = -1.0
            else:
                self.SubPathName = para_dict['SubPathName']
                self.num_samples = para_dict['num_samples']
                self.num_steps = para_dict['num_steps']
                self.maturity = para_dict['maturity']
                self.Irre_time_level = para_dict['Irre_time_level']
                self.ZeroIni = para_dict['ZeroIni']
                self.hurst = para_dict['hurst']
                self.nonlinear = para_dict['nonlinear']
                self.constant_drift = para_dict['constant_drift']
            self.drift_function = lambda y, t, p: p * y
            self.diffusion_function = lambda y, t, p: 1.0
            self.diffusion_function_dx = lambda y, t, p: 0.0
            self.p = self.constant_drift
            self.para_dict = para_dict
            self.para_dict.update({
                "drift_function": self.drift_function,
                "diffusion_function": self.diffusion_function,
                "diffusion_function_dx": self.diffusion_function_dx,
                "p": self.p
            })
            self.Main_Data_path, self.Sub_Data_path, self.Sub_Results_path = add_paths(self.SubPathName)
            self.num_steps_ori = (self.num_steps - 1) * self.Irre_time_level + 1
            self.data_prefix = self.Sub_Data_path + '/cd_%.2f_nl_%d_fbm_%.2f_zi_%d_T_%.2f_ir_%d_nsp_%d_nst_%d_DS_%d/' \
                               % (self.constant_drift, self.nonlinear, self.hurst, self.ZeroIni, self.maturity, self.Irre_time_level,
                                  self.num_samples, self.num_steps, self.DS_Distri)

        elif self.case_str[:20].upper() == "Different_potentials".upper():
            if para_dict is None:
                self.SubPathName = 'Different_potentials/'
                self.num_samples = num_samples   # Number of samples
                self.num_steps = 1001  # Number of time steps
                self.maturity = 1  # The maturity of SDE, that is, the end-time of the time interval
                self.Irre_time_level = 1  # Int: if 1, Equidis time grid; Larger may cause more irregularity
                self.ZeroIni = 0  # Initial condition, if 1: Zero initial; not 1: standard Normal
                self.hurst = 0.5  # The Hurst parameter
                self.nonlinear = 0  # linear or nonlinear equation
                self.drift_function = lambda y, t, p: - y * (y - 1) * (y + 1)
                self.drift_name = "y(y-1)(y+1)"
            else:
                self.SubPathName = para_dict['SubPathName']
                self.num_samples = para_dict['num_samples']
                self.num_steps = para_dict['num_steps']
                self.maturity = para_dict['maturity']
                self.Irre_time_level = para_dict['Irre_time_level']
                self.ZeroIni = para_dict['ZeroIni']
                self.hurst = para_dict['hurst']
                self.nonlinear = para_dict['nonlinear']
                self.drift_function = para_dict['drift_function']
                self.drift_name = para_dict['drift_name']
                self.diffusion_function = para_dict['diffusion_function']
                self.diffusion_function_dx = para_dict['diffusion_function_dx']
            self.p = 0.0
            self.para_dict = para_dict
            self.para_dict.update({
                "diffusion_function": self.diffusion_function,
                "diffusion_function_dx": self.diffusion_function_dx,
                "p": self.p
            })
            self.Main_Data_path, self.Sub_Data_path, self.Sub_Results_path = add_paths(self.SubPathName)
            self.num_steps_ori = (self.num_steps - 1) * self.Irre_time_level + 1
            self.data_prefix = self.Sub_Data_path + '/dn_%s_nl_%d_fbm_%.2f_zi_%d_T_%.2f_ir_%d_nsp_%d_nst_%d_DS_%d/' \
                               % (self.drift_name, self.nonlinear, self.hurst, self.ZeroIni, self.maturity, self.Irre_time_level,
                                  self.num_samples, self.num_steps, self.DS_Distri)

        elif self.case_str[:12].upper() == "moredim_case".upper():
            if para_dict is None:
                self.SubPathName = 'moredim_case/'
                self.num_samples = num_samples  # Number of samples
                self.num_steps = 1001  # Number of time steps
                self.maturity = 1  # The maturity of SDE, that is, the end-time of the time interval
                self.Irre_time_level = 10  # Int: if 1, Equidis time grid; Larger may cause more irregularity
                self.ZeroIni = 0  # Initial condition, if 1: Zero initial; not 1: standard Normal
                self.hurst = 0.5  # The Hurst parameter
                self.nonlinear = 0  # linear or nonlinear equation
                self.num_dims  = 2  # Number of dims of y
                self.num_dims_noise = 3   # Number of dims of noise
                self.drift_function        = lambda y, t, p: p[0] * np.ones(self.num_dims)  # n-dim
                self.diffusion_function    = lambda y, t, p: p[1] * np.array([[1, 1, 1], [0, 1, 0]])  # n*m-dim
                self.diffusion_function_dx = lambda y, t, p: 0.0
                self.p = np.array([1.0, 1.0])
                self.additional_name = "constant_drift_constant_diffusion"
            else:
                self.SubPathName = para_dict['SubPathName']
                self.num_samples = para_dict['num_samples']
                self.num_steps = para_dict['num_steps']
                self.maturity = para_dict['maturity']
                self.Irre_time_level = para_dict['Irre_time_level']
                self.ZeroIni = para_dict['ZeroIni']
                self.hurst = para_dict['hurst']
                self.nonlinear = para_dict['nonlinear']
                self.num_dims  = para_dict['num_dims']
                self.num_dims_noise = para_dict['num_dims_noise']
                self.drift_function        = para_dict['drift_function']
                self.diffusion_function    = para_dict['diffusion_function']
                self.diffusion_function_dx = para_dict['diffusion_function_dx']
                self.additional_name = para_dict['additional_name']
                self.p = para_dict['p']
            self.para_dict = para_dict
            self.Main_Data_path, self.Sub_Data_path, self.Sub_Results_path = add_paths(self.SubPathName)
            self.num_steps_ori = (self.num_steps - 1) * self.Irre_time_level + 1
            self.para_dict.update({"num_steps_ori": self.num_steps_ori})
            self.data_prefix = self.Sub_Data_path + '/%s_n_%d_m_%d_p_%s_nl_%d_fbm_%.2f_zi_%d_T_%.2f_ir_%d_nsp_%d_nst_%d_DS_%d/' \
                               % (self.additional_name, self.num_dims, self.num_dims_noise, np.array2string(self.p),
                                  self.nonlinear, self.hurst, self.ZeroIni, self.maturity, self.Irre_time_level,
                                  self.num_samples, self.num_steps, self.DS_Distri)

        if not os.path.exists(self.data_prefix):
            os.makedirs(self.data_prefix)

        self.DataFile = self.data_prefix + 'output.npy'
        self.DatatimeFile = self.data_prefix + 'time.npy'
        self.OriDataFile = self.data_prefix + 'output_ori.npy'
        self.OriDatatimeFile = self.data_prefix + 'time_ori.npy'

    def ori_trjs_gen(self):
        if os.path.exists(self.OriDataFile) and os.path.exists(self.OriDatatimeFile):
            self.output_ori = np.load(self.OriDataFile)
            self.time_grid_ori = np.load(self.OriDatatimeFile)
            print('Pre-Saved original trajectories Loaded')
        else:
            if self.case_str[:9].upper() == "POTENTIAL":
                self.sde_data = scipy.io.loadmat(self.Sub_Data_path + self.OriFileName)
                if self.NewFileName[:16] == "LangLocalGradSys":
                    self.output_ori = self.sde_data['all_traj1'][0, :, :].T
                elif self.NewFileName[:2] == "x4":
                    self.output_ori = self.sde_data['all_traj2'][0, :, :].T
                elif self.NewFileName[:10] == "DoubleWell" and self.NewFileName[-3:-1] == "d1":
                    self.output_ori = self.sde_data['all_traj3'][0, :, :].T
                elif self.NewFileName[:10] == "DoubleWell" and self.NewFileName[-3:-1] == "d2":
                    self.output_ori = self.sde_data['all_traj3'][1, :, :].T
                self.time_grid_ori = self.maturity
            elif self.case_str[:5].upper() == "CASE1":
                BM_paths, self.output_ori = get_sde_trajs(self.hurst, nonlinear=self.nonlinear,
                                                          num_samples=self.num_samples,
                                                          num_steps=self.num_steps_ori, maturity=self.maturity,
                                                          ZeroIni=self.ZeroIni, para_dict=self.para_dict)
                self.time_grid_ori = self.maturity
            elif self.case_str[:14].upper() == "constant_drift".upper():
                BM_paths, self.output_ori = get_sde_trajs(self.hurst, nonlinear=self.nonlinear,
                                                          num_samples=self.num_samples,
                                                          num_steps=self.num_steps_ori, maturity=self.maturity,
                                                          ZeroIni=self.ZeroIni, para_dict=self.para_dict)
                self.time_grid_ori = self.maturity
            elif self.case_str[:10].upper() == "OU_process".upper():
                BM_paths, self.output_ori = get_sde_trajs(self.hurst, nonlinear=self.nonlinear,
                                                          num_samples=self.num_samples,
                                                          num_steps=self.num_steps_ori, maturity=self.maturity,
                                                          ZeroIni=self.ZeroIni, para_dict=self.para_dict)
                self.time_grid_ori = self.maturity
            elif self.case_str[:20].upper() == "Different_potentials".upper():
                BM_paths, self.output_ori = get_sde_trajs(self.hurst, nonlinear=self.nonlinear,
                                                          num_samples=self.num_samples,
                                                          num_steps=self.num_steps_ori, maturity=self.maturity,
                                                          ZeroIni=self.ZeroIni, para_dict=self.para_dict)
                self.time_grid_ori = self.maturity
            elif self.case_str[:12].upper() == "moredim_case".upper():
                BM_paths, self.output_ori = get_sde_trajs_moredim(para_dict=self.para_dict)
                self.time_grid_ori = self.maturity
            np.save(self.OriDataFile, self.output_ori)
            np.save(self.OriDatatimeFile, self.time_grid_ori)

    def get_final_trjs(self):
        if os.path.exists(self.DataFile) and os.path.exists(self.DatatimeFile):
            self.output = np.load(self.DataFile)
            self.time_grid = np.load(self.DatatimeFile)
            print('Pre-Saved final trajectories Loaded')
        else:
            self.ori_trjs_gen()
            self.output, self.time_grid = DownSample(self.output_ori, maturity=self.maturity,
                                                     num_samples=self.num_samples,
                                                     num_steps=self.num_steps, DS_Distri=self.DS_Distri)
            np.save(self.DataFile, self.output)
            np.save(self.DatatimeFile, self.time_grid)
        return self.output, self.time_grid, self.data_prefix, self.para_dict

    def get_ori_trjs(self):
        if os.path.exists(self.OriDataFile) and os.path.exists(self.OriDatatimeFile):
            self.output_ori = np.load(self.OriDataFile)
            self.time_grid_ori = np.load(self.OriDatatimeFile)
            print('Pre-Saved original trajectories Loaded')
        else:
            self.ori_trjs_gen()
        return self.output_ori, self.time_grid_ori


# Milstein's method  return Y_curr ; # to change the code to have SDE drift and diffusion as input
# def ComputeY_Case_1(Y_last, dt, dB, step, *, nonlinear):
#     if nonlinear:
#         Y_curr = Y_last + (-0.1 * np.abs(Y_last) + np.cos(Y_last)) * \
#                  dt + Y_last * dB + 0.5 * Y_last * (dB * dB - dt)
#     else:
#         Y_curr = Y_last + (-np.pi * Y_last + np.sin(np.pi * step * dt)) * \
#                  dt + Y_last * dB + 0.5 * Y_last * (dB * dB - dt)
#     return Y_curr


def ComputeY_Milstein(Y_last, dt, dB, step, *, para_dict):
    Y_curr = Y_last + para_dict["drift_function"](Y_last, step * dt, para_dict["p"]) * dt + \
             para_dict["diffusion_function"](Y_last, step * dt, para_dict["p"]) * dB + \
             0.5 * para_dict["diffusion_function"](Y_last, step * dt, para_dict["p"]) * \
             para_dict["diffusion_function_dx"](Y_last, step * dt, para_dict["p"]) * (dB * dB - dt)
    return Y_curr


def ComputeY_simple_moredim(Y_last, dt, dB, step, *, para_dict):
    Y_curr = Y_last + para_dict["drift_function"](Y_last, step * dt, para_dict["p"]) * dt + \
             np.matmul(para_dict["diffusion_function"](Y_last, step * dt, para_dict["p"]), dB)
    return Y_curr


# Simulate SDE with fractional Brownian motion driving path
def Sim_SDE_FBM(num_steps, *, T, H=0.75, nonlinear, ZeroIni=0, para_dict):
    dT              = T / (num_steps - 1)
    if H == 0.5:
        FBM_paths   = np.cumsum( np.concatenate((np.array([0]), np.random.randn(num_steps-1))) ) * np.sqrt(dT)
    else:
        FBM_paths   = fbm(n=num_steps - 1, hurst=H, length=T, method='daviesharte')
    output          = np.zeros(num_steps, dtype=float)
    if not ZeroIni==1:
        output[0]   = np.random.randn()
    for i in range(1, num_steps):
        output[i] = ComputeY_Milstein(output[i - 1], dT, FBM_paths[i] - FBM_paths[i - 1], i, para_dict=para_dict)
    return FBM_paths, output


def Sim_SDE_FBM_moredim(*, para_dict):
    dT = para_dict['maturity'] / (para_dict['num_steps_ori'] - 1)
    FBM_paths = np.zeros((para_dict['num_steps_ori'], para_dict['num_dims_noise']))
    if para_dict['hurst'] == 0.5:
        for i in range(para_dict['num_dims_noise']):
            mid_FBM_paths   = np.cumsum( np.concatenate((np.array([0]), np.random.randn(para_dict['num_steps_ori'] - 1))) ) * np.sqrt(dT)
            FBM_paths[:, i] = mid_FBM_paths
    else:
        for i in range(para_dict['num_dims_noise']):
            mid_FBM_paths   = fbm(n=para_dict['num_steps_ori'] - 1, hurst=para_dict['hurst'], length=para_dict['maturity'], method='daviesharte')
            FBM_paths[:, i] = mid_FBM_paths
    output = np.zeros((para_dict['num_steps_ori'], para_dict['num_dims']), dtype=float)
    if not para_dict['ZeroIni'] == 1:
        output[0, :]   = np.random.randn(para_dict['num_dims'])
    for i in range(1, para_dict['num_steps_ori']):
        output[i, :] = ComputeY_simple_moredim(output[i - 1, :], dT, FBM_paths[i, :] - FBM_paths[i - 1, :], i, para_dict=para_dict)
    return FBM_paths, output


def get_sde_trajs(hurst, *, nonlinear, num_samples, num_steps, maturity, ZeroIni, para_dict):
    BM_paths = np.zeros([num_samples, num_steps], dtype=float)
    output   = np.zeros([num_samples, num_steps], dtype=float)
    for i in tqdm(range(0, num_samples, 1), total=num_samples):
        BM_paths[i, :], output[i, :] = Sim_SDE_FBM(num_steps, T=maturity, H=hurst, nonlinear=nonlinear, ZeroIni=ZeroIni, para_dict=para_dict)
    return BM_paths, output


def get_sde_trajs_moredim(*, para_dict):
    BM_paths = np.zeros([para_dict['num_samples'], para_dict['num_steps_ori'], para_dict['num_dims_noise']], dtype=float)
    output   = np.zeros([para_dict['num_samples'], para_dict['num_steps_ori'], para_dict['num_dims']], dtype=float)
    for i in tqdm(range(0, para_dict['num_samples'], 1), total=para_dict['num_samples']):
        BM_paths[i, :, :], output[i, :, :] = Sim_SDE_FBM_moredim(para_dict=para_dict)
    return BM_paths, output


def DownSample(output_mid, *, maturity, num_samples, num_steps, DS_Distri=1):
    """
    Input:
        num_steps : Number of steps for new trajectories
        DS_Distri : Distribution of downsamples
            0: Produce unif_grid_step 1: Unif on the whole interval; 2: Piecewise Unif
    """
    if DS_Distri == 0:
        long_no_steps     = output_mid.shape[1]
    else:
        output            = np.zeros((num_samples, num_steps))
        time_grid         = np.zeros((num_samples, num_steps))
        long_no_steps     = output_mid.shape[1]
        output[:, 0]      = output_mid[:, 0]
        time_grid[:, 0]   = 0.0
        output[:, -1]     = output_mid[:, -1]
        time_grid[:, -1]  = maturity

    if DS_Distri == 0:
        mid_ind   = np.linspace(0, int(np.floor((long_no_steps-1)/(num_steps-1)) * (num_steps-1)), num_steps).astype(int)
        time_grid = maturity * mid_ind / (long_no_steps - 1)
        output    = output_mid[:num_samples, mid_ind]
    elif DS_Distri == 1:
        for i in range(num_samples):
            mid_ind            = np.sort( np.random.choice(range(1, long_no_steps - 1), num_steps - 2, replace=False) )
            time_grid[i, 1:-1] = maturity * mid_ind / (long_no_steps - 1)
            output[i, 1:-1]    = output_mid[i, mid_ind]
    elif DS_Distri == 2:
        for i in range(num_samples):
            My_P           = np.ones(long_no_steps - 2)
            NumPiece       = 5  # Number of pieces where downsamples uniformly distributed on
            mid_ind        = np.sort(np.random.choice(range(1, long_no_steps - 3), NumPiece - 1, replace=False))
            mid_ind        = np.concatenate((np.array([0]), mid_ind, np.array([long_no_steps - 3])))
            for j in range(NumPiece):
                My_P[mid_ind[j]:mid_ind[j + 1]] *= np.random.uniform(0.2, 5)
            My_P = My_P / sum(My_P)
            midmid_ind         = np.sort( np.random.choice(range(1, long_no_steps - 1), num_steps * 10, replace=True, p=My_P) )
            mid_ind            = np.sort( np.random.choice(np.unique(midmid_ind), num_steps - 2, replace=False) )
            time_grid[i, 1:-1] = maturity * mid_ind / (long_no_steps - 1)
            output[i, 1:-1]    = output_mid[i, mid_ind]
    return output, time_grid



















