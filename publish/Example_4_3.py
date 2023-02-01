# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.utils import get_para_dict_all
from data.generateData import TrjsGen

case_str = "moredim_case_Opinion"

maturity        = 10 * 2 ** int(0)
dt              = 1.0
num_steps_m1    = int(maturity/dt)
Irre_time_level = int(10)
para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                           dt=dt, Irre_time_level=Irre_time_level, num_samples=10)
DS_Distri1 = 0
trjs_class = TrjsGen(case_str, DS_Distri=DS_Distri1, para_dict=para_dict1)
output1, time_grid1, data_prefix1, para_dict1 = trjs_class.get_final_trjs()
output_ori1, time_grid_ori1 = trjs_class.get_ori_trjs()

DS_Distri2 = DS_Distri1
trjs_class = TrjsGen(case_str, DS_Distri=DS_Distri2, para_dict=para_dict2)
output2, time_grid2, data_prefix2, para_dict2 = trjs_class.get_final_trjs()
output_ori2, time_grid_ori2 = trjs_class.get_ori_trjs()

from utils.utils import OD_influence
V1_func  = lambda y: OD_influence(y, type_ind=4)
V2_func  = lambda y: OD_influence(y, type_ind=1)
x_toplot = np.linspace(-.5, 3, 1000)

y1       = [V1_func(x_toplot[i]) for i in range(x_toplot.__len__())]
y2       = [V2_func(x_toplot[i]) for i in range(x_toplot.__len__())]
# %%
# make plot
fig, axs = plt.subplots(1, 2, figsize = (14,4))
# left plot
axs[0].plot(x_toplot, y1, label=r"Kernel 1",lw = 2)
axs[0].plot(x_toplot, y2, "--",label=r"Kernel 2",lw = 2)
axs[0].set_xlabel("r", fontsize = 14)  # default: labelpad=4.0
axs[0].set_ylabel(r"Kernel: $\phi$",fontsize = 14)
axs[0].set_title("Case: different kernels",fontsize = 16)
# rigth plot
for i in range(10):
    axs[1].plot(time_grid1, output1[i, :, 0], c="C0",lw = 2,alpha = 0.8)
    axs[1].plot(time_grid2, output2[i, :, 0],"--", c="C1",lw = 2)
axs[1].set_xlabel("Time: t", fontsize = 14)
axs[1].set_title("Sample paths in 1st dimension",fontsize = 18)
axs[1].set_ylim([-6,10])
axs[0].tick_params(axis = "both",labelsize = 12)
axs[1].tick_params(axis = "both",labelsize = 12)

axs[0].set_title("a)",loc = "left",fontsize = 14)
axs[1].set_title("b)",loc = "left",fontsize = 14)

fig.legend(fontsize = 14,frameon = False,ncol = 2,loc = (0.4,0.02))
fig.tight_layout()
root = "/Users/esthersida/Documents/Code/UC_ts_sig/marcc/sigUCts/results"
fig.savefig(root+"/Example_4_3.pdf")











# %%
