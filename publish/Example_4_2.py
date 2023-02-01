# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.utils import get_para_dict_all
from data.generateData import TrjsGen

case_str = "CASE1"

maturity        = 1 * 2 ** int(0)
dt              = 0.1
num_steps_m1    = int(maturity/dt)
Irre_time_level = int(1000 * dt)
para_dict1, para_dict2 = get_para_dict_all(case_str=case_str, num_steps_m1=num_steps_m1, maturity=maturity,
                                           dt=dt, Irre_time_level=Irre_time_level)
DS_Distri1 = 0
trjs_class = TrjsGen(case_str, DS_Distri=DS_Distri1, para_dict=para_dict1)
output1, time_grid1, data_prefix1, para_dict1 = trjs_class.get_final_trjs()
output_ori1, time_grid_ori1 = trjs_class.get_ori_trjs()

DS_Distri2 = DS_Distri1
trjs_class = TrjsGen(case_str, DS_Distri=DS_Distri2, para_dict=para_dict2)
output2, time_grid2, data_prefix2, para_dict2 = trjs_class.get_final_trjs()
output_ori2, time_grid_ori2 = trjs_class.get_ori_trjs()

V1_func  = lambda y: para_dict1['drift_function'](y, 0, 0)
V2_func  = lambda y: para_dict2['drift_function'](y, 0, 0)
x_toplot = np.linspace(-1, 1, 1000)


# make plot
fig = plt.figure(figsize = (14,4))

gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])



# top plot
ax1.plot(x_toplot, V1_func(x_toplot), label=r"$-\pi X_t^{(1)} + \sin(\pi t)$",lw = 2)
ax1.plot(x_toplot, V2_func(x_toplot), "--",label=r"$-0.1 X_t^{(2)} + \cos(\pi X_t^{(2)})$",lw = 2)

ax1.set_xlabel("x, with fix t=0", fontsize = 14)  # default: labelpad=4.0
fig.supylabel("Drift",fontsize = 16)
ax1.set_title("Case: linear vs. nonlinear",fontsize = 16)

V1_func  = lambda y: para_dict1['drift_function'](-.2, y, 0)
V2_func  = lambda y: para_dict2['drift_function'](-.2, y, 0)

# bottom plot
ax2.plot(x_toplot, V1_func(x_toplot), lw = 2)
ax2.plot(x_toplot, V2_func(x_toplot)* np.ones_like(x_toplot), "--",lw = 2)
ax2.set_xlabel("t, with fix x=-.2", fontsize = 14)  # default: labelpad=4.0


# rigth plot
for i in range(25,36):
    ax3.plot(time_grid1, output1[i, :], c="C0",lw = 2,alpha = 0.8)
    ax3.plot(time_grid2, output2[i, :],"--", c="C1",lw = 2)
ax3.set_xlabel("Time: t", fontsize = 14)
ax3.set_title("Sample paths",fontsize = 18)
ax3.set_ylim([-2,2])
ax3.tick_params(axis = "both",labelsize = 12)
ax2.tick_params(axis = "both",labelsize = 12)
ax1.tick_params(axis = "both",labelsize = 12)

ax1.set_title("a)",loc = "left",fontsize = 14)
ax2.set_title("b)",loc = "left",fontsize = 14)
ax3.set_title("c)",loc = "left",fontsize = 14)

fig.legend(fontsize = 12,ncol = 2,frameon = False,loc = (0.36,0.01))
fig.tight_layout()
root = "/Users/esthersida/Documents/Code/UC_ts_sig/marcc/sigUCts/results"
fig.savefig(root+"/Example_4_2.pdf")











# %%
