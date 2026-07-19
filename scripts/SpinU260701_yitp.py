import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import sobol_seq
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp   import autocast, GradScaler
from scipy.stats      import norm
import time
import os
import argparse
current_timestamp = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GPUが利用可能であれば'cuda'、そうでなければ'cpu'を使用
class Config: # Parameterのセット
        e_ic             =    1             # energy density [GeV unit]（初期条件）0.004 [GeV unit]

        log_file_name    = "training_log_SpinU_6.txt"
        obs_model_name   = "model_spininit.pth"
        file_gamma       =    2
        file_time        =    4
        gamma            =  2.0 *e_ic**0.75 # spin transport coefficient
        t_max2           = 0.42             # upper limit of the training time range
        N_LAYER          =    5             # ニューラルネットワークの層の数
        ang_con          =    1             # 0: w/o conservation, 1:w/ conservation
        SWITCH_LOSS2     =    1             # 0: w/o ang loss, 1: w/ ang loss
        SWITCH_INIT      =    1             # 0: orbital, 1: spin
        N_obs            = 200000           # if N_obs > 0, save and stop when global_step reaches N_obs


        tau_phi          = gamma/e_ic       # 緩和時間 [GeV unit]
        NEURON_PER_LAYER =  250             # 64*ne_x*nb_y # 各層に含まれるニューロンの数
        LOSS_TYPE        =    3             # 0: Normal, 1: Uncertainty Weighting, 2: Uncertainty Weighting (ver2), 3: Uncertainty Weighting (ver3)
        N_print_max      =   50
        uniform_option   = 0                # 0: all uniform, 1: 4/5 uniform + 1/5 residual-weighted
        log_file_option  = 1                # 0: use timestamped log file, 1: use fixed log file name
        remove_old_log   = True


        Nc           = 3                  # SU(N)ゲージ理論における色の数
        Nadj         = Nc * Nc - 1        # adjoint表現（ゲージ場の自由度）の数
        Nf           = 2                  # 軽いquarkの数
        R            = 1                  # 系の半径を表すパラメータ, (b_{imp}/2=R_A/2) 3fm=16[GeV unit]
        t_bin        = 0.2                # initial time range and base unit for time-domain extension
        t_max        = 0.2                # 時間の最大値です。10fm=50 [GeV unit]
        omega_ic     = 0.0/R              # 角速度 [GeV unit]（初期条件）,  0.007[GeV unit]
        omega_ic_dot = 0.0/R              # 角速度 [GeV unit]（初期条件）,  0.007[GeV unit]
        eta          = 0   * e_ic**0.75   # shear viscosity, 2.0 (KS bound)
        zeta         = 0   * e_ic**0.75   # bulk  viscosity
        tau_sh       = eta  /e_ic       # 緩和時間 [GeV unit]
        tau_bu       = zeta /e_ic       # 緩和時間 [GeV unit]
        factor_s     = 6/19 * np.sqrt(29/15) * np.pi
        Np               =  25000           # initial number of collocation points
        EPOCHS           =  1000           # 1000    # 学習におけるエポック数（全データセットに対して学習する回数）
        batch_size       =  Np//5          #//20    #        # ミニバッチサイズ %5で割り切れなければならない
        LEARNING_RATE    =  1e-3           # ニューラルネットワークの学習率
        LEARNING_RATE2   =  1e-3           # ニューラルネットワークの学習率
        epsilon_om       = 2
        sigma_om         = np.pi/R
        LOSS_WEIGHT      = 1.0       # 損失関数に対する重み
        SWITCH_LOSS      = 0         # 0: w/o boundary loss, 1: w/ boundary loss
        act_type         = 0         # 0:tanh, 1:SeLu, 2:sin
        N_ang_con        = 5.0
        log_sigma_Rd     = 0.0       #
        log_sigma_Rd_b2  = 0.0       #
        count_epoch      = 0       #
        sum_epoch        = 0       #
        N_refresh_margin = 20000

class DataHandler:
        def __init__(self, config):
                self.config = config

        def generate_collocation_points(self):
                samples = np.random.uniform(0, 1, (self.config.Np, 2))  # 3次元のランダムサンプル
                data    = samples.copy()
                data[:, 0]  = data[:, 0] * self.config.t_max
                data[:, 1]  = np.sqrt(data[:, 1]) * self.config.R
                t_c = np.expand_dims(data[:, 0], axis=1)
                x_c = np.expand_dims(data[:, 1], axis=1)

                x_c = np.maximum(x_c, 0.0001 * self.config.R    )  # 原点回避（必要なら）
                return t_c, x_c

        def generate_data(self):
                t_c, x_c  = self.generate_collocation_points()    # コロケーションポイントを生成
                return t_c, x_c

        def generate_collocation_points2(self):
                samples = np.random.uniform(0, 1, (self.config.Np, 2))  # 3次元のランダムサンプル
                data    = samples.copy()
                data[:, 0]  = data[:, 0] * self.config.t_max
                data[:, 1]  = data[:, 1] * self.config.R
                t_c = np.expand_dims(data[:, 0], axis=1)
                x_c = np.expand_dims(data[:, 1], axis=1)

                x_c = np.maximum(x_c, 0.0001 * self.config.R    )  # 原点回避（必要なら）
                return t_c, x_c

        def generate_data2(self):
                t_c, x_c  = self.generate_collocation_points2()    # コロケーションポイントを生成
                return t_c, x_c

        def generate_collocation_points_bc(self):
                if config.Np % 5 != 0:
                        print("config.Np is not divisible by 5")
                t_c2 = np.random.uniform(low=0.0,               high=config.t_max, size=(config.Np//5, 1))
                x_c2 = np.random.uniform(low=0.9999 * config.R, high=config.R,     size=(config.Np//5, 1))

                return t_c2, x_c2

        def generate_data_bc(self):
                t_c2, x_c2  = self.generate_collocation_points_bc()    # コロケーションポイントを生成
                return t_c2, x_c2

###################################################################################################
config = Config()

parser = argparse.ArgumentParser()
parser.add_argument("--log_file_name", type=str, default=config.log_file_name)
parser.add_argument("--obs_model_name", type=str, default=config.obs_model_name)
parser.add_argument("--gamma_factor", type=float, default=2.0)
parser.add_argument("--t_max2", type=float, default=config.t_max2)
parser.add_argument("--N_LAYER", type=int, default=config.N_LAYER)
parser.add_argument("--ang_con", type=int, default=config.ang_con)
parser.add_argument("--SWITCH_LOSS2", type=int, default=config.SWITCH_LOSS2)
parser.add_argument("--SWITCH_INIT", type=int, default=config.SWITCH_INIT)
parser.add_argument("--N_obs", type=int, default=config.N_obs)

args = parser.parse_args()

config.log_file_name  = args.log_file_name
config.obs_model_name = args.obs_model_name
config.gamma_factor = args.gamma_factor
config.gamma        = config.gamma_factor * config.e_ic**0.75
config.tau_phi      = config.gamma / config.e_ic
config.t_max2  = args.t_max2
config.N_LAYER = args.N_LAYER
config.ang_con      = args.ang_con
config.SWITCH_LOSS2 = args.SWITCH_LOSS2
config.SWITCH_INIT  = args.SWITCH_INIT
config.file_gamma = round(config.gamma_factor)
config.file_time  = round(10 * config.t_max2)
config.N_obs = args.N_obs

data_handler = DataHandler(config)


print("===== Effective config parameters =====", flush=True)
print(f"log_file_name    = {config.log_file_name}", flush=True)
print(f"obs_model_name   = {config.obs_model_name}", flush=True)
print(f"gamma_factor     = {config.gamma_factor}", flush=True)
print(f"gamma            = {config.gamma}", flush=True)
print(f"tau_phi          = {config.tau_phi}", flush=True)
print(f"t_max2           = {config.t_max2}", flush=True)
print(f"N_LAYER          = {config.N_LAYER}", flush=True)
print(f"ang_con          = {config.ang_con}", flush=True)
print(f"SWITCH_LOSS2     = {config.SWITCH_LOSS2}", flush=True)
print(f"SWITCH_INIT      = {config.SWITCH_INIT}", flush=True)
print(f"N_obs            = {config.N_obs}", flush=True)
print(f"LOSS_TYPE        = {config.LOSS_TYPE}", flush=True)
print(f"uniform_option   = {config.uniform_option}", flush=True)
print(f"Np               = {config.Np}", flush=True)
print(f"EPOCHS           = {config.EPOCHS}", flush=True)
print(f"batch_size       = {config.batch_size}", flush=True)
print(f"LEARNING_RATE    = {config.LEARNING_RATE}", flush=True)
print(f"LEARNING_RATE2   = {config.LEARNING_RATE2}", flush=True)
print(f"N_ang_con        = {config.N_ang_con}", flush=True)
print(f"N_refresh_margin = {config.N_refresh_margin}", flush=True)
print("=======================================", flush=True)


####################################################################


class NN(nn.Module):
        def __init__(self,n_input,n_output,N_layer=config.N_LAYER,neuron_per_layer=config.NEURON_PER_LAYER):
                super().__init__()
                self.N_layer   = N_layer
                self.act_type  = config.act_type  # 0: tanh, 1: selu, 2: siren

                # ---- 層の定義 ----
                self.affine_layers = nn.ModuleDict()
                self.affine_layers["affine1"] = nn.Linear(n_input, neuron_per_layer)
                for i in range(2, self.N_layer):
                        self.affine_layers[f"affine{i}"] = nn.Linear(neuron_per_layer,neuron_per_layer)
                self.affine_layers[f"affine{self.N_layer}"] = nn.Linear(neuron_per_layer,n_output)

                # ---- 重みの初期化 ----
                for idx, layer in enumerate(self.affine_layers.values(), start=1):
                        if isinstance(layer, nn.Linear):
                                if self.act_type in (0, 1):
                                        # tanh, selu 用に Xavier
                                        nn.init.xavier_uniform_(layer.weight)
                                else:
                                        # SIREN 用の初期化
                                        in_features = layer.weight.size(1)
                                        if idx == 1:
                                                # 第一層: bound = 1 / in_features
                                                bound = 1.0 / in_features
                                        else:
                                                # それ以降: bound = sqrt(6/in_features) / w0  (w0=1)
                                                bound = math.sqrt(6.0 / in_features)
                                        nn.init.uniform_(layer.weight, -bound, bound)

                                if layer.bias is not None:
                                        nn.init.zeros_(layer.bias)

        def forward(self, t):
                # --- 第一層 + 活性化 ---
                x = self.affine_layers["affine1"](t)
                if   self.act_type == 0:
                        x = torch.tanh(x)
                elif self.act_type == 1:
                        x = torch.selu(x)
                else:
                        # SIREN 第一層は w0=30
                        x = torch.sin(30.0 * x)

                # --- 隠れ層 ---
                for i in range(2, self.N_layer):
                        x = self.affine_layers[f"affine{i}"](x)
                        if   self.act_type == 0:
                                x = torch.tanh(x)
                        elif self.act_type == 1:
                                x = torch.selu(x)
                        else:
                            # SIREN 隠れ層は w0=1
                                x = torch.sin(1.0 * x)

                # --- 出力層（活性化なし）---
                x = self.affine_layers[f"affine{self.N_layer}"](x)

                return x


class PINNs():
        def __init__(self):


                self.dnn   = NN(2, 5).to(device)

                if config.LOSS_TYPE < 2:
                        self.optimizer = torch.optim.Adam([
                        {'params': list(self.dnn.parameters()),
                         'lr': config.LEARNING_RATE},
                        ])
                else:
                        if config.LOSS_TYPE == 2:
                                self.log_sigma_Rd_uw = nn.Parameter(torch.zeros(1, device=device))
                        else:
                                size = 7 + config.SWITCH_LOSS2 + config.ang_con
                                self.log_sigma_Rd_uw = nn.Parameter(torch.zeros(size, device=device))

                        self.optimizer = torch.optim.Adam([
                        {'params': list(self.dnn.parameters()),
                         'lr': config.LEARNING_RATE},
                         {'params': [self.log_sigma_Rd_uw],
                          'lr':     config.LEARNING_RATE2},
                        ])


# 勾配計算
        def compute_grad(self, tensor, var, retain=True, create_graph=True):
                grad = torch.autograd.grad(
                tensor, var,
                grad_outputs=torch.ones_like(tensor),
                retain_graph=retain,
                create_graph=create_graph
                )[0]
                return grad


# ニューラルネットワーク変数aと定数a0
        def net_a(self, t, x, t_b2=None, x_b2=None):

                e_ic_tensor     = torch.tensor(config.e_ic,     dtype=torch.float32, device=device)
                sigma_om_tensor = torch.tensor(config.sigma_om, dtype=torch.float32, device=device)
                R_tensor        = torch.tensor(config.R,        dtype=torch.float32, device=device)
                rigid_tensor    = 1-(x[:,0]  *config.omega_ic)**2
                rigid_tensor_b  = 1-(R_tensor*config.omega_ic)**2

############################## Initial Condition ##############################
                with torch.no_grad():

                        a_0        = torch.zeros((x.shape[0], 5)).to(device)
                        a_0_r      = torch.zeros((x.shape[0], 5)).to(device)
                        d_a_0      = torch.zeros((x.shape[0], 5)).to(device)
                        d_a_0_r    = torch.zeros((x.shape[0], 5)).to(device)
                        a_b_0      = torch.zeros(5).to(device)
                        a_b_0_r    = torch.zeros(5).to(device)
                        a_c_0      = torch.zeros(5).to(device)
                        a_c_0_r    = torch.zeros(5).to(device)


# Rigid Rotation (Staionary Condition)
                        a_0[:,0]    = config.e_ic                                                       /            rigid_tensor**2
                        a_0[:,2]    = config.omega_ic                                                   / torch.sqrt(rigid_tensor)
                        a_0[:,3]    = config.omega_ic * torch.sqrt(e_ic_tensor) * 0.5 / config.factor_s /            rigid_tensor**2
                        a_0_r[:,0]  = config.e_ic                                                       * 4*x[:,0]*config.omega_ic**2 / rigid_tensor**3
                        a_0_r[:,2]  = config.omega_ic                                                   *   x[:,0]*config.omega_ic**2 / rigid_tensor**1.5
                        a_0_r[:,3]  = config.omega_ic * 0.5 / config.factor_s * torch.sqrt(e_ic_tensor) * 4*x[:,0]*config.omega_ic**2 / rigid_tensor**3

                        a_b_0[0]    = config.e_ic                                                       /            rigid_tensor_b**2
                        a_b_0[2]    = config.omega_ic                                                   / torch.sqrt(rigid_tensor_b)
                        a_b_0[3]    = config.omega_ic * torch.sqrt(e_ic_tensor) * 0.5 / config.factor_s /            rigid_tensor_b**2
                        a_b_0_r[0]    = config.e_ic                                                       * 4*config.R*config.omega_ic**2 / rigid_tensor_b**3
                        a_b_0_r[2]    = config.omega_ic                                                   *   config.R*config.omega_ic**2 / rigid_tensor_b**1.5
                        a_b_0_r[3]    = config.omega_ic * 0.5 / config.factor_s * torch.sqrt(e_ic_tensor) * 4*config.R*config.omega_ic**2 / rigid_tensor_b**3

                        a_c_0[0]    = config.e_ic
                        a_c_0[2]    = config.omega_ic
                        a_c_0[3]    = config.omega_ic * torch.sqrt(e_ic_tensor) * 0.5 / config.factor_s

                        if config.SWITCH_INIT==0:
# Orbital Initial Condition
                                omega_ic2    = 0.2*config.e_ic**0.25
                                d_a_0  [:,2] =                            omega_ic2 * torch.sin(  sigma_om_tensor*x[:,0])**4
                                d_a_0_r[:,2] = 2.0 * config.sigma_om    * omega_ic2 * torch.sin(2*sigma_om_tensor*x[:,0]) * torch.sin(sigma_om_tensor*x[:,0])**2
                                a_0  [:,2]  += d_a_0  [:,2]
                                a_0_r[:,2]  += d_a_0_r[:,2]

                        else:

                                omega_ic2    = 0.2 * config.e_ic**0.75
                                d_a_0[:,3]   =                         omega_ic2 * torch.sin(  sigma_om_tensor*x[:,0])**4
                                d_a_0_r[:,3] = 2.0 * config.sigma_om * omega_ic2 * torch.sin(2*sigma_om_tensor*x[:,0]) * torch.sin(sigma_om_tensor*x[:,0])**2
                                a_0[:,3]    += d_a_0  [:,3]
                                a_0_r[:,3]  += d_a_0_r[:,3]
##############################


############################## NN ##############################
                tt  = 0.5*(2.0*t   /config.t_max2 - 1.0)
                xx  = 0.5*(2.0*x   /config.R      - 1.0)
                a   = self.dnn(torch.cat([tt, xx], dim=1))


                t_i  = torch.zeros_like(t, dtype=torch.int)
                x_b  = torch.ones_like (x,  dtype=torch.int) * config.R
                tt_i = 0.5*(2.0*t_i   /config.t_max2 - 1.0)
                xx_b = 0.5*(2.0*x_b   /config.R      - 1.0)
                a_i  = self.dnn(torch.cat([tt_i, xx  ], dim=1))
                a_b  = self.dnn(torch.cat([tt  , xx_b], dim=1))
                a_ib = self.dnn(torch.cat([tt_i, xx_b], dim=1))

                x_b3    = np.random.uniform(low=0.9999 * config.R, high=config.R, size=t.shape)
                x_b3    = torch.tensor(x_b3, dtype=torch.float32, device=device, requires_grad=True)
                xx_b3   = 0.5*(2.0*x_b3   /config.R      - 1.0)
                a_b3    = self.dnn(torch.cat([tt,   xx_b3], dim=1))
                a_ib3   = self.dnn(torch.cat([tt_i, xx_b3], dim=1))
##############################


############################## Derivative ##############################
                a_t_list    = []
                a_r_list    = []
                ai_r_list   = []
                ab_t_list   = []

                ab3_r_list  = []
                aib3_r_list = []
                for i in range(a.shape[1]):
                        a_i_component  = a  [:, i]
                        ai_i_component = a_i[:, i]
                        ab_i_component = a_b[:, i]
                        a_t_i  = self.compute_grad(a_i_component,  t, retain=True, create_graph=True)
                        a_r_i  = self.compute_grad(a_i_component,  x, retain=True, create_graph=True)
                        ai_r_i = self.compute_grad(ai_i_component, x, retain=True, create_graph=True)
                        ab_t_i = self.compute_grad(ab_i_component, t, retain=True, create_graph=True)
                        a_t_list.append (a_t_i )
                        a_r_list.append (a_r_i )
                        ai_r_list.append(ai_r_i)
                        ab_t_list.append(ab_t_i)

                        ab3_i_component  = a_b3 [:, i]
                        aib3_i_component = a_ib3[:, i]
                        ab3_r_i  = self.compute_grad(ab3_i_component,  x_b3, retain=True, create_graph=True)
                        aib3_r_i = self.compute_grad(aib3_i_component, x_b3, retain=True, create_graph=True)
                        ab3_r_list.append(ab3_r_i )
                        aib3_r_list.append(aib3_r_i)


                a_t   = torch.cat(a_t_list,  dim=1)
                a_r   = torch.cat(a_r_list,  dim=1)
                a_i_r = torch.cat(ai_r_list, dim=1)
                a_b_t = torch.cat(ab_t_list, dim=1)

                a_b3_r  = torch.cat(ab3_r_list,  dim=1)
                a_ib3_r = torch.cat(aib3_r_list, dim=1)


##############################

                return a_0, a_0_r, a, a_t, a_r, a_i, a_i_r, a_b, a_b_t, a_ib, a_b3_r, a_ib3_r


        def net_f(self, t, x):
                a_0, a_0_r, a, a_t, a_r, a_i, a_i_r, a_b, a_b_t, a_ib, a_b3_r, a_ib3_r = self.net_a(t, x)


#################### Full ####################
### e, e_t, e_r
                e   = a_0  [:,0:1] + x*(a  [:,0:1]-a_i[:,0:1])
                e_t =                x* a_t[:,0:1]
                e_r = a_0_r[:,0:1] +   (a  [:,0:1]-a_i[:,0:1]) + x*(a_r[:,0:1]-a_i_r[:,0:1])


# u^r, u^r_t, u^r_r
# Boundary: lim_{r \to 0} \partial_r u^r = finite
# Boundary: lim_{r \to 0} u^r            = 0
# Boundary: lim_{r \to R} u^r            = 0
                ur_or   = (a  [:,1:2]-a_i[:,1:2]) - (a_b  [:,1:2]-a_ib[:,1:2])
                ur      = x * ur_or
                ur_t_or =  a_t[:,1:2]             -  a_b_t[:,1:2]
                ur_t    = x * ur_t_or
                ur_r    =   (a  [:,1:2]-a_i  [:,1:2]) -   (a_b  [:,1:2]-a_ib[:,1:2])\
                        + x*(a_r[:,1:2]-a_i_r[:,1:2])


# u^\theta, u^\theta_t, u^\theta_r
# Boundary: lim_{r \to 0} \partial_t u^th = finite
# Boundary: lim_{r \to 0} u^th = lim_{r \to 0} [2*config.factor_s*omega/\sqrt{e}]
                uth   = a_0  [:,2:3] + x*(a  [:,2:3]-a_i[:,2:3])
                uth_t =                x* a_t[:,2:3]
                uth_r = a_0_r[:,2:3] +   (a  [:,2:3]-a_i[:,2:3]) + x*(a_r[:,2:3]-a_i_r[:,2:3])


# \omega^3, \omega^3_t, \omega^3_r
# Boundary: lim_{r \to 0} \partial_r om = finite
# Boundary: lim_{r \to 0} om            = 0
# Boundary: lim_{r \to R} om            = 0
                om      = a_0  [:,3:4] + x * x * (a  [:,3:4]-a_i[:,3:4]) - x * x * (a_b  [:,3:4]-a_ib[:,3:4])
                om_t_or =                    x *  a_t[:,3:4]             -     x *  a_b_t[:,3:4]


# phi^{r \theta}, phi^{r \theta}_t, phi^{r \theta}_r
# Boundary: lim_{r \to 0} \partial_r u^th = finite
# Boundary: lim_{r \to 0} u^th            = 0
# Boundary: lim_{r \to R} u^th            = 0
                phi_or   = (a  [:,4:5]-a_i[:,4:5]) - (a_b  [:,4:5]-a_ib[:,4:5])
                phi      = x * phi_or
                phi_t_or =  a_t[:,4:5]             -  a_b_t[:,4:5]
                phi_t    = x * phi_t_or
                phi_r    =   (a  [:,4:5]-a_i  [:,4:5]) -   (a_b  [:,4:5]-a_ib[:,4:5])\
                         + x*(a_r[:,4:5]-a_i_r[:,4:5])


# u^t, u^t_t, u^t_
                ut            = torch.sqrt(1 + ur**2 + x*x*uth**2)
                ut_ut_t       = ur*ur_t + x*x*uth*uth_t
                ut_ut_r       = ur*ur_r + x*x*uth*uth_r + x * uth**2


# De, Du^r, Du^theta, D\ometa^3, Dphi^{r \theta}
                De   = ut * e_t   + ur * e_r
                Dur  = ut * ur_t  + ur * ur_r
                Duth = ut * uth_t + ur * uth_r
                Dphi = ut * phi_t + ur * phi_r


# Du^t
                ut_Dut = ut * ut_ut_t + ur * ut_ut_r

# theta
                ut_theta    = ut_ut_t    + ut*ur_r

# \partial_\mu \phi^{\mu \nu}, u_\nu \partial_\mu \phi^{\mu \nu}
                ut_ut_ut_par_phir      = x*x*phi*( uth*ut_ut_t     - ut*ut*uth_t    ) - x*x*ut*ut*uth*phi_t
                ut_ut_ut_par_phith    =     phi   *(-ur *ut_ut_t + ut*ut* ur_t) + ut*ut*ur *phi_t    + ut*ut*ut*phi_r
                ut_ut_u_par_phi      = -2*x*ut*ut*uth*phi     + x*x*phi   *(uth*ut_ut_r + ut*uth*ur_t - ut*ut*uth_r - ut*ur*uth_t)

# R
                if config.SWITCH_LOSS2 == 0:
                        Rd = torch.zeros((t.shape[0], 7)).to(device)
                else:
                        Rd = torch.zeros((t.shape[0], 8)).to(device)
                Rd[:,0:1] = De + 4/3*e*(ut_theta/ut+ut*ur_or) - ut_ut_u_par_phi/ut/ut

                Rd[:,1:2] = e*(Dur-x*uth**2) + 0.25*(ur*De+e_r) + 3*0.25*ur*ut_ut_u_par_phi/ut/ut + 3*0.25*ut_ut_ut_par_phir/ut/ut/ut

                Rd[:,2:3] = e*(Duth + 2*ur_or*uth) + 0.25*uth*De\
                          + 3*0.25*uth*ut_ut_u_par_phi/ut/ut + 3*0.25*ut_ut_ut_par_phith/ut/ut/ut +  3*0.25*phi_or

                Rd[:,3:4] = om_t_or + 2*phi

                ut_Dut = ur*Dur + x*x*uth*Duth + x*ur*uth**2

                # Duth  to T * Dbth   = Duth   + uth * T * Dbeta   = Duth   - 0.25 * uth * De  /e
                # Dur   to T * Dbr    = Dur    + ur  * T * Dbeta   = Dur    - 0.25 * ur  * De  /e
                # uth_r to T *  bth_r =  uth_r + uth * T *  beta_r =  uth_r - 0.25 * uth *  e_r/e

                TDbth  = Duth   - 0.25 * uth * De  /e
                TDbr   = Dur    - 0.25 * ur  * De  /e
                Tbth_r =  uth_r - 0.25 * uth *  e_r/e

                Rd[:,4:5] = config.tau_phi * torch.sqrt(torch.abs(e)) * Dphi\
                          +                  torch.sqrt(torch.abs(e)) * phi\
                          - config.tau_phi * torch.sqrt(torch.abs(e)) * phi    * (ut_Dut/ut/ut+x*ur*uth*uth-2/3*ut_theta/ut-5/3*ur_or)\
                          + config.gamma   * torch.sqrt(torch.abs(e))          * (ur*TDbth-uth*TDbr+Tbth_r+x*uth**3)\
                          + config.gamma   * 2*(torch.sqrt(torch.abs(e))*uth-2*config.factor_s*ut*om)/x


                if config.SWITCH_LOSS2 != 0:
                        om_t = x * om_t_or
                        Rd[:,7:8] = 4/3*x*x*(De*uth + e*uth*ut_theta/ut + e*Duth) \
                                  + x*x*ut_ut_ut_par_phith/ut/ut/ut\
                                  + 4*x*e*uth*ur + 3 * x*ut*phi/ut + om_t

                sqrt_r = torch.sqrt(x[:, 0]).unsqueeze(1)
                Rd = sqrt_r * Rd


### R
                ur_r      = config.R*(a_b3_r[:,1:2]-a_ib3_r[:,1:2])
                phi_r     = config.R*(a_b3_r[:,4:5]-a_ib3_r[:,4:5])
                Rd[:,5:6] = ur_r
                Rd[:,6:7] = phi_r



                return Rd


        def net_j(self, t, x):
                a_0, _, a, _, _, a_i, _, a_b, _, a_ib, *_ = self.net_a(t,x)

                e   = a_0  [:,0:1] +     x * (a  [:,0:1]-a_i[:,0:1])
                ur  =                    x * (a  [:,1:2]-a_i[:,1:2]) -     x * (a_b  [:,1:2]-a_ib[:,1:2])
                uth = a_0  [:,2:3] +     x * (a  [:,2:3]-a_i[:,2:3])
                om  = a_0  [:,3:4] + x * x * (a  [:,3:4]-a_i[:,3:4]) - x * x * (a_b  [:,3:4]-a_ib[:,3:4])
                phi =                    x * (a  [:,4:5]-a_i[:,4:5]) -     x * (a_b  [:,4:5]-a_ib[:,4:5])

                ut    = torch.sqrt(1 + ur**2 + x*x*uth**2)

                orb  = x*x*( 4/3*e*ut*uth + ur/ut*phi )
                spin = om
                j = orb + spin

                return j



        def loss_func(self, t_batch, x_batch, t_values, r_values, Nr):
                t_batch  = t_batch.clone().detach().requires_grad_(True)
                x_batch  = x_batch.clone().detach().requires_grad_(True)


                Rd_pred = self.net_f(t_batch, x_batch)


                if config.ang_con == 0:
                        loss_Rd_2 = torch.mean(Rd_pred**2, dim=0)

###################################################################################################
                        t_values = t_values.clone().detach().requires_grad_(True)
                        r_values = r_values.clone().detach().requires_grad_(True)

                        dr  = config.R / Nr
                        Nt, Nr = t_values.shape[0], r_values.shape[0]
                        T_in = t_values.view(Nt,1).repeat(1, Nr).view(-1,1)   # (Nt*Nr,1)
                        R_in = r_values.view(1,Nr).repeat(Nt,1).view(-1,1)    # (Nt*Nr,1)

                        j_all = self.net_j(T_in, R_in)       # (Nt*Nr,1), requires_grad=True
                        weights = 2 * np.pi * R_in * dr # (Nt*Nr,1)
                        j_all = (j_all * weights).view(Nt, Nr)
                        j_total = j_all.sum(dim=1)     # shape (Nt,)
                        j0 = j_total[0].detach()
                        loss_j = torch.mean((j_total[1:]/j0 - 1.0)**2)
###################################################################################################

                else:
                        t_values = t_values.clone().detach().requires_grad_(True)
                        r_values = r_values.clone().detach().requires_grad_(True)

                        dr  = config.R / Nr
                        Nt, Nr = t_values.shape[0], r_values.shape[0]
                        T_in = t_values.view(Nt,1).repeat(1, Nr).view(-1,1)   # (Nt*Nr,1)
                        R_in = r_values.view(1,Nr).repeat(Nt,1).view(-1,1)    # (Nt*Nr,1)

                        j_all = self.net_j(T_in, R_in)       # (Nt*Nr,1), requires_grad=True
                        weights = 2 * np.pi * R_in * dr # (Nt*Nr,1)
                        j_all = (j_all * weights).view(Nt, Nr)
                        j_total = j_all.sum(dim=1)     # shape (Nt,)
                        j0 = j_total[0].detach()
                        loss_j = torch.mean((j_total[1:]/j0 - 1.0)**2)

                        loss_Rd_2_dot = torch.mean(Rd_pred**2, dim=0)
                        loss_Rd_2     = torch.cat([loss_Rd_2_dot, loss_j.unsqueeze(0)], dim=0)

                loss_Rd_dot = loss_Rd_2.clone()

                if config.LOSS_TYPE == 0:
                        loss_Rd = torch.mean(loss_Rd_2)
                        loss    = loss_Rd.clone()


                elif config.LOSS_TYPE == 1:
                        #################
                        log_sigma_Rd_tensor = torch.tensor(config.log_sigma_Rd, device=device)
                        loss = 0.5 * torch.exp(-log_sigma_Rd_tensor) * torch.mean(loss_Rd_2)  + 0.5 * log_sigma_Rd_tensor
                        #################

                        loss_Rd_dot  = torch.cat([loss_Rd_dot,torch.exp(log_sigma_Rd_tensor).unsqueeze(0)], dim=0)
                        loss_Rd     = torch.mean(loss_Rd_2)


                else:
                        #################
                        if config.LOSS_TYPE == 2:
                                loss = 0.5 * torch.exp(-self.log_sigma_Rd_uw) * torch.mean(loss_Rd_2)  + 0.5 * self.log_sigma_Rd_uw
                                loss_Rd_dot  = torch.cat([loss_Rd_dot,torch.exp(self.log_sigma_Rd_uw)], dim=0)
                        else:

                                delta = math.log(100.0)
                                alpha = self.log_sigma_Rd_uw
                                max_alpha, _ = alpha[:5].max(dim=0, keepdim=True)
                                min_alpha = max_alpha - delta
                                alpha_clamped = torch.clamp(alpha, min=min_alpha)
                                loss = torch.mean( 0.5 * torch.exp(-alpha_clamped) * loss_Rd_2  + 0.5 * alpha_clamped )
                                loss_Rd_dot  = torch.cat([loss_Rd_dot,torch.exp(alpha),torch.exp(alpha_clamped)], dim=0)
                        #################

                        loss_Rd     = torch.mean(loss_Rd_2[:5])


                return loss, loss_Rd, loss_Rd_dot, loss_j



    # 学習
        def train(self):


                # GradScalerのインスタンスを作成
                scaler = torch.cuda.amp.GradScaler()

                if config.log_file_option == 0:
                        file_name = f"training_log_{current_timestamp}.txt"

                else:
                        file_name = config.log_file_name

                        if config.remove_old_log and config.global_step == 0 and os.path.exists(file_name):
                                os.remove(file_name)

                with open(file_name, 'a') as log_file:

                        Nt = round(config.t_max/(config.t_bin/config.N_ang_con))
                        delta_t = config.t_max / Nt
                        it = torch.arange(Nt-1, device=device, dtype=torch.float32)
                        t_blocks = (it + torch.rand(Nt-1, device=device)) * delta_t
                        t_values = torch.cat((torch.tensor([0.0], device=device),t_blocks,torch.tensor([config.t_max], device=device)))


                        Nr = 1000
                        dr = config.R / Nr
                        r_values = torch.linspace(0.5*dr, config.R-0.5*dr, Nr, device=device)

#####
                        check = 1
                        if config.EPOCHS > 100: check = config.EPOCHS//100

                        for epoch in range(config.EPOCHS):

                                for t_batch, x_batch in dataloader:

                                        t_batch  = t_batch.to(device)
                                        x_batch  = x_batch.to(device)


                                        # モデルを訓練モードに設定
                                        self.dnn.train()
                                        # 勾配のリセット
                                        self.optimizer.zero_grad()
                                        # 混合精度での順伝播と損失計算
                                        with torch.cuda.amp.autocast():
                                            # 損失関数を計算（ミニバッチのデータを使用）
                                            self.loss, self.loss_1, self.loss_Rd_dot, self.loss_j = self.loss_func(t_batch,x_batch,t_values,r_values,Nr)
                                        # 損失に基づいて勾配を計算（勾配スケーリングを適用）
                                        scaler.scale(self.loss).backward()
                                        # オプティマイザを使ってモデルのパラメータを更新
                                        scaler.step(self.optimizer)
                                        scaler.update()
                                        config.global_step += 1

                                        if config.N_obs > 0 and config.global_step >= config.N_obs:


                                                model_name = config.obs_model_name
                                                torch.save(self, model_name)

                                                print(
                                                        "Saved observation model:",
                                                        model_name,
                                                        "at global_step =",
                                                        config.global_step,
                                                        flush=True
                                                )

                                                return True                                # 10エポックごとに損失を表示

                                config.count_epoch = epoch
                                note_epoch = config.sum_epoch + config.count_epoch

                                if epoch % check ==0:
                                        print( 'Iter %d, Loss: %.5e, %.5e, %.5e'\
                                           % (epoch, self.loss.item()
                                             , self.loss_1.item(), self.loss_j.item()
                                             ))
                                        log_file.write(
                                                f'{config.global_step:d} '
                                                f'{note_epoch:.5e} '
                                                f'{self.loss.item():.5e} '
                                                f'{self.loss_1.item():.5e} '
                                                f'{self.loss_j.item():.5e} '
                                                f'{" ".join(["%.2e" % x for x in self.loss_Rd_dot.tolist()])}\n'
                                        )
                                        log_file.flush()
                return False







        def predict(self, t, x): # 入力された座標 x と時間 t に基づいて、モデルの予測を行います。

                t = torch.tensor(t, requires_grad=True).float().to(device)  # tをGPUに移動
                x = torch.tensor(x, requires_grad=True).float().to(device)  # xをGPUに移動

                # eval() 関数で評価モードに設定し、パラメータの更新が行われないようにします。
                self.dnn.eval()
                a_0, a_0_r, a, a_t, a_r, a_i, _, a_b, _, a_ib, *_ = self.net_a(t,x)

                e   = a_0  [:,0:1] +     x * (a  [:,0:1]-a_i[:,0:1])
                ur  =                    x * (a  [:,1:2]-a_i[:,1:2]) -     x * (a_b  [:,1:2]-a_ib[:,1:2])
                uth = a_0  [:,2:3] +     x * (a  [:,2:3]-a_i[:,2:3])
                om  = a_0  [:,3:4] + x * x * (a  [:,3:4]-a_i[:,3:4]) - x * x * (a_b  [:,3:4]-a_ib[:,3:4])
                phi =                    x * (a  [:,4:5]-a_i[:,4:5]) -     x * (a_b  [:,4:5]-a_ib[:,4:5])

# u^t
                ut    = torch.sqrt(1 + ur**2 + x*x*uth**2)

# T^tt
                Ttt = e*ut**2 + e/3 * (ut**2-1.0)

# J^txy = sigma^{txy} + ( xT^ty - yT^tx ) = omega^3 + r^2 T^{t \theta}
# Jtxy  = om + x*x*( 4/3*e*ut*uth + ur/ut*phi )

# l^3 = r^2 u^\theta
                orb  = x*x*( 4/3*e*ut*uth + ur/ut*phi )

# s^3 = u^t \omega^3
                spin = om

#  Rd, Rd_con, Rd_i
                Rd = self.net_f(t,x)

                return e, ur, uth, om, phi, Ttt, orb, spin, Rd


original_Np         = config.Np
original_EPOCHS     = config.EPOCHS
original_batch_size = config.batch_size

i_time = 1.0
config.t_max      = i_time * config.t_bin
config.Np         = round(i_time * original_Np)
Np_A              = config.Np - config.Np % config.batch_size
Np_B              = Np_A + config.batch_size
config.Np         = Np_A if (config.Np - Np_A) <= (Np_B - config.Np) else Np_B
config.EPOCHS     = round(original_EPOCHS/i_time)

file_number =  0                 + 1000 * round(config.SWITCH_LOSS2 + 2*config.LOSS_TYPE + 8*config.ang_con + 16*config.SWITCH_INIT)
N_print     = config.N_print_max + 1000 * round(config.SWITCH_LOSS2 + 2*config.LOSS_TYPE + 8*config.ang_con + 16*config.SWITCH_INIT)
print(file_number,i_time,config.t_max,config.Np,config.EPOCHS,config.batch_size)

pinns = PINNs()
#pinns = torch.load(f'modelUspin250731_0_{file_number}.pth', map_location=torch.device('cuda'))

t_c,  x_c  = data_handler.generate_data()
#######################################################
N_bunkatu   = 5
Np_original = config.Np
Np_sample   = int(Np_original / N_bunkatu)
for k in range(N_bunkatu):
        # コロケーション候補点を取得
        if k < N_bunkatu-1:
                config.Np = Np_sample
                t_c2, x_c2 = data_handler.generate_data2()
                t_c[k * Np_sample:(k + 1) * Np_sample, :] = t_c2
                x_c[k * Np_sample:(k + 1) * Np_sample, :] = x_c2
        else:
                config.Np = Np_sample
                t_c2, x_c2 = data_handler.generate_data2()
                t_c[k * Np_sample:(k + 1) * Np_sample, :] = t_c2
                x_c[k * Np_sample:(k + 1) * Np_sample, :] = x_c2

config.Np = Np_original
#######################################################
count      = 0
reach_t_max = 0
lr4_count = 0
lr5_has_started = False
config.sum_epoch = 0
config.global_step = 0


for i in range(file_number + 1, N_print + 1):

        j  = i % 100
        count = count+1
        if j != 1:
                Nt = 20
                t_values = np.linspace(0, config.t_max, Nt)  # tau_maxまでの100サンプルの時間

                Nr       = 2000  # グリッドサイズ（例: 100）
                dr       = config.R/Nr
                r_values = np.linspace(0.5*dr, config.R-0.5*dr, Nr)

                pinns.dnn.eval()

                e_total_values  = []
                l_total_values  = []
                s_total_values  = []
                Rd_total_values = []

                for t_val in t_values:
                        e_sum  = 0.0
                        l_sum  = 0.0
                        s_sum  = 0.0
                        Rd_sum = 0.0
                        t_current = np.full(Nr, t_val, dtype=np.float32).reshape(-1, 1)
                        r_current = r_values.astype(np.float32).reshape(-1, 1)

                        _, _, _, _, _, e, orb, spin, Rd = pinns.predict(t_current, r_current)
                        e_sum  = np.sum(e.detach().cpu().numpy()    * 2 * np.pi * r_current * dr)
                        l_sum  = np.sum(orb.detach().cpu().numpy()  * 2 * np.pi * r_current * dr)
                        s_sum  = np.sum(spin.detach().cpu().numpy() * 2 * np.pi * r_current * dr)

                        Rd_np = Rd.detach().cpu().numpy()
                        Rd_first5 = Rd_np[:, :5]
                        Rd_sum = np.sum(Rd_first5**2 * 2 * np.pi * dr, axis=0)

                        e_total_values.append(e_sum)
                        l_total_values.append(l_sum)
                        s_total_values.append(s_sum)
                        Rd_total_values.append(Rd_sum)
                Rd_arr = np.stack(Rd_total_values, axis=0)
                Rd_sum = Rd_arr.sum(axis=0)
                D_Rd   = Rd_sum.shape[0]
                Rd_sum/= (D_Rd * Nt * np.pi * config.R**2)
                max_Rd = np.max(Rd_sum)
                # --- 1. t=0 の基準値を取り出す ---
                e0 = e_total_values[0]
                L0 = l_total_values[0] + s_total_values[0]
                # --- 2. 全時刻にわたって相対誤差を計算 ---
                e_arr = np.array(e_total_values)
                L_arr = np.array(l_total_values) + np.array(s_total_values)
                # 0除算を避けるため、絶対値ゼロのときは分母を1に置き換え
                den_e = e0 if abs(e0) > 0 else 1.0
                den_L = L0 if abs(L0) > 0 else 1.0
                rel_err_e = np.abs(e_arr - e0) / np.abs(den_e)
                rel_err_L = np.abs(L_arr - L0) / np.abs(den_L)
                max_rel_err_e = np.max(rel_err_e)
                max_rel_err_L = np.max(rel_err_L)


                allow_refresh = config.global_step < config.N_obs - config.N_refresh_margin
                if (max_rel_err_e <= 0.01 and max_rel_err_L <= 0.01 and max_Rd <= 0.01 and reach_t_max != 2 and allow_refresh):
                        i_time = 1.2 * i_time
                        config.t_max      = i_time * config.t_bin
                        config.Np         = round(i_time * original_Np)
                        Np_A              = config.Np - config.Np % config.batch_size
                        Np_B              = Np_A + config.batch_size
                        config.Np         = Np_A if (config.Np - Np_A) <= (Np_B - config.Np) else Np_B
                        config.EPOCHS     = round(original_EPOCHS/i_time)

                        optimizer = pinns.optimizer
                        new_lr1, new_lr2 = 1e-3, 1e-3
                        for i_lr, pg in enumerate(optimizer.param_groups):
                            if i_lr == 0:
                                pg['lr'] = new_lr1
                            elif i_lr == 1:
                                pg['lr'] = new_lr2
                        print([{'lr': g['lr']} for g in optimizer.param_groups])



                        if config.t_max > config.t_max2:
                                config.t_max  = config.t_max2
                                reach_t_max   = 2



                                if reach_t_max  == 2:
                                        optimizer = pinns.optimizer

                                        # Do not immediately reduce to 1e-5.
                                        # Keep lr = 1e-4 for several outer iterations.
                                        new_lr1, new_lr2 = 1e-4, 1e-4

                                        for i_lr, pg in enumerate(optimizer.param_groups):
                                            if i_lr == 0:
                                                pg['lr'] = new_lr1
                                            elif i_lr == 1:
                                                pg['lr'] = new_lr2

                                        lr4_count = 0
                                        lr5_has_started = False

                                        print("Enter final-stage lr=1e-4:",
                                              [{'lr': g['lr']} for g in optimizer.param_groups])


                                        i_time            = i_time/1.2
                                        config.Np         = round(i_time * original_Np)
                                        Np_A              = config.Np - config.Np % config.batch_size
                                        Np_B              = Np_A + config.batch_size
                                        config.Np         = Np_A if (config.Np - Np_A) <= (Np_B - config.Np) else Np_B
                                        config.EPOCHS     = round(original_EPOCHS/i_time)


                        t_c,  x_c  = data_handler.generate_data()
                        #######################################################
                        N_bunkatu   = 5
                        Np_original = config.Np
                        Np_sample   = int(Np_original / N_bunkatu)
                        for k in range(N_bunkatu):
                                # コロケーション候補点を取得
                                if k < N_bunkatu-1:
                                        config.Np = Np_sample
                                        t_c2, x_c2 = data_handler.generate_data2()
                                        t_c[k * Np_sample:(k + 1) * Np_sample, :] = t_c2
                                        x_c[k * Np_sample:(k + 1) * Np_sample, :] = x_c2
                                else:
                                        config.Np = Np_sample
                                        t_c2, x_c2 = data_handler.generate_data2()
                                        t_c[k * Np_sample:(k + 1) * Np_sample, :] = t_c2
                                        x_c[k * Np_sample:(k + 1) * Np_sample, :] = x_c2

                        config.Np = Np_original
                        #######################################################


                        print(f"All within 1%: advancing i_time to {i_time}")
                        print(f"Max of rel_L = {max_rel_err_L}", f"Max of rel_e = {max_rel_err_e}", f"Max of Rd = {max_Rd}")
                        print(
                        "outer_i, count, t_max, Np, EPOCHS, batch_size, global_step =",
                        i, count, config.t_max, config.Np, config.EPOCHS, config.batch_size, config.global_step
                        )
                        count=0
                else:
                        if count % 5 == 1 and count != 1 and allow_refresh:

                                t_c, x_c   = data_handler.generate_data()
                                #######################################################
                                N_bunkatu   = 5
                                Np_original = config.Np
                                Np_sample   = int(Np_original / N_bunkatu)

                                # radial-bin residual-informed resampling parameters
                                N_bins           = 100

                                # probe settings for estimating radial-bin residual weights
                                N_probe_total = 50000
                                N_probe_chunk = 5000
                                eps_weight    = 1.0e-12


                                for k in range(N_bunkatu):

                                        # Uniform sampling:
                                        # - all blocks if uniform_option == 0
                                        # - first 4/5 blocks if uniform_option == 1
                                        if config.uniform_option == 0 or k < N_bunkatu-1:
                                                config.Np = Np_sample
                                                t_c2, x_c2 = data_handler.generate_data2()
                                                t_c[k * Np_sample:(k + 1) * Np_sample, :] = t_c2
                                                x_c[k * Np_sample:(k + 1) * Np_sample, :] = x_c2

                                                # Last 1/5 block: radial-bin residual-weighted sampling
                                                # This block is used only when uniform_option == 1.
                                        else:

                                                # ---- estimate residual weight for each radial bin ----
                                                bins = np.linspace(0.0, config.R, N_bins + 1)
                                                bin_sum   = np.zeros(N_bins)
                                                bin_count = np.zeros(N_bins)

                                                pinns.dnn.eval()

                                                n_done = 0
                                                while n_done < N_probe_total:
                                                        n_now = min(N_probe_chunk, N_probe_total - n_done)
                                                        config.Np = n_now

                                                        t_probe, x_probe = data_handler.generate_data2()
                                                        t_probe = torch.tensor(t_probe, requires_grad=True).float().to(device)
                                                        x_probe = torch.tensor(x_probe, requires_grad=True).float().to(device)

                                                        f_pred = pinns.net_f(t_probe, x_probe)

                                                        # Use only the first five local residuals for radial-bin weighting.
                                                        # Boundary residuals Rd[:,5:7] are generated from separate near-boundary points,
                                                        # and the angular-momentum residual Rd[:,7] is not used for this weighting.
                                                        res2 = torch.mean(f_pred[:, :5]**2, dim=1).detach().cpu().numpy()
                                                        r_np = x_probe.detach().cpu().numpy().ravel()

                                                        bin_id = np.clip(np.digitize(r_np, bins) - 1, 0, N_bins - 1)

                                                        np.add.at(bin_sum,   bin_id, res2)
                                                        np.add.at(bin_count, bin_id, 1.0)

                                                        del t_probe, x_probe, f_pred
                                                        n_done += n_now

                                                bin_mean = bin_sum / np.maximum(bin_count, 1.0)

                                                # Fill empty bins, just in case.
                                                if np.any(bin_count == 0):
                                                        global_mean = np.sum(bin_sum) / max(np.sum(bin_count), 1.0)
                                                        bin_mean[bin_count == 0] = global_mean

                                                # Use sqrt to avoid overconcentration on a few high-residual bins.
                                                weights = np.sqrt(bin_mean + eps_weight)

                                                if (not np.all(np.isfinite(weights))) or np.sum(weights) <= 0.0:
                                                        probs = np.ones(N_bins) / N_bins
                                                else:
                                                        probs = weights / np.sum(weights)

                                                # ---- sample weighted radial bins ----
                                                chosen_bins = np.random.choice(N_bins, size=Np_sample, replace=True, p=probs)

                                                r_low  = bins[chosen_bins]
                                                r_high = bins[chosen_bins + 1]

                                                # Avoid exactly r=0.
                                                r_low = np.maximum(r_low, 0.0001 * config.R)


                                                x_new = r_low + (r_high - r_low) * np.random.uniform(size=Np_sample)
                                                x_new = x_new.reshape(-1, 1)


                                                t_new = np.random.uniform(
                                                        low=0.0,
                                                        high=config.t_max,
                                                        size=(Np_sample, 1)
                                                )


                                                t_c[k * Np_sample:(k + 1) * Np_sample, :] = t_new
                                                x_c[k * Np_sample:(k + 1) * Np_sample, :] = x_new

                                                del t_new, x_new
                                                torch.cuda.empty_cache()


                                config.Np = Np_original
                                print("Refresh Collocation Points")

                                #######################################################

                        print("Not yet within 1%, keeping i_time =", i_time)
                        print(f"Max of rel_L = {max_rel_err_L}", f"Max of rel_e = {max_rel_err_e}", f"Max of Rd = {max_Rd}")
                        print(
                                "outer_i, count, t_max, Np, EPOCHS, batch_size, global_step =",
                                i, count, config.t_max, config.Np, config.EPOCHS, config.batch_size, config.global_step
                        )
        dataset     = TensorDataset(torch.tensor(t_c).float(),  torch.tensor(x_c).float())
        dataloader  = DataLoader(dataset, batch_size=config.batch_size,     shuffle=True, pin_memory=True)

        config.current_outer_i = i
        hit_N_obs = pinns.train()

        if hit_N_obs:
                print(
                        "Training stopped because global_step reached N_obs =",
                        config.N_obs,
                        flush=True
                )
                break

        if reach_t_max == 2 and not lr5_has_started:
                lr4_count += 1

                if lr4_count >= 5:
                        optimizer = pinns.optimizer
                        new_lr1, new_lr2 = 1e-5, 1e-5

                        for i_lr, pg in enumerate(optimizer.param_groups):
                            if i_lr == 0:
                                pg['lr'] = new_lr1
                            elif i_lr == 1:
                                pg['lr'] = new_lr2

                        lr5_has_started = True

                        print("Start final fine-tuning lr=1e-5:",
                              [{'lr': g['lr']} for g in optimizer.param_groups])


        model_tag = os.path.splitext(config.obs_model_name)[0]
        model_name = f'{model_tag}_{config.file_time}_{config.file_gamma}_{i}.pth'

        torch.save(pinns, model_name)
        print(
        "Saved model:",
        model_name,
        "at global_step =",
        config.global_step
        )
        config.sum_epoch += config.count_epoch

exit()
