import torch                   
import torch.nn as nn          
import numpy as np             
import math
import matplotlib.pyplot as plt
import sobol_seq               
#import matplotlib as mpl
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp   import autocast, GradScaler
from scipy.stats      import norm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import time
current_timestamp = time.time()



# scipyのモジュールであるscipy.statsの中にあるサブモジュールがqmcらしい
# ライブラリ      = パッケージやモジュールが集まったもの。単なるディレクトリではないらしい
# パッケージ      = __init__.pyが含まれているディレクトリ
# __init__.py     = ディレクトリをパッケージと認識させるファイル。パッケージの初期化を行う。
# モジュール      = ファイル
# サブモジュール・サブパッケージ = パッケージの中に入っているモジュールやパッケージ
# importするのはモジュールやパッケージであることがおおい。
# 例えば、"import numpy"はライブラリnumpyの中のnumpyパッケージをインポートしている。


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
# GPUが利用可能であれば'cuda'、そうでなければ'cpu'を使用
class Config: # Parameterのセット
        Nc           = 3                  # SU(N)ゲージ理論における色の数
        Nadj         = Nc * Nc - 1        # adjoint表現（ゲージ場の自由度）の数
        Nf           = 2                  # 軽いquarkの数 
        R            = 1                  # 系の半径を表すパラメータ, (b_{imp}/2=R_A/2) 3fm=16[GeV unit]
        t_bin        = 0.2                # 時間の最大値です。10fm=50 [GeV unit]
        t_max        = 0.2                # 時間の最大値です。10fm=50 [GeV unit]
        t_max2       = 0.5                # 時間の最大値です。10fm=50 [GeV unit]
        omega_ic     = 0.0/R              # 角速度 [GeV unit]（初期条件）,  0.007[GeV unit]
        omega_ic_dot = 0.0/R              # 角速度 [GeV unit]（初期条件）,  0.007[GeV unit]
        e_ic         = 1                  # energy density [GeV unit]（初期条件）0.004 [GeV unit]
        eta          = 0   * e_ic**0.75   # shear viscosity, 2.0 (KS bound)
        zeta         = 0   * e_ic**0.75   # bulk  viscosity
        gamma        = 2.0 * e_ic**0.75   # spin transport coefficient
        tau_sh       = eta  /e_ic       # 緩和時間 [GeV unit]
        tau_bu       = zeta /e_ic       # 緩和時間 [GeV unit]
        tau_phi      = gamma/e_ic       # 緩和時間 [GeV unit]
        factor_s     = 6/19 * np.sqrt(29/15) * np.pi
        N_LAYER          =    5            # ニューラルネットワークの層の数
        NEURON_PER_LAYER =  250            # 64*ne_x*nb_y # 各層に含まれるニューロンの数
        Np               =  25000           # 160000 PINNsで学習に使用するデータ点の数（コロケーションポイント）
        EPOCHS           =  1000           # 1000    # 学習におけるエポック数（全データセットに対して学習する回数）
        batch_size       =  Np//5          #//20    #        # ミニバッチサイズ %5で割り切れなければならない
        LEARNING_RATE    =  1e-3           # ニューラルネットワークの学習率
        LEARNING_RATE2   =  1e-3           # ニューラルネットワークの学習率
        epsilon_om       = 4
        sigma_om         = np.pi/R
        LOSS_WEIGHT      = 1.0       # 損失関数に対する重み
        LOSS_TYPE        = 3         # 0: Normal, 1: Uncertainty Weighting, 2: Uncertainty Weighting (ver2), 3: Uncertainty Weighting (ver3)
        SWITCH_LOSS      = 0         # 0: w/o boundary loss, 1: w/ boundary loss
        SWITCH_LOSS2     = 1         # 0: w/o ang loss, 1: w/ ang loss
        act_type         = 0         # 0:tanh, 1:SeLu, 2:sin
        ang_con          = 1         # 0: w/o conservation, 1:w/ conservation
        N_ang_con        = 5.0
        SWITCH_INIT      = 0         # 0: orbital, 1: spin
        log_sigma_Rd     = 0.0       #
        log_sigma_Rd_b2  = 0.0       #
        count_epoch      = 0       #
        sum_epoch        = 0       #
#    e  = 29/15 *np.pi*np.pi * T**4 # energy density for Nc=3, Nf=2
# w/ e1 and e2 plus large net_j

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
config               = Config()            # Config オブジェクトを作成, configはクラス
data_handler         = DataHandler(config) # データを生成するクラス。クラスがクラスの引数となっている。
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

##                        if config.SWITCH_LOSS == 0:
                        self.optimizer = torch.optim.Adam([
                        {'params': list(self.dnn.parameters()),
                         'lr': config.LEARNING_RATE},
                         {'params': [self.log_sigma_Rd_uw],
                          'lr':     config.LEARNING_RATE2},
                        ])

##                        else:
##                                self.log_sigma_Rd_b2_uw = nn.Parameter(torch.zeros(4, device=device))
##
##                                self.optimizer = torch.optim.Adam([
##                                {'params': list(self.dnn.parameters()),
##                                 'lr': config.LEARNING_RATE,},
##                                 {'params': [self.log_sigma_Rd_uw, self.log_sigma_Rd_b2_uw],
##                                  'lr':     config.LEARNING_RATE2,},
##                                ])

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


##                if x_b2 is None:
##                        a_b2    = torch.zeros_like(a)
##                        a_b2_r  = torch.zeros_like(a)
##                        a_b2_t  = torch.zeros_like(a)
##
##                        a_ib2   = torch.zeros_like(a)
##                        a_ib2_r = torch.zeros_like(a)
##                else:
##                        t_ib2   = torch.zeros_like(t_b2, dtype=torch.int)  
##                        tt_b2   = 2.0*t_b2   /config.R      - 1.0
##                        tt_ib2  = 2.0*t_ib2  /config.R      - 1.0
##                        xx_b2   = 2.0*x_b2   /config.R      - 1.0
##
##                        a_b2    = self.dnn(torch.cat([tt_b2 , xx_b2], dim=1))
##                        a_ib2   = self.dnn(torch.cat([tt_ib2, xx_b2], dim=1))

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
##                ab2_t_list  = []
##                ab2_r_list  = []
##                aib2_r_list = []

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


##                        if x_b2 is not None:
##                                ab2_i_component  = a_b2 [:, i]
##                                aib2_i_component = a_ib2[:, i]
##                                ab2_t_i  = self.compute_grad(ab2_i_component,  t_b2, retain=True, create_graph=True)
##                                ab2_r_i  = self.compute_grad(ab2_i_component,  x_b2, retain=True, create_graph=True)
##                                aib2_r_i = self.compute_grad(aib2_i_component, x_b2, retain=True, create_graph=True)
##                                ab2_t_list.append(ab2_t_i )
##                                ab2_r_list.append(ab2_r_i )
##                                aib2_r_list.append(aib2_r_i)


                a_t   = torch.cat(a_t_list,  dim=1)
                a_r   = torch.cat(a_r_list,  dim=1)
                a_i_r = torch.cat(ai_r_list, dim=1)
                a_b_t = torch.cat(ab_t_list, dim=1)

                a_b3_r  = torch.cat(ab3_r_list,  dim=1)
                a_ib3_r = torch.cat(aib3_r_list, dim=1)


##                if x_b2 is not None:
##                        a_b2_t  = torch.cat(ab2_t_list,  dim=1)
##                        a_b2_r  = torch.cat(ab2_r_list,  dim=1)
##                        a_ib2_r = torch.cat(aib2_r_list, dim=1)
##############################

##                return a_0, a_0_r, a, a_t, a_r, a_i, a_i_r, a_b, a_b_t, a_ib, a_b2_r, a_ib2_r, a_b_0, a_b_0_r, a_b2, a_ib2, a_b2_t, a_b3_r, a_ib3_r
                return a_0, a_0_r, a, a_t, a_r, a_i, a_i_r, a_b, a_b_t, a_ib, a_b3_r, a_ib3_r


##        def net_f(self, t, x, t_b2=None, x_b2=None):
##                if t_b2 is None:
##                        t_b2 = torch.zeros_like(t)
##                        a_0, a_0_r, a, a_t, a_r, a_i, a_i_r, a_b, a_b_t, a_ib, a_b2_r, a_ib2_r, a_b_0, a_b_0_r, a_b2, a_ib2, a_b2_t, a_b3_r, a_ib3_r = self.net_a(t, x)
##                
##                else:
##                        a_0, a_0_r, a, a_t, a_r, a_i, a_i_r, a_b, a_b_t, a_ib, a_b2_r, a_ib2_r, a_b_0, a_b_0_r, a_b2, a_ib2, a_b2_t, a_b3_r, a_ib3_r = self.net_a(t, x, t_b2, x_b2)
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
                phi_r    =   (a  [:,4:5]-a_i  [:,4:5]) -   (a_b  [:,1:2]-a_ib[:,4:5])\
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
                Rd[:,4:5] = config.tau_phi * torch.sqrt(torch.abs(e)) * Dphi\
                          +                  torch.sqrt(torch.abs(e)) * phi\
                          - config.tau_phi * torch.sqrt(torch.abs(e)) * phi    * (ut_Dut/ut/ut+x*ur*uth*uth-2/3*ut_theta/ut-5/3*ur_or)\
                          + config.gamma   * torch.sqrt(torch.abs(e))          * (ur*Duth-uth*Dur+uth_r+x*uth**3)\
                          + config.gamma   * 2*(torch.sqrt(torch.abs(e))*uth-2*config.factor_s*ut*om)/x

#                Rd[:,0:1] = 0.1 * Rd[:,0:1]
#                Rd[:,1:2] = 0.1 * Rd[:,1:2]

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

##                Rd_b2 = torch.zeros((t_b2.shape[0], 4)).to(device)            
##                if config.SWITCH_LOSS != 0:                        
##                        e     = a_b_0[0:1]   + config.R * (a_b2  [:,0:1]-a_ib2[:,0:1])
##                        e_t   =                config.R *  a_b2_t[:,0:1]
##                        e_r   = a_b_0_r[0:1] +            (a_b2  [:,0:1]-a_ib2[:,0:1]) + config.R * (a_b2_r[:,0:1]-a_ib2_r[:,0:1])
##                        uth   = a_b_0  [2:3] + config.R * (a_b2  [:,2:3]-a_ib2[:,2:3])
##                        uth_t =                config.R *  a_b2_t[:,2:3]
##                        uth_r = a_b_0_r[2:3] +            (a_b2  [:,2:3]-a_ib2[:,2:3]) + config.R * (a_b2_r[:,2:3]-a_ib2_r[:,2:3])
##                        ut    = torch.sqrt(1 + config.R*config.R*uth**2)      
##
##                        Rd_b2[:,0:1] =  ut*ut*e_t         + 4/3*e*config.R*config.R*uth*uth_t # e_t, uth_t
##                        Rd_b2[:,1:2] = -e*config.R*uth**2 + 0.25    *e_r                      # e_r
##                        Rd_b2[:,2:3] =  e*         uth_t  + 0.25*uth*e_t                      # uth_t, e_t
##                        Rd_b2[:,3:4] =  config.gamma * (uth_r+config.R*uth**3+2*uth/config.R) # uth_r


##                return Rd ##, Rd_b2
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



##        def loss_func(self, t_batch, x_batch, t_batch2, x_batch2, t_values, r_values, Nr):
        def loss_func(self, t_batch, x_batch, t_values, r_values, Nr):
                t_batch  = t_batch.clone().detach().requires_grad_(True)
                x_batch  = x_batch.clone().detach().requires_grad_(True)
##                t_batch2 = t_batch2.clone().detach().requires_grad_(True)
##                x_batch2 = x_batch2.clone().detach().requires_grad_(True)


##                Rd_pred, Rd_b2_pred = self.net_f(t_batch, x_batch, t_batch2, x_batch2)
##                Rd_pred, Rd_b2_pred = self.net_f(t_batch, x_batch)
                Rd_pred = self.net_f(t_batch, x_batch)
                

                if config.ang_con == 0:
                        loss_Rd_2 = torch.mean(Rd_pred**2, dim=0)    

###################################################################################################
##                        loss_j = torch.mean(loss_Rd_2)
                        t_values = t_values.clone().detach().requires_grad_(True)
                        r_values = r_values.clone().detach().requires_grad_(True)

                        dr  = config.R / Nr
                        Nt  = round(config.t_max/(config.t_bin/config.N_ang_con))
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
                        Nt  = round(config.t_max/(config.t_bin/config.N_ang_con))
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

##                        if config.SWITCH_LOSS == 0:
##                                loss_Rd_b2 = loss_Rd.clone()
##                        else:
##                                loss_Rd_b2 = torch.mean(Rd_b2_pred ** 2)
##                                loss      += torch.mean(loss_Rd_b2)

                elif config.LOSS_TYPE == 1:
                        #################
                        log_sigma_Rd_tensor = torch.tensor(config.log_sigma_Rd, device=device)
                        loss = 0.5 * torch.exp(-log_sigma_Rd_tensor) * torch.mean(loss_Rd_2)  + 0.5 * log_sigma_Rd_tensor
                        #################

                        loss_Rd_dot  = torch.cat([loss_Rd_dot,torch.exp(log_sigma_Rd_tensor).unsqueeze(0)], dim=0)
                        loss_Rd     = torch.mean(loss_Rd_2)

##                        if config.SWITCH_LOSS == 0:
##                                loss_Rd_b2   = loss_Rd.clone()
##                        else:
##                                loss_Rd_b2_2 = torch.mean(Rd_b2_pred ** 2, dim=0)
##
##                                #################
##                                log_sigma_Rd_b2_tensor = torch.tensor(config.log_sigma_Rd_b2, device=device)
##                                loss += 0.5 * torch.exp(-log_sigma_Rd_b2_tensor) * torch.mean(loss_Rd_b2_2)  + 0.5 * log_sigma_Rd_b2_tensor
##                                #################
##
##                                loss_Rd_dot  = torch.cat([loss_Rd_dot,torch.exp(log_sigma_Rd_b2_tensor).unsqueeze(0)], dim=0)
##                                loss_Rd_b2 = torch.mean(loss_Rd_b2_2)

                else:
                        #################
                        if config.LOSS_TYPE == 2:
                                loss = 0.5 * torch.exp(-self.log_sigma_Rd_uw) * torch.mean(loss_Rd_2)  + 0.5 * self.log_sigma_Rd_uw
                                loss_Rd_dot  = torch.cat([loss_Rd_dot,torch.exp(self.log_sigma_Rd_uw)], dim=0)
                        else:
##                                loss = torch.mean( 0.5 * torch.exp(-self.log_sigma_Rd_uw) * loss_Rd_2  + 0.5 * self.log_sigma_Rd_uw )
##                                loss_Rd_dot  = torch.cat([loss_Rd_dot,torch.exp(self.log_sigma_Rd_uw)], dim=0)

                                delta = math.log(100.0)            
                                alpha = self.log_sigma_Rd_uw
                                #min_alpha, _ = alpha[:5].min(dim=0, keepdim=True)
#                                max_alpha = min_alpha + delta
#                                alpha_clamped = torch.clamp(alpha, min=min_alpha, max=max_alpha)
                                max_alpha, _ = alpha[:5].max(dim=0, keepdim=True)
                                min_alpha = max_alpha - delta
                                alpha_clamped = torch.clamp(alpha, min=min_alpha)
                                loss = torch.mean( 0.5 * torch.exp(-alpha_clamped) * loss_Rd_2  + 0.5 * alpha_clamped )
                                loss_Rd_dot  = torch.cat([loss_Rd_dot,torch.exp(alpha),torch.exp(alpha_clamped)], dim=0)
                        #################

                        loss_Rd     = torch.mean(loss_Rd_2[:5])

##                        if config.SWITCH_LOSS == 0:
##                                loss_Rd_b2   = loss_Rd.clone()
##                        else:
##                                loss_Rd_b2_2 = torch.mean(Rd_b2_pred ** 2, dim=0)
##
##                                #################
##                                if config.LOSS_TYPE == 2:
##                                        loss += 0.5 * torch.exp(-self.log_sigma_Rd_b2_uw) * torch.mean(loss_Rd_b2_2)  + 0.5 * self.log_sigma_Rd_b2_uw
##                                        loss_Rd_dot  = torch.cat([loss_Rd_dot,loss_Rd_b2_2,torch.exp(self.log_sigma_Rd_b2_uw)], dim=0)
##                                else:
##                                        beta = self.log_sigma_Rd_b2_uw
##                                        beta_clamped = self.soft_clamp(beta, gamma, gamma + delta, window=5.0)
##
##                                        loss += torch.mean( 0.5 * torch.exp(-beta_clamped) * loss_Rd_b2_2 + 0.5 * beta_clamped )
##                                        loss_Rd_dot = torch.cat([loss_Rd_dot,loss_Rd_b2_2,torch.exp(beta),torch.exp(beta_clamped)], dim=0)
##                                #################             
##
##                                loss_Rd_b2 = torch.mean(loss_Rd_b2_2)

##                return loss, loss_Rd, loss_Rd_b2, loss_Rd_dot, loss_j
                return loss, loss_Rd, loss_Rd_dot, loss_j



    # 学習
        def train(self):


                # GradScalerのインスタンスを作成
                scaler = torch.cuda.amp.GradScaler()            
                file_name = f"training_log_{current_timestamp}.txt"
                with open(file_name, 'a') as log_file:


##### j conservation #####
                        Nt = round(config.t_max/(config.t_bin/config.N_ang_con))
                        delta_t = config.t_max / (Nt-1)
                        it = torch.arange(Nt-1, device=device, dtype=torch.float32)
                        t_values = (it + torch.rand(Nt-1, device=device)) * delta_t
                        t_values = torch.cat((t_values, torch.tensor([config.t_max], device=device)))
                        
                        Nr = 1000
                        dr = config.R / Nr
                        r_values = torch.linspace(0.5*dr, config.R-0.5*dr, Nr, device=device)
#####
                        check = 1
                        if config.EPOCHS > 100: check = config.EPOCHS//100

                        for epoch in range(config.EPOCHS): 

##                                for (t_batch, x_batch), (t_batch2, x_batch2) in zip(dataloader, dataloader2):
                                for t_batch, x_batch in dataloader:
                                                                        
                                        t_batch  = t_batch.to(device)
                                        x_batch  = x_batch.to(device)
##                                        t_batch2 = t_batch2.to(device)
##                                        x_batch2 = x_batch2.to(device)


                                        # モデルを訓練モードに設定
                                        self.dnn.train()           
                                        # 勾配のリセット
                                        self.optimizer.zero_grad()         
                                        # 混合精度での順伝播と損失計算
                                        with torch.cuda.amp.autocast():
                                            # 損失関数を計算（ミニバッチのデータを使用）
##                                            self.loss, self.loss_1, self.loss_2, self.loss_Rd_dot, self.loss_j = self.loss_func(t_batch,x_batch,t_batch2,x_batch2,t_values,r_values,Nr)
                                            self.loss, self.loss_1, self.loss_Rd_dot, self.loss_j = self.loss_func(t_batch,x_batch,t_values,r_values,Nr)
                                        # 損失に基づいて勾配を計算（勾配スケーリングを適用）
                                        scaler.scale(self.loss).backward()         
                                        # オプティマイザを使ってモデルのパラメータを更新
                                        scaler.step(self.optimizer)           
                                        # 更新
                                        scaler.update()            
                                # 10エポックごとに損失を表示

                                config.count_epoch = epoch
                                note_epoch = config.sum_epoch + config.count_epoch

                                if epoch % check ==0:
                                        print( 'Iter %d, Loss: %.5e, %.5e, %.5e'\
                                           % (epoch, self.loss.item()
                                             , self.loss_1.item(), self.loss_j.item()
                                             ))
                                        log_file.write(
                                            f'{note_epoch:.5e} '
                                            f'{self.loss.item():.5e} '
                                            f'{self.loss_1.item():.5e} '
                                            f'{self.loss_j.item():.5e} '
                                            f'{" ".join(["%.2e" % x for x in self.loss_Rd_dot.tolist()])}\n'
                                        )
                                        log_file.flush()







        def predict(self, t, x): # 入力された座標 x と時間 t に基づいて、モデルの予測を行います。

                t = torch.tensor(t, requires_grad=True).float().to(device)  # tをGPUに移動
                x = torch.tensor(x, requires_grad=True).float().to(device)  # xをGPUに移動

                # eval() 関数で評価モードに設定し、パラメータの更新が行われないようにします。
                self.dnn.eval()         
#                a_0, a_0_r, a, a_t, a_r, a_i, _, a_b, _, a_ib, *_ = self.net_a(t,x)
                a_0, a_0_r, a, a_t, a_r, a_i, a_i_r, a_b, a_b_t, a_ib, _, _ = self.net_a(t, x)

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
##                Rd, _ = self.net_f(t,x)
                Rd = self.net_f(t,x)

                ur_r    =   (a  [:,1:2]-a_i  [:,1:2]) -   (a_b  [:,1:2]-a_ib[:,1:2])\
                        + x*(a_r[:,1:2]-a_i_r[:,1:2])
                ur_t    = x * ( a_t[:,1:2] - a_b_t[:,1:2] )
                        
                uth_t =                x* a_t[:,2:3]
                uth_r = a_0_r[:,2:3] +   (a  [:,2:3]-a_i[:,2:3]) + x*(a_r[:,2:3]-a_i_r[:,2:3])

                Dur  = ut * ur_t  + ur * ur_r 
                Duth = ut * uth_t + ur * uth_r

                vorticity     = (ur*Duth-uth*Dur+uth_r+x*uth**3 + 2.0*uth/x)
                spinpotential = config.factor_s*ut*om/torch.sqrt(torch.abs(e))/x

                return e, ur, uth, om, phi, Ttt, orb, spin, Rd, vorticity, spinpotential



for file_number_dot in range(8, 9):
        config.SWITCH_LOSS2 = 1
        size_Rd = 7 + config.SWITCH_LOSS2

        i_length = 1 # 0: short, 1: middle, 2: long
        if file_number_dot==0: i_length = 0
        if file_number_dot==1: i_length = 0
        if file_number_dot==2: i_length = 1
        if file_number_dot==3: i_length = 1
        if file_number_dot==4: i_length = 2
        if file_number_dot==5: i_length = 2
        if file_number_dot==6: i_length = 2
        if file_number_dot==7: i_length = 2
        if file_number_dot==8: i_length = 2
        if file_number_dot==9: i_length = 2

        i_gamma  = 1 # 0: gamma=0, 1, gamma=2
        if file_number_dot==0: i_gamma = 1
        if file_number_dot==1: i_gamma = 1
        if file_number_dot==2: i_gamma = 1
        if file_number_dot==3: i_gamma = 1
        if file_number_dot==4: i_gamma = 0
        if file_number_dot==5: i_gamma = 0
        if file_number_dot==6: i_gamma = 1
        if file_number_dot==7: i_gamma = 1
        if file_number_dot==8: i_gamma = 0
        if file_number_dot==9: i_gamma = 0

        if i_length == 0: config.t_max2  = 0.2
        if i_length == 1: config.t_max2  = 0.4
        if i_length == 2: config.t_max2  = 0.5
        config.gamma   = i_gamma * 2.0
        config.tau_phi = config.gamma
        
        config.t_max   = 0.4
        if file_number_dot==0: config.t_max = 0.2
        if file_number_dot==1: config.t_max = 0.2
        if file_number_dot==2: config.t_max = 0.4
        if file_number_dot==3: config.t_max = 0.4
        if file_number_dot==4: config.t_max = 0.5
        if file_number_dot==5: config.t_max = 0.5
        if file_number_dot==6: config.t_max = 0.5
        if file_number_dot==7: config.t_max = 0.5
        if file_number_dot==8: config.t_max = 0.4
        if file_number_dot==9: config.t_max = 0.4

        i_LOSS_TYPE        = 3         # 0: Normal, 1: Uncertainty Weighting, 2: Uncertainty Weighting (ver2), 3: Uncertainty Weighting (ver3)

        i_SWITCH_LOSS2     = 1         # 0: w/o ang loss, 1: w/ ang loss
        i_ang_con          = 1         # 0: w/o conservation, 1:w/ conservation
        if file_number_dot==1: i_SWITCH_LOSS2 = 0
        if file_number_dot==1: i_ang_con      = 0

        i_SWITCH_INIT      = 0         # 0: orbital, 1: spin
        if file_number_dot==0: i_SWITCH_INIT = 0
        if file_number_dot==1: i_SWITCH_INIT = 0
        if file_number_dot==2: i_SWITCH_INIT = 0
        if file_number_dot==3: i_SWITCH_INIT = 1
        if file_number_dot==4: i_SWITCH_INIT = 0
        if file_number_dot==5: i_SWITCH_INIT = 1
        if file_number_dot==6: i_SWITCH_INIT = 0
        if file_number_dot==7: i_SWITCH_INIT = 1
        if file_number_dot==8: i_SWITCH_INIT = 0
        if file_number_dot==9: i_SWITCH_INIT = 1
        config.SWITCH_INIT = i_SWITCH_INIT

        file_number = 50 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==0: file_number = 50 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==1: file_number = 50 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==2: file_number = 50 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==3: file_number = 50 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==4: file_number = 25 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==5: file_number = 25 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==6: file_number = 75 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==7: file_number = 75 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==8: file_number = 25 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        if file_number_dot==9: file_number = 25 + 1000 * round(i_SWITCH_LOSS2 + 2*i_LOSS_TYPE + 8*i_ang_con + 16*i_SWITCH_INIT)
        
        pinns = torch.load(f'model_orbinit_ideal.pth', map_location=torch.device('cuda'))

        file_number = file_number+7310*00
        if file_number_dot==0: file_number = file_number+7310*100000 # short
        if file_number_dot==1: file_number = file_number+7310*100000 # short, w/o C.L.
        if file_number_dot==2: file_number = file_number+7311*100000 # middle
        if file_number_dot==3: file_number = file_number+7311*100000 # middle, spin init.
        if file_number_dot==4: file_number = file_number+7312*100000 # long, ideal
        if file_number_dot==5: file_number = file_number+7312*100000 # long, spin init., ideal
        if file_number_dot==6: file_number = file_number+7313*100000 # long
        if file_number_dot==7: file_number = file_number+7313*100000 # long, spin init.
        if file_number_dot==8: file_number = file_number+7314*100000 # long2, ideal
        if file_number_dot==9: file_number = file_number+7314*100000 # long2, spin init., ideal
        print(file_number_dot, file_number)

        fig_1 = 1 # Heatmap in t-r plane
        fig_2 = 0 # r-distribution
        fig_3 = 0 # Time Evolution of Energy and Total Angular Momentum
        fig_4 = 0 # Time Evolution of Total Angular Momentum for devided spatial region
        fig_5 = 0 # Evaluate Violation
        print_pdf = 1
        print_dat = 1

###################### 
# Heatmap in t-r plane
        if fig_1 == 1:
                print("Start: Heatmap in t-r plane")
######################
                # make spacetime points
                t_values = np.linspace(0.0 * config.t_max, config.t_max, 100)
                r_values = np.linspace(0.0001 * config.R, config.R, 400)
                t_grid, r_grid = np.meshgrid(t_values, r_values)
                t_current = t_grid.flatten().reshape(-1, 1)
                r_current = r_grid.flatten().reshape(-1, 1)


                # make hydrodynamic variables
                e, ur, uth, om, phi, _, orb, _, _, vor, pot = pinns.predict(t_current, r_current)

                # change data of  hydrodynamic variables for plot
                r_tensor = torch.tensor(r_current, device=phi.device, dtype=phi.dtype)
                phixy    = phi * r_tensor                
                vorxy    = vor * r_tensor
                potxy    = pot * r_tensor
                e_all     = e  [:].reshape(len(r_values), len(t_values))
                ur_all    = ur [:].reshape(len(r_values), len(t_values))
                uth_all   = uth[:].reshape(len(r_values), len(t_values))
                om_all    = om [:].reshape(len(r_values), len(t_values))
                phi_all   = phi[:].reshape(len(r_values), len(t_values))
                orb_all   = orb[:].reshape(len(r_values), len(t_values))
                vor_all   = vor[:].reshape(len(r_values), len(t_values))
                pot_all   = pot[:].reshape(len(r_values), len(t_values))
                phixy_all = phixy[:].reshape(len(r_values), len(t_values))
                vorxy_all = vorxy[:].reshape(len(r_values), len(t_values))
                potxy_all = potxy[:].reshape(len(r_values), len(t_values))
                e_all_cpu     = e_all.detach().cpu().numpy()
                ur_all_cpu    = ur_all.detach().cpu().numpy()
                uth_all_cpu   = uth_all.detach().cpu().numpy()
                om_all_cpu    = om_all.detach().cpu().numpy()
                phi_all_cpu   = phi_all.detach().cpu().numpy()
                orb_all_cpu   = orb_all.detach().cpu().numpy()
                vor_all_cpu   = vor_all.detach().cpu().numpy()
                pot_all_cpu   = pot_all.detach().cpu().numpy()
                phixy_all_cpu = phixy_all.detach().cpu().numpy()
                vorxy_all_cpu = vorxy_all.detach().cpu().numpy()
                potxy_all_cpu = potxy_all.detach().cpu().numpy()
                e_plot     = e_all_cpu.T
                ur_plot    = ur_all_cpu.T
                uth_plot   = uth_all_cpu.T
                om_plot    = om_all_cpu.T
                phi_plot   = phi_all_cpu.T
                orb_plot   = orb_all_cpu.T
                vor_plot   = vor_all_cpu.T
                pot_plot   = pot_all_cpu.T
                phixy_plot = phixy_all_cpu.T
                vorxy_plot = vorxy_all_cpu.T
                potxy_plot = potxy_all_cpu.T


                def plot_heatmap(
                    data, vmax_real, vmin_real, norm, file_name,
                    label_text, label_kwargs, r_values, t_values, print_pdf, tick_decimals, label_coords,
                    x_shift, tick_scientific, 
                ):
                    fig, ax = plt.subplots(figsize=(8,6))
                    im = ax.imshow(
                        data,
                        aspect='auto',
                        origin='lower',
                        extent=[r_values[0], r_values[-1], t_values[0], t_values[-1]],
                        cmap='viridis',
                        norm=norm,    
                    )
                    ax.set_box_aspect(1)

                    # カラーバーの設定
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=-0.5)
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.ax.tick_params(labelsize=15)

                    # ラベル設定の適用
                    cbar.set_label(label_text, **label_kwargs)
                    cbar.ax.yaxis.set_label_position('right')
                    cbar.ax.yaxis.set_label_coords(*label_coords)
                    cbar.ax.yaxis.get_offset_text().set_fontsize(15)

                    # 目盛り（Ticks）の設定
                    tick_labels = np.linspace(vmin_real, vmax_real, 6)
                    tick_positions = np.interp(tick_labels, [vmin_real, vmax_real], [vmin_real, vmax_real])

                    # 目盛りの適用
                    if tick_scientific:
                        format_string = f"{{v:.{tick_decimals}e}}"
                    else:
                        format_string = f"{{v:.{tick_decimals}f}}"
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels([format_string.format(v=v) for v in tick_labels])
                    cbar.ax.set_ylim(vmin_real, vmax_real)


                    # 軸ラベルの設定
                    ax.set_xlabel('r', fontsize=25)
                    ax.set_ylabel('t', fontsize=25, rotation=0, labelpad=20)
                    ax.tick_params(axis='both', labelsize=15)
                    fig.subplots_adjust(left=0.15)

                    original_left  = 0.15
                    original_right = 0.9
                    new_left       = original_left + x_shift
                    new_right      = original_right + x_shift
                    fig.subplots_adjust(left=new_left, right=new_right) # ★ leftとrightの両方を指定

                    # ファイルの保存
                    fig.savefig(file_name + ".png")
                    if print_pdf == 1:
                        fig.savefig(file_name + ".pdf")

                    plt.close(fig)

                # plot e
                data            = e_plot
                vmax_real       = np.max(data) # for value range
                vmin_real       = 0.990 # np.min(data) # for value range
                vmax_sym        = max(abs(vmin_real-1), abs(vmax_real-1))
                norm            = Normalize(vmin=1-vmax_sym, vmax=1+vmax_sym) # for color range
                label_text      = r"$e$"
                label_kwargs    = {"fontsize": 40,"rotation": 0,"labelpad": 20}
                label_coords    = (-2.5, 1.125)
                tick_decimals   = 3
                tick_scientific = False
                x_shift         = -0.05
                file_name       = f"e_heatmap_{file_number}" 
                plot_heatmap(data, vmax_real, vmin_real, norm, file_name, label_text, label_kwargs, r_values, t_values, print_pdf, tick_decimals, label_coords, x_shift, tick_scientific)
              # plot ur
                data            = ur_plot
                vmax_real       =  0.005 # np.max(data) # for color range
                vmin_real       =  0.000 # np.min(data) # for color range
                vmax_sym        = max(abs(vmin_real), abs(vmax_real))
                norm            = Normalize(vmin=-vmax_sym, vmax=vmax_sym)
                label_text      = r"$u^r$"
                label_kwargs    = {"fontsize": 40,"rotation": 0,"labelpad": 20}
                label_coords    = (-2.5, 1.125)
                tick_decimals   = 3
                tick_scientific = False
                x_shift         = -0.05
                file_name       = f"ur_heatmap_{file_number}" 
                plot_heatmap(data, vmax_real, vmin_real, norm, file_name, label_text, label_kwargs, r_values, t_values, print_pdf, tick_decimals, label_coords, x_shift, tick_scientific)
              # plot uth
                data            = uth_plot
                vmax_real       = 0.2 # np.max(data) # for color range
                vmin_real       = 0.0 # np.min(data) # for color range
                vmax_sym        = max(abs(vmin_real), abs(vmax_real))
                norm            = Normalize(vmin=-vmax_sym, vmax=vmax_sym)
                label_text      = r"$u^\theta$"
                label_kwargs    = {"fontsize": 40,"rotation": 0,"labelpad": 20}
                label_coords    = (-2.5, 1.125)
                tick_decimals   = 2
                tick_scientific = False
                x_shift         = -0.05
                file_name       = f"uth_heatmap_{file_number}" 
                plot_heatmap(data, vmax_real, vmin_real, norm, file_name, label_text, label_kwargs, r_values, t_values, print_pdf, tick_decimals, label_coords, x_shift, tick_scientific)
              # plot om
                data            = om_plot
                vmax_real       = np.max(data) # for color range
                vmin_real       = np.min(data) # for color range
                vmax_sym        = max(abs(vmin_real), abs(vmax_real))
                norm            = Normalize(vmin=vmin_real, vmax=vmax_real)
                label_text      = r"$S^z$"
                label_kwargs    = {"fontsize": 40,"rotation": 0,"labelpad": 20}
                label_coords    = (-2.5, 1.125)
                tick_decimals   = 0
                tick_scientific = True
                x_shift         = -0.05
                file_name       = f"om_heatmap_{file_number}" 
                plot_heatmap(data, vmax_real, vmin_real, norm, file_name, label_text, label_kwargs, r_values, t_values, print_pdf, tick_decimals, label_coords, x_shift, tick_scientific)
              # plot orb
                data            = orb_plot
                vmax_real       = np.max(data) # for color range
                vmin_real       = 0.00 # np.min(data) # for color range
                vmax_sym        = max(abs(vmin_real), abs(vmax_real))
                norm            = Normalize(vmin=vmin_real, vmax=vmax_real)
                label_text      = r"$L^z$"
                label_kwargs    = {"fontsize": 40,"rotation": 0,"labelpad": 20}
                label_coords    = (-2.5, 1.125)
                tick_decimals   = 2
                tick_scientific = False
                x_shift         = -0.05
                file_name       = f"orb_heatmap_{file_number}" 
                plot_heatmap(data, vmax_real, vmin_real, norm, file_name, label_text, label_kwargs, r_values, t_values, print_pdf, tick_decimals, label_coords, x_shift, tick_scientific)



