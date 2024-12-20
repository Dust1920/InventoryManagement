"""
    S
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


RV_P = 0.05  # Parametro Demanda Aleatoria

REW_SALE = 20  # Precio de Venta
REW_COST = 10  # Costo Unitario
DYN_ETA = 0.1  # Factor de Pérdida
M_K = 100  # Capacidad máxima del inventario

X_0 = 60  # Inventario Inicial
PERIOD = 100 # Periodo Inicial


def demmand(p):
    """
        Define the Random demmand in the inventory model.
        Params:
            p (float): Probabiliy parameter.
        Return:
            d (float): Random value
    """
    d = np.random.geometric(p)
    return d


def admisible_actions(x, k):
    """
        Calculate the admisible actions for each state X
        Params:
            x (int): state value.
            k (int): max inventory capacity.
        Returns:
            v (list): admisible action set by value x
    """ 
    v = list(range(k-x+1))
    return v


def reward_function(x, a, d, p_v, p_c):
    """
        Reward Function: Alternative Cost Function
        Params:
            x (int): state value
            a (int): action value
            d (int): demmand value
            p_v (float): sale price
            p_c (float): unitary cost
        Return:
            r (float): Reward by x,a and d with parameters p_v and p_c
    """
    lost_state = 1  # Activate or Deactivate additional cost.
    lost_cost = max(d - x - a, 0)
    c = a + lost_state * lost_cost
    r = p_v * min(x + a, d) - p_c * c
    return r


def dynamic(x, a, d, eta):
    """
        Calcuate the next state
        Params:
            x (int): state value
            a (int): action value
            d (int): demmand value
            eta (float): loss parameter
        Return:
            xp1 (int): next state
    """
    xp1 = x + a - d - np.floor(eta * x)
    xp1 = max(xp1, 0)
    return xp1

def expected_reward(x, a, d_v):
    """
        Calculate the Expected Reward
        x (int): state value
        a (int): action value
        d_v (list): random demmand vector
        Return:
            data (list) reward demmand function (next to calculate mean)
    """  
    p_v = REW_SALE
    p_c = REW_COST
    data = [reward_function(x, a, d, p_v, p_c) for d in d_v]
    return data


def expected_vf(vf, x_0, a, d_v):
    """
        Calculate the function v
        Params:
            vf (function): Interpolation
            x_0 (int): state value
            a (int) : action value
            d_v (list): demmand random sample
        Return:
        vf_d (int): Expected vf function
    """
    vf_d = [float(vf(dynamic(x_0, a, d, DYN_ETA))) for d in d_v]
    vf_d = int(np.array(vf_d).mean())
    return vf_d

STATES_SET = list(range(M_K + 1))
def evol_vk(vk, hk, beta, der):
    """
        Calculus of v_k
        Params:
            vk (list): iterative function vk
            hk (list): iterative function hk
            beta (float):
            der (list):
        Return:
            vk (list): next function iterative hk
            hk (list): next function iterative vk
    """
    for u, x_k in enumerate(STATES_SET):
        a_x = admisible_actions(x_k, M_K)
    #     print("Admisible Actions",a_x)
        cost_ax = []
        expec_ax = []
        v_function = CubicSpline(STATES_SET, vk)
        for action in a_x:
            cost = [reward_function(x_k, action, d, REW_SALE, REW_COST) for d in der]
            cost = int(np.array(cost).mean())
            cost_ax.append(cost)
            e_vf = expected_vf(v_function, x_k, action, der)
            expec_ax.append(e_vf)
        v_cost = np.array(cost_ax)
        v_vf = np.array(expec_ax)
        v_sum = v_cost + beta * v_vf
        v_max = v_sum.max()
        vk[u] = v_max
        hk[u] = v_sum.argmax()
    return vk, hk


final_df = pd.DataFrame()

for ks in range(10):
    print(ks)
    demmand_vector = [demmand(RV_P) for _ in range(200)]
    v_0 = list(STATES_SET)  # v_0(x) = x
    h_0 = np.zeros(len(v_0))

    v_mem = [v_0]
    h_mem = [h_0]

    BPAR= 0.9
    for kt in range(25):
        v_1, h_1 = evol_vk(v_0, h_0, beta= BPAR, der = demmand_vector)
        v_mem.append(v_1)
        h_mem.append(h_1)
   #      print(f"{kt+1} - {kt}",np.array(v_1) - np.array(v_mem[kt]))
        v_0 = v_1.copy()
        h_0 = h_1.copy()

    test_eval = CubicSpline(STATES_SET, h_mem[-1])

    df = pd.DataFrame(index = list(range(PERIOD)),
                    columns=["state","action","demmand","r"])

    x_hist = [X_0]
    a_hist = []
    d_hist = []
    for q in range(PERIOD):
        a = int(test_eval(x_hist[q]))
        d = demmand(RV_P)
        r = reward_function(x_hist[q], a, d, REW_SALE, REW_COST)
        df.loc[q] = [x_hist[q], a, d, r]
        xp1 = dynamic(x_hist[q], a, d, DYN_ETA)
        x_hist.append(xp1)
        a_hist.append(a)
        d_hist.append(d)
    df['SAMPLE'] = ks
    if ks == 0:
        final_df = df
    else:
        final_df = pd.concat([final_df, df])


# Save data in DataFrame
final_df.to_excel(f"S_{PERIOD}_({M_K, DYN_ETA})_{RV_P}_({REW_SALE, REW_COST}).xlsx")
