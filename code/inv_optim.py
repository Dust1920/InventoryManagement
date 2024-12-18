"""
    Calculate the Optim Policy with Bellman Equation
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from code import inv_pars as ip
from code import inv_model as m


def expected_reward(x, a, d_v):
    """
        Calculate the Expected Reward
    """
    p_v = ip.REW_SALE
    p_c = ip.REW_COST
    data = [m.reward_function(x, a, d, p_v, p_c) for d in d_v]
    return data


def expected_vf(vf, x_0, a, d_v):
    """
        Calculate the function v
    """
    vf_d = [float(vf(m.dynamic(x_0, a, d, ip.DYN_ETA))) for d in d_v]
    vf_d = int(np.array(vf_d).mean())
    return vf_d

STATES_SET = list(range(ip.M_K + 1))
def evol_vk(vk, hk, beta, der):
    """
        Calculus of v_k
    """
    for u, x_k in enumerate(STATES_SET):
        a_x = m.admisible_actions(x_k, ip.M_K)
    #     print("Admisible Actions",a_x)
        cost_ax = []
        expec_ax = []
        v_function = CubicSpline(STATES_SET, vk)
        for action in a_x:
            cost = [m.reward_function(x_k, action, d,
                                    ip.REW_SALE,
                                    ip.REW_COST) for d in der]
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
    demmand_vector = [m.demmand(ip.RV_P) for _ in range(200)]
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

    df = pd.DataFrame(index = list(range(ip.PERIOD)),
                    columns=["state","action","demmand","r"])

    x_hist = [ip.X_0]
    a_hist = []
    d_hist = []
    for q in range(ip.PERIOD):
        a = int(test_eval(x_hist[q]))
        d = m.demmand(ip.RV_P)
        r = m.reward_function(x_hist[q], a, d, ip.REW_SALE, ip.REW_COST)
        df.loc[q] = [x_hist[q], a, d, r]
        xp1 = m.dynamic(x_hist[q], a, d, ip.DYN_ETA)
        x_hist.append(xp1)
        a_hist.append(a)
        d_hist.append(d)
    df['SAMPLE'] = ks
    if ks == 0:
        final_df = df
    else:
        final_df = pd.concat([final_df, df])



final_df.to_excel(f"S_{ip.PERIOD}_({ip.M_K,ip.DYN_ETA})_{ip.RV_P}_({ip.REW_SALE,ip.REW_COST}).xlsx")


"""

demmand_vector = [m.demmand(ip.RV_P) for _ in range(200)]
v_0 = list(STATES_SET)  # v_0(x) = x
h_0 = np.zeros(len(v_0))

v_mem = [v_0]
h_mem = [h_0]

BPAR= 0.9
for kt in range(25):
    v_1, h_1 = evol_vk(v_0, h_0, beta= BPAR, der = demmand_vector)
    v_mem.append(v_1)
    h_mem.append(h_1)
    print(f"{kt+1} - {kt}",np.array(v_1) - np.array(v_mem[kt]))
    v_0 = v_1.copy()
    h_0 = h_1.copy()

test_eval = CubicSpline(STATES_SET, h_mem[-1])

df = pd.DataFrame(index = list(range(ip.PERIOD)),
                  columns=["state","action","demmand","r"])

x_hist = [ip.X_0]
a_hist = []
d_hist = []
for q in range(ip.PERIOD):
    a = int(test_eval(x_hist[q]))
    d = m.demmand(ip.RV_P)
    r = m.reward_function(x_hist[q], a, d, ip.REW_SALE, ip.REW_COST)
    df.loc[q] = [x_hist[q], a, d, r]
    xp1 = m.dynamic(x_hist[q], a, d, ip.DYN_ETA)
    x_hist.append(xp1)
    a_hist.append(a)
    d_hist.append(d)

"""