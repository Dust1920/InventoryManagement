"""
    Inventory Management:
        X_p1 = (X + a - D - kX)^+
"""

import numpy as np
import matplotlib.pyplot as plt


def demmand(p_k):
    """
        Demand Model
        p: Probabily of sell K products
    """
    r = np.random.geometric(p_k)
    return r


def l(x, pars):
    """
        Function 
    """
    eta = pars
    y = np.ceil(eta * x)
    return y


def dyn(x, a, model_p):
    """
        Dynamic of the Inventory Model
    """
    prob = model_p['p']
    eta = model_p['eta']
    d = demmand(prob)
    f_in = x + a  # In Vale
    lx = l(x, eta)
    f_out = d + lx
    return f_in, f_out, d


def reward_t(x, a, d, mpars):
    """
        Calculate Rewards in time t
    """
    p_v = mpars['sell']
    p_s = mpars["cost"]
    rew = p_v * min(x + a, d)
    cost = p_s * a
    reward = rew - cost
    return reward


def new_state(f):
    """
        Calculate the new state
    """
    return max(f,0)


def icm(ipars, pi, mpars):
    """
        Simulate Inventory Control Model 
    """
    # History Lists
    i_hist = [ipars['x_0']]
    d_hist = []
    rew_hist = [- mpars['cost'] * i_hist[0]]
    n_stages = ipars["T"]
    for n in range(n_stages):
        a = pi[n]
        fi, fo, d = dyn(i_hist[n],a, mpars)
        x = new_state(fi-fo)
        r = reward_t(x, a, d, mpars)
        rew_hist.append(r)
        i_hist.append(x)
        d_hist.append(d)
    rew_hist = np.array(rew_hist)
    return i_hist, d_hist, rew_hist


PVI = {"x_0": 100,
       "T": 50,
       "K": 200,
       "SAMPLES": 50}


MODEL_PARS = {"sell": 30,
              "cost": 20,
              "eta": 0.1,
              "p": 0.05}


inventory, inv_out, rt = icm(PVI, [0 for _ in range(PVI['T'])], MODEL_PARS)


fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (13, 8))
ax[0, 0].plot(rt)
ax[0, 0].set_xlabel("Stages")
ax[0, 0].set_ylabel("Reward")
ax[0, 1].plot(rt.cumsum())
ax[0, 1].set_xlabel("Stages")
ax[0, 1].set_ylabel("Accumulated Reward")
ax[1, 0].plot(inventory)
ax[1, 0].set_xlabel("Stages")
ax[1, 0].set_ylabel("Inventory Level")
plt.show()
