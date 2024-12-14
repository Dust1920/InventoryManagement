"""
    Simulation of the Inventory Model    
"""


import numpy as np
import matplotlib.pyplot as plt


def demmand(rv_pars):
    """
        Construction of the Demmand as Random Variable. 
    """
    p = rv_pars['p']
    y = np.random.geometric(p)
    return y


def dynamic(vector, mpars):
    """
        Dynamics of the Model
    """
    eta = mpars['eta']
    x,a, d_v = vector
    y = x + a - d_v - np.floor(eta * x)
    y = max(y,0)
    return y


def reward(vector, rpars):
    """
        Stage Reward
    """
    c_v = rpars['cost value']
    s_v = rpars['sale value']
    x, a, d = vector
    u = 1
    c = c_v * max(d - x - a, 0)
    r = s_v * min(x + a, d) - c_v * a - u * c
    return r


def admisible_actions(x, mpars):
    """
        Calculate the Admisible Actions Set A(x)
    """
    max_cap = mpars['K']
    return list(range(max_cap-x + 1))





MODEL_P = {
    "K": 200,
    "eta": 0.1
}

RANDOM_P = {
    "p": 0.05
}


REWARDS_P = {
    "sale value" : 30,
    "cost value": 20
}


XS = 10
d = [demmand(RANDOM_P) for _ in range(100)]
a_x = admisible_actions(XS, MODEL_P)


def rv_reward(x, a, rpars, random_v):
    """
        Calculate Random Rewards
    """
    rs = np.array([reward((x, a, d), rpars) for d in random_v])
    rs = np.floor(rs.mean())
    return rs


def rv_dynamic(x, a, mpars, random_v):
    """
        Calculate the Function Rewards
    """
    rd = np.array([dynamic((x, a, d), mpars) for d in random_v])
    rd = np.floor(rd.mean())
    return rd


def simul_as(x, mpars, rpars, random_v):
    """
        Calculate
    """
    ad_acts = admisible_actions(x, mpars)
    dv = np.array([rv_dynamic(x, a, mpars, random_v) for a in ad_acts])
    dr = np.array([rv_reward(x, a, rpars, random_v) for a in ad_acts])
    return dv, dr, ad_acts


#av, bv, u = simul_as(10, MODEL_P, REWARDS_P, d)
#s = av + bv

#print(s)
#print(len(s))
#print(s.max())
#print(s.argmax())
#v = s.argmax()
# print(u[v])

X_0 = 150

#av, bv, u = simul_as(30, MODEL_P, REWARDS_P, d)
#s = av + bv





dataset = [demmand(RANDOM_P) for _ in range(100)]
SAMPLES = 1

data_hist, ax = plt.subplots()
ax.hist(dataset)

samples_hist, axs = plt.subplots()

for t in range(SAMPLES):
    print(t)
    history = [X_0]
    actions = []
    for i in range(100):
        av, bv, u = simul_as(history[i], MODEL_P, REWARDS_P, dataset)
        s = av + bv
        a_op = s.argmax()
        d = demmand(RANDOM_P)
        x_1 = dynamic((history[i], a_op, d), MODEL_P)
        x_1 = int(x_1)
        history.append(x_1)
        actions.append(a_op)
    axs.plot(history)

plt.show()
