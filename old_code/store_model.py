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
    x, a, d_v = vector
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
    lost_cost = c_v * max(d - x - a, 0)
    reward_total = 0 * s_v * min(x + a, d) - c_v * a - u * lost_cost
    return reward_total


def admisible_actions(x, mpars):
    """
        Calculate the Admisible Actions Set A(x)
    """
    max_cap = mpars['K']
    adm_acts = list(range(max_cap - x + 1))
    return adm_acts


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
    sum_d = dv + dr
    return sum_d


#av, bv, u = simul_as(10, MODEL_P, REWARDS_P, d)
#s = av + bv

#print(s)
#print(len(s))
#print(s.max())
#print(s.argmax())
#v = s.argmax()
# print(u[v])


#av, bv, u = simul_as(30, MODEL_P, REWARDS_P, d)
#s = av + bv


MODEL_P = {
    "K": 200,
    "eta": 0.01
}


RANDOM_P = {
    "p": 0.05
}


REWARDS_P = {
    "sale value" : 30,
    "cost value": 20
}


X_0 = 200
SAMPLES = 5


data_hist, ax = plt.subplots(nrows = 2, ncols = 2)
dataset = [demmand(RANDOM_P) for _ in range(100)]

ax[0, 0].hist(dataset)  # Distribución de la muestra aleatoria 

for t in range(SAMPLES):
    print(t)
    history = [X_0]
    actions = []
    rewards = []
    for i in range(51):
        print("i", i)
        g = simul_as(history[i], MODEL_P, REWARDS_P, dataset)
        print("STAGE",i)
        print("STATE", history[i])
        print("Funcion W", g)
        print("Maximo W", g.max())
        a_op = g.argmax()
        d = demmand(RANDOM_P)
        x_1 = dynamic((history[i], a_op, d), MODEL_P)
        r = reward(((history[i], a_op, d)), REWARDS_P)
        rewards.append(r)
        x_1 = int(x_1)
        history.append(x_1)
        actions.append(a_op)
    rewards = np.array(rewards)
    ax[1, 0].plot(rewards)
    ax[1, 0].plot([0,50], [0,0])
    ax[1, 1].plot(rewards.cumsum())
    ax[0, 1].plot(history)
plt.show()


custom, ax = plt.subplots(nrows=2,ncols=2)
for t in range(SAMPLES):
    test_history = [X_0]
    test_actions = []
    test_rewards = []
    for t in range(50):
        required = demmand(RANDOM_P)
        ACTION = 0
        if t % 5 == 0:
            ACTION = min(70, MODEL_P['K'] - test_history[t])
        x = dynamic((test_history[t], ACTION, required), MODEL_P)
        r = reward((x,ACTION,required), REWARDS_P)
        test_history.append(x)
        test_actions.append(ACTION)
        test_rewards.append(r)

    ax[0, 0].plot(test_history)
    ax[0, 1].plot(test_actions)
    ax[1, 0].plot(test_rewards)
    ax[1, 1].plot(np.array(test_rewards).cumsum())


plt.show()
