"""
    Define the Control Inventory Model
"""


import numpy as np
import matplotlib.pyplot as plt
import inv_pars as ip


def demmand(p):
    """
        Define the Random demmand in the inventory model.
    """
    d = np.random.geometric(p)
    return d


def admisible_actions(x, k):
    """
        Calculate the admisible actions for each state X
    """
    v = list(range(k-x+1))
    return v


def reward_function(x, a, d, p_v, p_c):
    """
        Reward Function: Alternative Cost Function
    """
    lost_state = 1  # Activate or Deactivate additional cost.
    lost_cost = max(d - x - a, 0)
    c = a + lost_state * lost_cost
    r = p_v * min(x + a, d) - p_c * c
    return r


def dynamic(x, a, d, eta):
    """
        Calcuate the next state
    """
    xp1 = x + a - d - np.floor(eta * x)
    xp1 = max(xp1, 0)
    return xp1


def inventory(x_0, policy, p_c, p_v, eta):
    """
        Simulate Inventory model.
    """
    history = [x_0]
    rewards = [- p_c * x_0]
    d_h = []
    for t, a in enumerate(policy):
        d = demmand(ip.RV_P)
        d_h.append(d)
        x = history[t]
        x1 = dynamic(x, a, d, eta)
        x1 = int(x1)
        history.append(x1)
        rewards.append(reward_function(x1, a, d, p_v, p_c))
#        print(f"x_{t},a_{t},xi_{t}")
#        print((x1, a, d, p_v, p_c))
    return history, rewards, d_h


if __name__ == "__main__":
    test_policy = [50 if s % 3 == 0 else 0 for s in range(100)]
    X0 = 50
    H, R, XI = inventory(X0, test_policy,ip.REW_COST, ip.REW_SALE, ip.DYN_ETA)

    # print("History", H)
    # print("Policy", test_policy)
    # print("Rewards", R)
    # print("Demmand", XI)

    model, ax = plt.subplots(nrows=3, ncols = 1)
    ax[0].plot(H)
    ax[1].plot(R)
    cum_reward = np.array(R).cumsum()
    ax[2].plot(cum_reward)
    print("Recompensa Final", cum_reward[-1])
    plt.show()
