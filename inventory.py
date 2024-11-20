"""
    Inventory Management:
        X_p1 = (X + a - D - kX)^+
"""
import numpy as np
import matplotlib.pyplot as plt


def demmand():
    """
        Demand Model 
    """
    r = np.random.geometric(p = 0.05)
    return r

def dynamic(x, a, eta):
    """
        Dynamic of the Inventory Model
    """
    d = demmand()
    f = x + a - d - np.ceil(eta * x)
    f = max(f, 0)
    return f, d


def dynamic_manual(x, a, d, eta):
    """
        Dynamic Manual Calculus
    """
    f = x + a - d - np.ceil(eta * x)
    return f


def inventory_history(x_0, eta, k, period):
    """
        Inventory Simulation
    """
    inv_h = []
    act_h = []
    d_h = []
    x = x_0
    for _ in range(period):
        a = 10
        if x + a > k:
            a = k - x
        inv_h.append(x)
        act_h.append(a)
        xp, dem_d = dynamic(x, a, eta)
        d_h.append(dem_d)
        x = xp
    inv_h.append(x)
    return inv_h, act_h, d_h


def inv_samples(nums):
    """
        Inventory Model Samples
    """
    for _ in range(nums):
        inventory, _, _ = inventory_history(INIT_VALUE, ETA, MAX_CAP, NDAYS)
        plt.plot(inventory)
    plt.plot([0, NDAYS], [0,0])
    plt.plot([0, NDAYS], [MAX_CAP,MAX_CAP])
    plt.show()


ETA = 0.1
INIT_VALUE = 100
NDAYS = 20
MAX_CAP = 200
SAMPLES = 50

# inv_samples(SAMPLES)

inv, policy, dem = inventory_history(INIT_VALUE, ETA, MAX_CAP, NDAYS)

# print(inv)
# print(len(inv))


def ben_by_step(x, a, d, p_s, p_v):
    """
        Calculate Benefits by state
    """
    in_money = p_v * min(x + a, d)
    out_money = p_s * a
    g = in_money - out_money
    return g


def benefits(history, pol, dmd, p_s, p_v):
    """
        Calculate the benefits
    """
    prize = - history[0] * p_s
    ben = [prize]
    for k in range(1,len(history)):
        x_k = history[k]
        a_k = pol[k - 1]
        d_k = dmd[k - 1]
        ben.append(ben_by_step(x_k, a_k, d_k, p_s, p_v))
    return ben


bs = benefits(inv, policy, dem, p_s = 10, p_v = 10)
print(len(bs))


plt.plot(inv)
plt.plot(bs)
plt.show()
