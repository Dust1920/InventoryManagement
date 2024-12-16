"""
    Calculate the Optim Policy with Bellman Equation
"""
import numpy as np
import pandas as pd
import inv_pars as ip
import inv_model as model


# v_0 sería la matriz cero
v_0 = np.zeros((ip.M_K,ip.M_K))
bellman = v_0


# v_1
for i in range(ip.M_K):
    bellman[i, :] = model.reward_function(0, i, 0, ip.REW_SALE, ip.REW_COST)


ACTION_SET = list(range(ip.M_K + 1))
STATES_SET = list(range(ip.M_K + 1))
RSAMPLES = 3


data = pd.DataFrame(index = list(range((ip.M_K + 1) * (ip.M_K + 1))),
    columns=['State','Action'] + [f"r_{s}" for s in range(RSAMPLES)])
demmand = data.copy()


S = 0
for state in STATES_SET:
    for action in ACTION_SET:
        data.loc[S, ['State','Action']] = (state, action)
        demmand.loc[S, ['State','Action']] = (state, action)
        demmand_vector = []
        for _ in range(RSAMPLES):
            d = model.demmand(ip.RV_P)
            demmand_vector.append(d)
        data.loc[S, [f"r_{S}" for S in range(RSAMPLES)]] = [
            model.reward_function(state,
                                  action,
                                  dem_v,
                                  ip.REW_SALE,
                                  ip.REW_COST) for dem_v in demmand_vector]
        demmand.loc[S, [f"r_{S}" for S in range(RSAMPLES)]] = demmand_vector
        S += 1
print(data)
print(demmand)
