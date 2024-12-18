"""
    Graphics for the Inventory Control Model
"""

# Librerias Requeridas
import pandas as pd
import matplotlib.pyplot as plt

# Leemos los datoa generados.
data = pd.read_excel("S_100_((100, 0.1))_0.05_((20, 10)).xlsx", index_col=0)
print(data)

# Hacemos las graficas solcitadas.
samples = data['SAMPLE'].unique()
figure_x, ax0 = plt.subplots(figsize = (6, 4))  # Grafica de Nivel de Inventario.
figure_r, ax1 = plt.subplots(figsize = (6, 4))  # Grafica de la Recompensa por etapa
figure_acr, ax2 = plt.subplots(figsize = (6, 4))  # Gráfica de la Recompensa acumulada. 
figure_act, ax3 = plt.subplots(figsize = (6, 4))  # Gráfica de las politicas. 
x = 0
r = 0
acr = 0
for w, s in enumerate(samples[:3]):
    data_s = data[data['SAMPLE'] == s]
    ax0.plot(data_s['state'], lw = 0.7, label = f"l_{s}")
    ax0.set_xlabel("Stages")
    ax0.set_ylabel("Level")
    ax1.plot(data_s['r'], lw = 0.7, label = f"l_{s}")
    ax1.set_ylabel("Reward")
    ax1.set_xlabel("Stages")
    ax2.plot(data_s['r'].cumsum(), lw = 0.7, label = f"l_{s}")
    ax2.set_xlabel("Stages")
    ax2.set_ylabel("Acc. Reward")
    ax3.plot(data_s['action'], lw = 0.7, label = f"l_{s}")
    ax3.set_xlabel("Stages")
    ax3.set_ylabel("Action")
    if w == 0:
        x = data_s['state']
        r = data_s['r']
        acr = data_s['r'].cumsum()
    else:
        x += data_s['state']
        r += data_s['r']
        acr += data_s['r'].cumsum()
ax0.plot(x / 3, lw = 0.7, color = "black", label = "x mean")    
ax1.plot(r / 3, lw = 0.7, color = "black", label = "r mean")
ax2.plot(acr / 3, lw = 0.7, color = "black", label = "acc. r mean")
figure_x.legend()
figure_x.tight_layout()
figure_x.savefig("Data_x.png")
figure_r.legend()
figure_r.tight_layout()
figure_r.savefig("Data_r.png")
figure_acr.legend(loc = "upper center")
figure_acr.tight_layout()
figure_acr.savefig("Data_acr.png")
figure_act.legend()
figure_act.tight_layout()
figure_act.savefig("Data_act.png")
plt.show()
