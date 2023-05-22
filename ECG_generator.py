import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk  # Load the package
from ecg_plot import ecg_plot

def GetECG():
    heart_rate = np.random.randint(50, 120)
    print(heart_rate)
    ecgs = nk.ecg_simulate(duration=8, sampling_rate=500,noise=0.1, heart_rate=heart_rate,method="multileads")
    return np.transpose(ecgs.to_numpy())
    plt.savefig('./hh.png')
wGetECG()
GetECG()
GetECG()
GetECG()