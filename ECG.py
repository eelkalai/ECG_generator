import pickle

import numpy as np
import neurokit2 as nk  # Load the package

sample_rate = 500


class ECG:
    def __init__(self):
        self.heart_rate = np.random.randint(50, 120)
        ecg = nk.ecg_simulate(duration=8, sampling_rate=sample_rate, noise=0.1, heart_rate=self.heart_rate,
                              method="multileads")
        self.signals = np.transpose(ecg.to_numpy())
        self.names = list(ecg.columns)

    def get_signals(self):
        return self.signals

    def get_name(self, index):
        if index < len(self.names):
            return self.names[index]
        else:
            raise IndexError("Index out of range")

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
