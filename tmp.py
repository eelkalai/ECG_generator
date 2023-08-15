import ecg_plot
import numpy as np
import ECGImageGenerator
import ECG as ECG

row_height = 6
columns = 2
ecg = ECG.ECG()
ECGImageGenerator.plot(ecg.get_signals(),row_height=6, columns=[4,3,3,1,1],sample_rate = 500)
ECGImageGenerator.show()


