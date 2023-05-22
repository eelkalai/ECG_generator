import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk  # Load the package
from ecg_plot import ecg_plot
from PIL import Image, ImageDraw

width, height = 1600, 720


def GetECG(sample_rate):
    heart_rate = np.random.randint(50, 120)
    print(heart_rate)
    ecg = nk.ecg_simulate(duration=8, sampling_rate=sample_rate, noise=0.1, heart_rate=heart_rate, method="multileads")
    return np.transpose(ecg.to_numpy())


def createLabel(x1, x2, y1, y2):
    center_x = (x1 + x2) / 2 / width
    center_y = (y1 + y2) / 2 / height
    width_rec = (x2 - x1) / width
    height_rec = (y2 - y1) / height
    label = f"0 {center_x:.6f} {center_y:.6f} {width_rec:.6f} {height_rec:.6f}"
    return label


def SaveTrainVal(index, train_or_val):
    scale_size = 40
    duration = 8
    sample_rate = 100
    row_height = 6
    columns = 2
    ecgs = GetECG(sample_rate=sample_rate)
    labels = ""
    y_max = 0
    y_min = 0
    for i, ecg in enumerate(ecgs):
        y_max = ecg.max()
        y_min = ecg.min()
        labels += createLabel(int(i/6)*800, (int(i/6)+1)* 800, scale_size * (1.5 + 3*(i % 6)-y_max),scale_size * (1.5 + 3*(i % 6)-y_min)) + "\n"
    ecg_plot.plot(ecgs, sample_rate=sample_rate, row_height=row_height, columns=columns, title='ECG 12')
    filename = 'train_data/images/'+train_or_val+f'/ECG_gen_{index + 1}.png'
    plt.savefig(filename)
    plt.close()
    filename = 'train_data/labels/'+train_or_val+f'/ECG_gen_{index + 1}.txt'
    with open(filename, 'w') as f:
        f.write(labels)


def drawOnData(x, y):
    img = Image.open("./train/square_4.png")
    draw = ImageDraw.Draw(img)
    draw.rectangle((800, x, 1600, y), fill=None, outline='black')
    img.show()
