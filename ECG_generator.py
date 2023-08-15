import cv2
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk  # Load the package
import PaperSimulator as ps
from ecg_plot import ecg_plot
from PIL import Image, ImageDraw

from Utils import load_random_image, convert_to_homogeneous, convert_from_homogeneous

width, height = 1600, 720


def GetECG(sample_rate):
    heart_rate = np.random.randint(50, 120)
    ecg = nk.ecg_simulate(duration=8, sampling_rate=sample_rate, noise=0.1, heart_rate=heart_rate, method="multileads")
    return np.transpose(ecg.to_numpy())



def createPolyLabel(points, type,image):
    height, width, _ = image.shape
    label = f"{type}"
    for x, y in points:
        x = x / width
        y = y / height
        label += f" {x:.6f} {y:.6f}"
    label += "\n"
    return label


def SaveTrainVal(index, train_or_val):
    scale_size = 40
    duration = 8
    sample_rate = 500
    row_height = 6
    columns = 2
    ecgs = GetECG(sample_rate=sample_rate)
    ecg_plot.plot(ecgs, sample_rate=sample_rate, row_height=row_height, columns=columns, title='ECG 12')
    filename = 'train_data2/images/' + train_or_val + f'/ECG_gen_{index + 1}.png'
    plt.savefig(filename)
    plt.close()
    image = cv2.imread(filename)
    background = load_random_image('./train2014')
    perspective_image, transformation = ps.generate_random_perspective(ps.light_spot(ps.blur(image)), background)
    perspective_image = ps.blur(perspective_image)
    height, width, _ = image.shape
    corners = np.array([[0, 0], [width, 0],[width, height], [0, height]])
    new_corners = convert_from_homogeneous((transformation@(convert_to_homogeneous(corners).transpose())).transpose())
    labels = ""
    # for i, ecg in enumerate(ecgs):
    #     x = np.linspace(0, (sample_rate * duration - 1) / sample_rate, sample_rate * duration)*100 + int(i / 6) * width/2
    #     y = scale_size * (1.5 + 3 * (i % 6) - ecg)
    #     points = np.stack((x, y), axis=-1)
    #     points = np.append(points+1, np.flip(points,0)-1, axis=0)
    #     labels += createPolyLabel(points, 0)
    #     box = np.array([[np.amax(x),np.amax(y)],[np.amax(x),np.amin(y)],[np.amin(x),np.amin(y)],[np.amin(x),np.amax(y)]])
    #     labels += createPolyLabel(box, 1)
    # page_points = np.array([[width-1, height-1],[width-1, 1],[1,1],[1, height-1]])
    labels += createPolyLabel(new_corners, 0,perspective_image)
    cv2.imwrite(filename,perspective_image)
    filename = 'train_data2/labels/' + train_or_val + f'/ECG_gen_{index + 1}.txt'
    with open(filename, 'w') as f:
        f.write(labels)


