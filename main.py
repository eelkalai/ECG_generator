import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random

from ecg_plot import ecg_plot

import ECG_generator as ecgen


def createLabel(x1, x2, y1, y2, pagesize):
    center_x = (x1 + x2) / 2 / pagesize
    center_y = (y1 + y2) / 2 / pagesize
    width_rec = (x2 - x1) / pagesize
    height_rec = (y2 - y1) / pagesize
    label = f"0 {center_x:.6f} {center_y:.6f} {width_rec:.6f} {height_rec:.6f}"
    return label


# Set the size of the square images
width, height = 1600, 720

# Set the number of images to generate
num_images = 100


def Rectangle():
    for i in range(num_images):
        # Create a new image with a white background
        img = Image.new('RGB', (width, height), color='white')
        existing_rectangles = []
        labels = ""
        ecg =[]
        ecg_plot.plot(ecg, sample_rate=500, title='ECG 12')

        # Generate a random size and position for the square
        for j in range(10):
            print(j)
            x_size = random.randint(50, 200)
            y_size = random.randint(50, 200)
            x = random.randint(0, width - x_size)
            y = random.randint(0, height - y_size)
            overlap = False
            for rectangle in existing_rectangles:
                if rectangle[0] < x + x_size and rectangle[1] > x and rectangle[2] < y + y_size and rectangle[3] > y:
                    overlap = True
            if overlap:
                continue
            labels += createLabel(x, x + x_size, y, y + y_size, width) + "\n"
            existing_rectangles.append([x, x + x_size, y, y + y_size])
            draw = ImageDraw.Draw(img)
            draw.rectangle((x, y, x + x_size, y + y_size), fill=None, outline='black')

        # Save the image
        filename = f'train/square_{i + 1}.png'
        img.save(filename)
        filename = f'val/square_{i + 1}.txt'
        with open(filename, 'w') as f:
            f.write(labels)


def main():
    for i in range(100):
        ecgen.SaveTrainVal(i,'train')
    for i in range(10):
        ecgen.SaveTrainVal(i, 'val')

if __name__ == "__main__":
    main()

