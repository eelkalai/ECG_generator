import os
import random
import cv2
import numpy as np


def load_random_image(directory):
    # Get the list of image files in the directory
    image_files = [filename for filename in os.listdir(directory) if filename.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError("No image files found in the directory")
    # Select a random image file
    random_image_file = random.choice(image_files)
    # Load and return the random image
    image_path = os.path.join(directory, random_image_file)
    image = cv2.imread(image_path)
    return image

def convert_to_homogeneous(points):
    num_points = len(points)
    homogeneous_points = np.ones((num_points,3))
    homogeneous_points[:, :2] = points
    return homogeneous_points

def convert_from_homogeneous(homogeneous_points):
    points = homogeneous_points[:, :2] / homogeneous_points[:, 2][:, np.newaxis]
    return points