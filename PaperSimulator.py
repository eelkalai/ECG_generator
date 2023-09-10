import cv2
import numpy as np


def blur(image):
    mean = 0
    stddev = 30
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    blurred_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)
    return blurred_image


def generate_random_perspective(image, background):
    # Randomly generate perspective transformation points within bounds
    height, width, _ = image.shape
    max_height, max_width, _ = background.shape
    max_offset = 0.49  # Maximum offset as a fraction of image dimensions

    random_pts = np.float32([[0, 0],
                             [width, 0],
                             [0, height],
                             [width, height]])

    destination_pts = np.float32([[np.random.uniform(0, max_offset) * max_width, np.random.uniform(0, max_offset) * max_height],
                                  [np.random.uniform(1 - max_offset, 1) * max_width, np.random.uniform(0, max_offset) * max_height],
                                  [np.random.uniform(0, max_offset) * max_width, np.random.uniform(1 - max_offset, 1) * max_height],
                                  [np.random.uniform(1 - max_offset, 1) * max_width, np.random.uniform(1 - max_offset, 1) * max_height]])

    transformation_matrix = cv2.getPerspectiveTransform(random_pts, destination_pts)

    # Apply the perspective transformation to the image
    transformed_image = cv2.warpPerspective(image, transformation_matrix,(background.shape[1], background.shape[0]))

    # Overlay the transformed image onto the background
    mask = np.where(transformed_image != 0)
    background[mask] = transformed_image[mask]

    return background , transformation_matrix

def light_spot(image):
    center_x = np.random.randint(0, 1600)  # X-coordinate of the light source center
    center_y = np.random.randint(0, 720)  # Y-coordinate of the light source center
    radius = np.random.randint(10, 50)  # Radius of the light source
    brightness = np.random.randint(50, 1000)  # Brightness level of the light source (adjust as desired)
    alpha = 0.1 + np.random.random() * 0.5  # Opacity of the glow effect (adjust as desired)
    blur_radius = np.random.randint(5, 25) * 2 + 1  # Blur radius for the glow effect (should be odd)
    # Create a blank mask with the same dimensions as the image
    mask = np.zeros_like(image, dtype=np.uint8)
    # Draw a circular shape on the mask
    cv2.circle(mask, (center_x, center_y), radius, (brightness, brightness, brightness), -1)
    # Apply a Gaussian blur to the mask
    blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    # Combine the blurred mask with the original image using an alpha blend
    output_image = cv2.addWeighted(image, 1 - alpha, blurred_mask, alpha, 0)
    return output_image

def add_padding(image):
    top_pad = 20
    bottom_pad = 20
    left_pad = 20
    right_pad = 20
    color =[256, 256 ,256]
    # padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
    #                                   value=0)
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                      value=color)
    return padded_image


def get_affine_matrix(scale=(1, 1), rotation=0, translation=(0, 0), shear=(0, 0)):
    # Convert rotation angle to radians
    rotation = np.radians(rotation)

    # Decompose the transformation into individual matrices
    scale_matrix = np.array([[scale[0], 0], [0, scale[1]]], dtype=np.float32)
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
                               dtype=np.float32)
    translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]]], dtype=np.float32)
    shear_matrix = np.array([[1, shear[0]], [shear[1], 1]], dtype=np.float32)

    # Combine the individual transformations
    affine_matrix = translation_matrix @ shear_matrix @ rotation_matrix @ scale_matrix

    return affine_matrix