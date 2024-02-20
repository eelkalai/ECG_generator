import glob
import json
import os
import cv2
import numpy as np

default_width, default_height = 1600, 720


def coco_segmentation_to_yolo_seg(coco_json_file, output_dir, class_mapping=None):
    # Load the COCO JSON file
    with open(coco_json_file, 'r') as coco_file:
        coco_data = json.load(coco_file)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each image in the COCO dataset
    for image_data in coco_data['images']:
        image_id = image_data['id']
        image_filename = image_data['file_name']

        # Create a YOLO-seg-style annotation file
        yolo_seg_annotation_file = os.path.join(output_dir, os.path.splitext(image_filename)[0] + '.txt')

        with open(yolo_seg_annotation_file, 'w') as yolo_file:
            # Iterate through the annotations for this image
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    category_id = annotation['category_id']

                    # If class mapping is provided, map the category_id to the YOLO class ID
                    if class_mapping is not None and category_id in class_mapping:
                        category_id = class_mapping[category_id]

                    # Get the segmentation mask (points)
                    segmentation = annotation['segmentation'][0]  # Assuming single polygon per annotation
                    # if len(segmentation) > 8:
                    #    yolo_file.close()
                    #    os.remove(output_dir+os.path.splitext(image_filename)[0] + '.txt')
                    #    continue
                    normalized_seg = []
                    for i, num in enumerate(segmentation):
                        if i % 2 == 0:
                            normalized_seg.append(num / image_data['width'])  # Even index
                        else:
                            normalized_seg.append(num / image_data['height'])  # Even index
                    # Convert the points to YOLO-seg format (X Y X Y ...)
                    yolo_seg_format = ' '.join(map(str, normalized_seg))

                    # Write the YOLO-seg annotation line to the file
                    yolo_file.write(f"{category_id} {yolo_seg_format}\n")


def renameFiles():
    # Define the folder path where your images are located
    folder_path = './RealData/images'

    # Define the new base name for the renamed images
    new_base_name = 'EGC_Image'

    # Get a list of all image files in the folder
    image_files = glob.glob(os.path.join(folder_path, '*'))  # Change the extension to match your image files

    # Loop through the image files and rename them
    for index, old_file_path in enumerate(image_files):
        index = index + 87
        # Construct the new file name
        file_extension = os.path.splitext(old_file_path)[1]  # Get the file extension
        new_file_name = f'{new_base_name}{index + 1}{file_extension}'  # Use index + 1 for a 1-based numbering

        # Generate the new file path
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)

        print(f'Renamed: {old_file_path} -> {new_file_path}')


def loadECGPoints(file_path: object) -> object:
    numbers_list = []
    with open(file_path, 'r') as file:
        # Read each line from the file and convert it to a number (integer or float)
        line = file.readline()
        numbers = line.strip().split()
        for num_str in numbers:
            number = float(num_str)  # Use float() if the numbers can have decimal points
            numbers_list.append(number)
    return numbers_list


def rearrangeLabels(file_path):
    points = np.array(loadECGPoints(file_path)[1:])
    points = points.reshape([4, 2])
    sorted_points = points[np.argsort(points[:, 1])]
    # Split the sorted points into two pairs: top and bottom
    top_points = sorted_points[:2]
    bottom_points = sorted_points[2:]
    top_points = top_points[np.argsort(top_points[:, 0])]
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    arranged_points = np.vstack((top_points[0], top_points[1], bottom_points[1], bottom_points[0]))
    saveLabel(arranged_points, file_path)


def saveLabel(arranged_points, file_path):
    formatted_line = " ".join([f"{x} {y}" for x, y in arranged_points])
    with open(file_path, 'w') as file:
        file.write("0 " + formatted_line)


def extractECG(filename):
    # Load the image
    image = cv2.imread('./RealData/images/' + filename + '.jpg')
    points = np.array(loadECGPoints('./RealData/labels/' + filename + '.txt')[1:])
    original_points = points.reshape([4, 2]) * [image.shape[1], image.shape[0]]
    result_point = np.array([[0, 0], [default_width, 0], [default_width, default_height], [0, default_height]],
                            dtype=np.float32)
    original_points = np.array(original_points, dtype=np.float32)
    M = cv2.getPerspectiveTransform(original_points, result_point)
    result_image = cv2.warpPerspective(image, M,
                                       (default_width, default_height))  # Specify the width and height of the new image
    # Save or display the result_image
    cv2.imwrite('./RealData/imagesStretched/' + filename + '.jpg', result_image)


def createDataSet():
    folder_path = './RealData/labels/'
    i = 1
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            for filename2 in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, filename)):
                    old = os.path.splitext(filename)[0]
                    new = os.path.splitext(filename2)[0]
                    if i % 10 == 0 and i % 20 != 0:
                        insertImageIntoTemplate(old, new, 'test/ECG' + str(i))
                    elif i % 20 == 0:
                        insertImageIntoTemplate(old, new, 'val/ECG' + str(i))
                    else:
                        insertImageIntoTemplate(old, new, 'train/ECG' + str(i))
                    i += 1


def insertImageIntoTemplate(old, new, result_file):
    original_image = cv2.imread('./RealData/images/' + old + '.jpg')
    pasted_image = cv2.imread('./RealData/imagesStretched/' + new + '.jpg')
    points = np.array(loadECGPoints('./RealData/labels/' + old + '.txt')[1:])
    original_points = np.array(points.reshape([4, 2]) * [original_image.shape[1], original_image.shape[0]],
                               dtype=np.int32)
    pasted_points = np.array([(0, 0), (pasted_image.shape[1], 0),
                              (pasted_image.shape[1], pasted_image.shape[0]),
                              (0, pasted_image.shape[0])], dtype=np.int32)
    matrix = cv2.getPerspectiveTransform(np.float32(pasted_points), np.float32(original_points))
    result_image = cv2.warpPerspective(pasted_image, matrix, (original_image.shape[1], original_image.shape[0]))
    # Create a mask for blending with smoother transitions
    mask = np.zeros(original_image.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(original_points), (1, 1, 1))
    # Apply Gaussian blur to the mask for a smoother transition
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    # Normalize the blurred mask to values between 0 and 1
    blurred_mask = blurred_mask
    result = original_image * (1 - blurred_mask) + result_image * blurred_mask
    # Display the result
    cv2.imwrite('./train_data/images/' + result_file + '.jpg', result)
    original_points = original_points/[original_image.shape[1], original_image.shape[0]]
    saveLabel(original_points, './train_data/labels/' + result_file + '.txt')


def rescaleImages():
    # Input and output directories
    input_folder = 'RealData/images/'
    # Scale factor for resizing (adjust as needed)
    scale_factor = 0.5  # Change this to the desired scaling factor

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg')):  # Add more extensions if needed
            # Read the image
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)
        if image is not None:
            # Get the dimensions of the original image
            height, width = image.shape[:2]
            scale_factor = default_width / width

            # Calculate the new dimensions while maintaining aspect ratio
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            # Resize the image using the calculated dimensions
            resized_image = cv2.resize(image, (new_width, new_height))
            # Create the output file path
            output_path = os.path.join(input_folder, filename)

            # Save the resized image
            cv2.imwrite(output_path, resized_image)

            print(f'Saved: {output_path}')


# folder_path = './RealData/labels/'
# for filename in os.listdir(folder_path):
#     # Check if the item is a file (not a subdirectory)
#     if os.path.isfile(os.path.join(folder_path, filename)):
#         base_name = os.path.splitext(filename)[0]
#         # rearrangeLabels(folder_path+filename)
#         extractECG(base_name)
