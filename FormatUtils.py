import glob
import json
import os
import cv2
import numpy as np

width, height = 1600, 720


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


def loadECGPoints(file_path):
    numbers_list = []
    with open(file_path, 'r') as file:
        # Read each line from the file and convert it to a number (integer or float)
        line = file.readline()
        numbers = line.strip().split()
        for num_str in numbers:
            number = float(num_str)  # Use float() if the numbers can have decimal points
            numbers_list.append(number)
    return numbers_list


def getECG(filename):
    # Load the image
    image = cv2.imread('./RealData/images/' + filename + '.jpg')
    points = np.array(loadECGPoints('./RealData/labels/' + filename + '.txt')[1:])
    original_points = points.reshape([4, 2]) * [image.shape[1], image.shape[0]]
    result_point = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    original_points = np.array(original_points, dtype=np.float32)

    M = cv2.getPerspectiveTransform(original_points, result_point)
    result_image = cv2.warpPerspective(image, M, (width, height))  # Specify the width and height of the new image

    # Save or display the result_image
    cv2.imwrite('./RealData/imagesStretched/' + filename + '.jpg', result_image)


folder_path = './RealData/images/'
# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    # Check if the item is a file (not a subdirectory)
    if os.path.isfile(os.path.join(folder_path, filename)):
        base_name = os.path.splitext(filename)[0]
        getECG(base_name)
