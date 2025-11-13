
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np  
from extract_char import *

# need to read images for testing from desktop folder 'test_images'
CAPTCHA_FOLDER_PATH = 'Datasets/full/val' # enter your folder path here
SEGMENTATION_FOLDER_PATH = 'Datasets/full/val/segmented' # enter your folder path here
max_num_img = 10000 # specify how many images to test

test_images_folder = os.path.expanduser(CAPTCHA_FOLDER_PATH)
test_images = [f for f in os.listdir(test_images_folder) if f.endswith(".png")][:max_num_img]

if SEGMENTATION_FOLDER_PATH:
    if not os.path.exists(SEGMENTATION_FOLDER_PATH):
        os.makedirs(SEGMENTATION_FOLDER_PATH)
    # Empty folder first
    for f in os.listdir(SEGMENTATION_FOLDER_PATH):
        os.remove(os.path.join(SEGMENTATION_FOLDER_PATH, f))

correct_count = 0
char_count = {}
for img_num, img_name in enumerate(test_images):
    print(f"Processing image {img_num + 1}/{len(test_images)}: {img_name}")
    # img_name = abc-0.png so we want only "abc"
    label = img_name.split('-')[0]
    
    img_path = os.path.join(test_images_folder, img_name)
    image = cv2.imread(img_path)
    image_cleaned, gray, processed = preprocess_image(image, visualize=False)

    boxes, vis = get_center_aligned_boxes(processed, gray, visualize=False, merge_thresh=1)
    segmentations = []
    if len(segmentations) < len(label):
        segmentations = handle_connected_characters(image_cleaned, boxes, visualize=False)
    elif len(segmentations) == len(label):
        segmentations = get_cropped_images(image_cleaned, boxes)
    else:
        print(f"Mismatch in letter count for image: {img_name}. Expected {len(label)}, Detected {len(segmentations)}")
        continue

    print(f"Detected {len(segmentations)} letters in image: {img_name}")

    if len(segmentations) == len(label):
        correct_count += 1
        
        if SEGMENTATION_FOLDER_PATH:        
            # Save segmented characters into SEGMENTATION_FOLDER_PATH
            for idx, segmentation in enumerate(segmentations):
                _, char_image = segmentation
                char_label = label[idx]
                char_count[char_label] = char_count.get(char_label, 0) + 1
                char_filename = f"{char_label}-{char_count[char_label]}.png"
                char_filepath = os.path.join(SEGMENTATION_FOLDER_PATH, char_filename)
                cv2.imwrite(char_filepath, char_image)
    else:
        print(f"Mismatch in letter count for image: {img_name}. Expected {len(label)}, Detected {len(segmentations)}")

print(f"Total correct detections: {correct_count} out of {len(test_images)}")
accuracy = correct_count / len(test_images) * 100
print(f"Accuracy: {accuracy:.2f}%")

    



