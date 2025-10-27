
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np  
from extract_char import preprocess_image, get_center_aligned_boxes

# need to read images for testing from desktop folder 'test_images'
FOLDER_PATH = '' # enter your folder path here
max_num_img = 200 # specify how many images to test

test_images_folder = os.path.expanduser(FOLDER_PATH)
test_images = [f for f in os.listdir(test_images_folder) if f.endswith(".png")][:max_num_img]

correct_count = 0
for img_name in test_images:
    print(f"Processing image: {img_name}")
    img_path = os.path.join(test_images_folder, img_name)
    image, gray, processed = preprocess_image(img_path, visualize=False)

    boxes, vis = get_center_aligned_boxes(processed, gray, visualize=False)
    print(f"Detected {len(boxes)} letters in image: {img_name}")

    # img_name = abc-0.png so we want only "abc"
    if len(boxes) == len(img_name.split('-')[0]):
        correct_count += 1

print(f"Total correct detections: {correct_count} out of {len(test_images)}")
accuracy = correct_count / len(test_images) * 100
print(f"Accuracy: {accuracy:.2f}%")

    



