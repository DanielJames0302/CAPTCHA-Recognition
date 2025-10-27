import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(filename, visualize=True):
    image = cv2.imread(filename)

    if visualize:
        plt.imshow(image, 'gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.show()

    # remove nearly black pixels (likely noise lines)
    mask_nonblack = np.any(image > [50, 50, 50], axis=2)  # keep pixels not near black
    image_clean = image.copy()
    image_clean[~mask_nonblack] = [255, 255, 255]  # turn black areas white

    if visualize:
        plt.imshow(image_clean, 'gray')
        plt.title('Clean Image')
        plt.axis('off')
        plt.show()

    mask = np.all(image <= [50, 50, 50], axis=2).astype(np.uint8) * 255
    image_blend = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    if visualize:
        plt.imshow(image_blend, 'gray')
        plt.title('Recovered Color Image')
        plt.axis('off')
        plt.show()

    # preprocessing (grayscale, binarization)
    gray = cv2.cvtColor(image_blend, cv2.COLOR_BGR2GRAY) 

    if visualize:
        plt.imshow(gray, 'gray')
        plt.title('Gray Image')
        plt.axis('off')
        plt.show()


    # adaptive thresholding for varying illumination 
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,
        15, 10
    )
    
    processed = thresh
    

    if visualize:
        plt.imshow(processed, 'gray')
        plt.title('Processed for Segmentation')
        plt.axis('off')
        plt.show()

    return image, gray, processed


def get_center_aligned_boxes(processed, gray, height_scale=1.3, merge_thresh=0.5, visualize=False):
    """
    detects letters with consistent vertical alignment and similar box height.
    
    processed: binary image (white text on black background)
    gray: grayscale version of original image
    height_scale: multiplier for uniform box height
    merge_thresh: how close boxes must be to merge (for broken letters)
    visualize: if True, shows the detected boxes and crops
    """
    h_img, w_img = processed.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)

    boxes = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < 10:
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        print("no boxes found")
        return [], None

    # sort left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])
    # print(len(boxes))

    # merge horizontally close boxes
    merged = []
    for b in boxes:
        if not merged:
            merged.append(b)
            continue
        x, y, w, h = b
        px, py, pw, ph = merged[-1]
        if x - (px + pw) <= merge_thresh:
            nx = min(x, px)
            ny = min(y, py)
            nw = max(x + w, px + pw) - nx
            nh = max(y + h, py + ph) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(b)

    # old_boxes = boxes
    boxes = merged
    # print(len(boxes))

    # compute a global median centerline and uniform height
    centers_y = [y + h // 2 for (_, y, _, h) in boxes]
    median_center_y = int(np.median(centers_y))
    highest_center_y = np.min(centers_y)
    median_center_y = int((median_center_y + highest_center_y) / 2)
    max_h = np.max([h for (_, _, _, h) in boxes]) # need to take max height !!!
    uniform_h = int(max_h * height_scale)

    # recenter each box vertically
    centered_boxes = []
    for (x, y, w, h) in boxes:
        center_x = x + w // 2
        center_y = median_center_y
        new_y = max(0, center_y - uniform_h // 2)
        new_h = min(uniform_h, h_img - new_y)
        new_x = max(0, x)
        new_w = min(w, w_img - new_x)
        centered_boxes.append((new_x, new_y, new_w, new_h))

    # make the width larger if enough space
    # will look at the distance to neighboring boxes and expand a bit
    final_boxes = []
    for i, (x, y, w, h) in enumerate(centered_boxes):
        left_space = x if i == 0 else (x - (centered_boxes[i-1][0] + centered_boxes[i-1][2]))
        right_space = (w_img - (x + w)) if i == len(centered_boxes) - 1 else (centered_boxes[i+1][0] - (x + w))
        expand_left = min(left_space // 2, w // 2)
        expand_right = min(right_space // 2, w // 2)
        new_x = max(0, x - expand_left)
        new_w = min(w + expand_left + expand_right, w_img - new_x)
        final_boxes.append((new_x, y, new_w, h))

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(final_boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(vis, str(i+1), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if visualize:

        plt.figure(figsize=(10, 3))
        plt.imshow(vis[..., ::-1])
        plt.title("Gray Center-Aligned Fixed-Height Boxes")
        plt.axis("off")
        plt.show()
        
        vis2 = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(final_boxes):
            cv2.rectangle(vis2, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(vis2, str(i+1), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        plt.figure(figsize=(10, 3))
        plt.imshow(vis2[..., ::-1])
        plt.title("Processed Center-Aligned Fixed-Height Boxes")
        plt.axis("off")
        plt.show()

        # vis3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # for i, (x, y, w, h) in enumerate(old_boxes):
        #     cv2.rectangle(vis3, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #     cv2.putText(vis3, str(i+1), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # plt.figure(figsize=(10, 3))
        # plt.imshow(vis3[..., ::-1])
        # plt.title("Gray Original Boxes")
        # plt.axis("off")
        # plt.show()

        # vis4 = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        # for i, (x, y, w, h) in enumerate(old_boxes):
        #     cv2.rectangle(vis4, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #     cv2.putText(vis4, str(i+1), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # plt.figure(figsize=(10, 3))
        # plt.imshow(vis4[..., ::-1])
        # plt.title("Processed Original Boxes")
        # plt.axis("off")
        # plt.show()

        # # show cropped letters
        # for i, (x, y, w, h) in enumerate(final_boxes):
        #     crop = processed[y:y+h, x:x+w]
        #     plt.figure(figsize=(2, 2))
        #     plt.imshow(crop, cmap="gray")
        #     plt.title(f"Processed Letter {i+1}")
        #     plt.axis("off")
        #     plt.show()

        # for i,(x,y,w_box,h_box) in enumerate(final_boxes):
        #     crop = gray[y:y+h_box, x:x+w_box]
        #     plt.figure(figsize=(2,2))
        #     plt.imshow(crop, cmap='gray')
        #     plt.title(f'Gray Letter {i+1}')
        #     plt.axis('off')
        #     plt.show()


    return final_boxes, vis