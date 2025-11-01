import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

"""Enhance light colors in the image to make them more pronounced."""
def enhance_light_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # decrease v for light colors
    light_mask = (v > 200) & (v < 255) 
    v[light_mask] = 200
    hsv_enhanced = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    return enhanced_image

def preprocess_image(image, visualize=True):
    image_clean = image.copy()

    # make light colors more pronounced
    image_clean = enhance_light_colors(image_clean)

    # remove nearly black pixels (likely noise lines)
    mask_nonblack = np.any(image_clean > [50, 50, 50], axis=2)  # keep pixels not near black
    image_clean[~mask_nonblack] = [255, 255, 255]  # turn black areas white

    mask = np.all( image_clean <= [50, 50, 50], axis=2).astype(np.uint8) * 255
    image_blend = cv2.inpaint(image_clean, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # thicken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    image_blend = cv2.morphologyEx(image_blend, cv2.MORPH_ERODE, kernel)

    # preprocessing (grayscale, binarization)
    gray = cv2.cvtColor(image_blend, cv2.COLOR_BGR2GRAY)

    # adaptive thresholding for varying illumination 
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,
        15, 10
    )
    
    processed = thresh

    # Show all images in one figure from top to bottom
    if visualize:
        plt.figure(figsize=(8, 12))

        plt.subplot(4, 1, 1)
        plt.imshow(image, 'gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(4, 1, 2)
        plt.imshow(image_clean, 'gray')
        plt.title('Clean Image')
        plt.axis('off')

        plt.subplot(4, 1, 3)
        plt.imshow(gray, 'gray')
        plt.title('Grayscale Image')
        plt.axis('off')

        plt.subplot(4, 1, 4)
        plt.imshow(processed, 'gray')
        plt.title('Processed for Segmentation')
        plt.axis('off')

        plt.show()

    return image_clean, gray, processed


def get_center_aligned_boxes(processed, gray, height_scale=1.3, merge_thresh=1.0, visualize=False):
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
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.imshow(vis[..., ::-1])
        plt.title("Gray Center-Aligned Fixed-Height Boxes")
        plt.axis("off")
        
        vis2 = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(final_boxes):
            cv2.rectangle(vis2, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(vis2, str(i+1), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        plt.subplot(2, 1, 2)
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

''' Get dominant hue peaks in the image '''
def get_hue_peaks(image, val_thresh=10, sat_thresh=2, hist_smooth=5, peak_thresh=0.1, hue_merge=5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Mask: keep pixels that are visibly colored
    mask = (S > sat_thresh) & (V > val_thresh)
    mask = mask.astype(np.uint8) * 255

    # Compute hue histogram of colored pixels only
    hist = cv2.calcHist([H], [0], mask, [180], [0, 180]).flatten()

    # Smooth histogram to merge tiny hue variations
    hist = cv2.GaussianBlur(hist.reshape(-1,1), (hist_smooth,1), 0).flatten()

    # Find peaks where histogram value > threshold
    peaks = np.where(hist > hist.max() * peak_thresh)[0]

    if len(peaks) == 0:
        return []

    # Merge nearby peaks (within hue_merge degrees)
    merged = []
    current = [peaks[0]]
    for p in peaks[1:]:
        if p - current[-1] <= hue_merge:
            current.append(p)
        else:
            merged.append(int(np.mean(current)))
            current = [p]
    merged.append(int(np.mean(current)))
    return merged

def get_color_clusterings(image, val_thresh=3, sat_thresh=2, hist_smooth=5, peak_thresh=0.1, hue_merge=3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    peaks = get_hue_peaks(image, val_thresh, sat_thresh, hist_smooth, peak_thresh, hue_merge)
    if not peaks:
        return [image.copy()]  # fallback, nothing found
    
    # Build color masks and images
    color_imgs = []
    for center in peaks:
        lower = np.array([max(center - hue_merge, 0), sat_thresh, val_thresh])
        upper = np.array([min(center + hue_merge, 179), 255, 255])
        color_mask = cv2.inRange(hsv, lower, upper)
        # optional: clean mask with morphology
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,
                                      np.ones((3,3), np.uint8))
        color_only = cv2.bitwise_and(image, image, mask=color_mask)

        # make non-mask areas white for clarity
        white_bg = np.full_like(image, 255)
        color_img = np.where(color_mask[:,:,None].astype(bool),
                             color_only, white_bg)
        color_imgs.append(color_img)

    return color_imgs

def extract_by_hue(image_clean):
    color_clusterings = get_color_clusterings(image_clean)
    segmentations = []

    for cluster_img in color_clusterings:
        # preprocessing (grayscale, binarization)
        cluster_gray = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2GRAY)

        # adaptive thresholding for varying illumination 
        thresh = cv2.adaptiveThreshold(
            cluster_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            15, 10
        )
        processed = thresh

        boxes, _ = get_center_aligned_boxes(processed, cluster_gray, merge_thresh=10, visualize=False)
        
        for box in boxes:
            (x, y, w, h) = box
            crop = cluster_img[y:y+h, x:x+w]
            segmentations.append((box, crop))

    return segmentations

def handle_connected_characters(cleaned_image, boxes, visualize=False):
    # Count from back to front to avoid index issues when modifying list
    segmentations = []
    for i in range(len(boxes)-1, -1, -1):
        x, y, w, h = boxes[i]
        cleaned_crop = cleaned_image[y:y+h, x:x+w]

        peaks = get_hue_peaks(cleaned_crop)
        # Multiple hue peaks indicate connected characters, further segmentation needed
        if len(peaks) > 1:
            new_segmentations = extract_by_hue(cleaned_crop)
            for box, crop in new_segmentations:
                # adjust box coordinates to original image
                abs_box = (box[0] + x, box[1] + y, box[2], box[3])
                segmentations.append((abs_box, crop))
        else:
            segmentations.append((boxes[i], cleaned_crop))

    # sort left-to-right
    segmentations = sorted(segmentations, key=lambda b: b[0])

    if visualize:
        crops = [crop for _, crop in segmentations]
        n = len(crops)

        cols = max(1, n)
        fig = plt.figure(figsize=(max(8, cols * 2), 6))
        gs = gridspec.GridSpec(2, cols, height_ratios=[2, 1])

        ax0 = fig.add_subplot(gs[0, :])
        # cleaned_image is BGR (cv2) — convert to RGB for matplotlib
        if cleaned_image.ndim == 3 and cleaned_image.shape[2] == 3:
            ax0.imshow(cleaned_image[..., ::-1])
        else:
            ax0.imshow(cleaned_image, cmap='gray')
        ax0.set_title("Original Cleaned Image")
        ax0.axis('off')

        if n == 0:
            plt.tight_layout()
            plt.show()
            return segmentations

        for i, crop in enumerate(crops):
            ax = fig.add_subplot(gs[1, i])
            if crop.ndim == 3 and crop.shape[2] == 3:
                ax.imshow(crop[..., ::-1])
            else:
                ax.imshow(crop, cmap='gray')
            ax.set_title(f'Crop {i+1}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return segmentations
