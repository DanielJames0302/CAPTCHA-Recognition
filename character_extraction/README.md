# Extraction of characters

Since Prof highly suggested looking at one character at a time instead of the whole picture, I was looking into ways to implement it. 

original find (where i took the inspiration from):
https://iopscience.iop.org/article/10.1088/1742-6596/2504/1/012036/pdf 
this one doesn't really work for us as it does not have any noises or overlapping characters 

on github: https://github.com/ferriswym/solving_captchas_code_examples 

## Preprocessing
- Removing black noise lines:
We can either turn black pixels to white (https://stackoverflow.com/questions/62494766/replace-a-color-threshold-within-numpy-image-python) or try to blend it.

For blending you can either replace with neighboring pixels (https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html) or use seamless cloning (i'm not using it but it is a possibility: https://blog.finxter.com/5-effective-ways-to-remove-horizontal-lines-in-images-using-opencv-python-and-matplotlib/)

- Gray level
- Processed (binary version)

## Extraction/Boxes

This basically just looks at the connected components of the processed (binary) image with cv2.connectedComponentsWithStats(): https://www.geeksforgeeks.org/machine-learning/python-opencv-connected-component-labeling-and-analysis/

Then it merges the boxes that are too close to each other (we can play with the threshold value: merge_thresh, we can also uncomment the old_boxes to compare with the initial boxes before merging if ever you think it merging too many letters together)

From what I've observed, most string characters are at the same height level, so I force a fixed centered height to make sure we get the entire character in the box.

