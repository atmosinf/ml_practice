import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image = cv2.imread('images/cat.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[12, 20, 300, 321]] # random bbox entered by me, just for demonstration. (xmin, ymin, xmax, ymax)

transform = A.Compose([
    A.Resize(width=1920, height=1080),
    A.RandomCrop(width=1280, height=720),
    A.Rotate(limit=40, p=0.9, border_mode = cv2.BORDER_CONSTANT), # p stands for the probability that this transform is applied. set: border_mode = cv2.BORDER_CONSTANT, if you want the rotate to be with black fill in rather than a zoomed cropped image 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.OneOf([
        A.Blur(blur_limit=4, p=0.5),
        A.ColorJitter(p=0.5),
    ], p=1.0) # one of either blur or colorjitter will be applied 100% of the time.
], bbox_params=A.BboxParams(format='pascal_voc', min_area=2048, min_visibility=0.3, label_fields=[])) # pascal voc because it uses the (xmin, ymin, xmax, ymax) format. yolo for instance, uses the (xcenter, ycenter, width, height) format

images_list = [image]
# image = np.array(image) no need to convert to a numpy array as opencv was used to read the image. when using PIL image.open, we have to convert the PIL image to a numpy array 
saved_bboxes = [bboxes[0]]
for i in range(15):
    augmentations = transform(image=image, bboxes=bboxes) # returns a dictionary
    augmented_img = augmentations['image'] # select the image key from the dictionary

    if len(augmentations['bboxes']) != 0: # this is because sometimes after augmentation, the bboxes might not be in the image, and so will return an empty list. not sure why this happens even after specifying min_area=2048, min_visibility=0.3.
        images_list.append(augmented_img) 
        saved_bboxes.append(augmentations['bboxes'][0])  

plot_examples(images_list, saved_bboxes) 
