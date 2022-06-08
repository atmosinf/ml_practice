import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image = Image.open('images/elon.jpeg')

transform = A.Compose([
    A.Resize(width=1920, height=1080),
    A.RandomCrop(width=1280, height=720),
    A.Rotate(limit=40, p=0.9), # p stands for the probability that this transform is applied. set: border_mode = cv2.BORDER_CONSTANT, if you want the rotate to be with black fill in rather than a zoomed cropped image 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.OneOf([
        A.Blur(blur_limit=4, p=0.5),
        A.ColorJitter(p=0.5),
    ], p=1.0) # one of either blur or colorjitter will be applied 100% of the time.
])

images_list = [image]
image = np.array(image)
for i in range(15):
    augmentations = transform(image=image) # returns a dictionary
    augmented_img = augmentations['image'] # select the image key from the dictionary
    images_list.append(augmented_img)

plot_examples(images_list) 
