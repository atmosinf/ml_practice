# ml_practice

---

# pytorch

## built in datasets
### CIFAR10
learn how to use the pytorch built-in CIFAR10 dataset. keywords: train test validation split, tqdm, progress bar, torchvision.utils.make_grid, CxHxW to HxWxC using np.transpose(npimg, (1,2,0))

## transfer learning
keywords: train test split an ImageFolder, transfer learning on mobilenetv3, freeze layers, learning rate scheduler, workaround for different transforms in ImageFolder which uses a random_split for train and test data and does not have separate folders for train and test. this will be useful when you use data augmentation in such a scenario (when the data for ImageFolder is in a single folder, rather than split into train and test)

## neural style transfer
neural style transfer is a great place to check the impact of a learning rate scheduler. check the notebook and the images to see the difference between training with a scheduler and without. keywords: PIL, lr, lr_scheduler.StepLR, gram matrix

## object detection
### IOU
calculate intersection over union. learn to create rectangles in matplotlib using matplotlib.patches.Rectangle and ax.add_patches. use plt.gca().invert_yaxis() to set the origin at top left. this is done because in computer vision, the origin is usually kept at the top left. keywords: iou, intersection, union, .clip(), matplotlib.patches

### object localization
notebook run in kaggle. used a dataset that provided the object location (x1,y1,x2,y2) and created a simple model that predicts the output. MSELoss was used.  

### YOLOv1
#### dataset.py and dataset_test.ipynb
create a custom dateset for the PASCAL VOC data. raw data has bbox in coordinates, the custom dataset returns the bboxes relative to the grid cell, with the midpoint and height and width. keywords: object detection dataset

---