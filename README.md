# ml_practice

# pytorch

## built in datasets
### CIFAR10
learn how to use the pytorch built-in CIFAR10 dataset. keywords: train test validation split, tqdm, progress bar, torchvision.utils.make_grid, CxHxW to HxWxC using np.transpose(npimg, (1,2,0))

## transfer learning
keywords: train test split an ImageFolder, transfer learning on mobilenetv3, freeze layers, learning rate scheduler, workaround for different transforms in ImageFolder which uses a random_split for train and test data and does not have separate folders for train and test. this will be useful when you use data augmentation in such a scenario (when the data for ImageFolder is in a single folder, rather than split into train and test)