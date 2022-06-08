## albumentations

[youtube link](https://www.youtube.com/watch?v=rAdLwKJBvPM&t=18s)<br>
[reference github link](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/Basics/albumentations_tutorial)<br>


### output of classification.py
![1][screenshots/1.png] <br><br>

### output of segmentation.py when run with a single mask
![2][screenshots/2.png] <br>
each image is followed by its mask <br>

### output of segmentation.py when run with 2 masks 
![3][screenshots/3.png] <br>
each image is followed by its 2 masks <br>

### output of detection.py
![4][screenshots/4.png] <br>
the bounding boxes (which were randomly created) follow the transformations. notice that when the image is rotated, the bbox is inflated. this is done deliberately and is expected. <br>

![5][screenshots/5.png] <br>
note that sometimes after augmentations, the bboxes can return empty lists<br>

### output of detection.py after setting min_area=2048, min_visibility=0.3
![6][screenshots/6.png] <br>
<br>

### output of full_pytorch_example.py
![7][screenshots/7.png] <br>
<br>