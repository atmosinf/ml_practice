'''
Create a function that copies over a folder (preserving the folder hierachy) with all its images, saving each image using PIL. 
This will get rid of all corrupt images (corrupt images might cause a model training session to crash in the middle of training).
The images after they're saved by PIL will also be smaller in size
'''

'''
Create a function that copies over a folder (preserving the folder hierachy) with all its images, saving each image using PIL. 
This will get rid of all corrupt images (corrupt images might cause a model training session to crash in the middle of training).
The images after they're saved by PIL will also be smaller in size
'''

import os
from PIL import Image 
from tqdm import tqdm
import shutil


def get_filelist(dirname):
    filelist = os.listdir(dirname)
    allfiles = []
    for f in filelist:
        fullpath = f'{dirname}/{f}'
        if os.path.isdir(fullpath):
            allfiles = allfiles + get_filelist(fullpath)
        else:
            allfiles.append(fullpath)
                
    return allfiles

def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def run():
    SOURCE_DIR = 'Lope-Waka'
    TARGET_DIR = 'Lope-Waka_256'
    TARGET_SIZE = (341, 256) # calculated using 256 as the smaller dim (height in this case). the width is calculated using 256 and the aspect ratio of the source image. all source images had the same aspect ratio for this test. 

    # copy over the directory tree, excluding the files
    shutil.copytree(SOURCE_DIR, TARGET_DIR, ignore=ig_f)

    filelist = get_filelist(SOURCE_DIR) 

    savedlist = []
    failedlist = []
    for f in tqdm(filelist):
        try:
            img = Image.open(f)
        except:
            print(f'\n{f} cannot be opened. Skipping..\n')
            failedlist.append(f)   
        targetloc = f.replace(SOURCE_DIR, TARGET_DIR).replace('JPG','jpg')
        
        try:
            img = img.resize(TARGET_SIZE)
            img.save(targetloc)
            savedlist.append(f)
        except:
            failedlist.append(f)
            print(f'\n{f} could not be saved\n')
    
    with open('COPY_SUCCESSFUL.txt', 'w') as f:
        for line in savedlist:
            f.write(f'{line}\n')

    with open('COPY_FAILED.txt', 'w') as f:
        for line in failedlist:
            f.write(f'{line}\n')

    print(f'{len(savedlist)} images were saved successfully.')
    print(f'{len(failedlist)} images were corrupted and could not be saved.')
    print('please check COPY_SUCCESSFUL.txt and COPY_FAILED.txt to see which images were saved and which were not')

run()

'''
INSTRUCTIONS:
set variables:
SOURCE_DIR -> example: 'sourcedir'
TARGET_DIR -> example: 'sourcedir_cleaned' (this directory is created by this script, and does not already exist)
TARGET_SIZE -> (width, height) example: (341, 256)

COPY_SUCCESSFUL.txt and COPY_FAILED.txt to see which images were saved and which were not
'''
