'''
Create a function that copies over a folder (preserving the folder hierachy) with all its images, saving each image using PIL. 
This will get rid of all corrupt images (corrupt images might cause a model training session to crash in the middle of training).
The images after they're saved by PIL will also be smaller in size
'''

import os
from PIL import Image 
from tqdm import tqdm
import shutil
import sys


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
    SOURCE_DIR = 'animals_africa'
    TARGET_DIR = 'animals_africa_cleaned'

    # copy over the directory tree, excluding the files
    shutil.copytree(SOURCE_DIR, TARGET_DIR, ignore=ig_f)

    filelist = get_filelist(SOURCE_DIR) 

    savedlist = []
    failedlist = []
    for f in tqdm(filelist):
        try:
            img = Image.open(f)
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except:
            failedlist.append(f)   
            print(f'\n{f} cannot be opened. Skipping..\n')
        targetloc = f.replace(SOURCE_DIR, TARGET_DIR).replace('JPG','jpg')
        
        try:
            img.save(targetloc)
            savedlist.append(f)
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
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
TARGET_DIR -> example: 'sourcedir_cleaned'

COPY_SUCCESSFUL.txt and COPY_FAILED.txt to see which images were saved and which were not
'''
