import cv2
import cv2 
import numpy as np
import os
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import feature, img_as_bool
from skimage.morphology import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
import imutils

code_path = os.getcwd()

def disp_img(img,win_name= "Name"):
    """
    Function to display image.
    Input:
        img = Image array
        win_name = Name of the window
    Output:
        Returns None
        Outputs the image in a new window.
    """
    cv2.imshow(win_name,img)
    cv2.waitKey(0)


def make_image_collage(img_list):
    try:
        if len(img_list)%2 == 0:
            pass
        else:
            img_list.append(img_list[0].copy())
        
        row1 = np.hstack(img_list[0:len(img_list)//2])
        row2 = np.hstack(img_list[len(img_list)//2:])

        final_image = np.vstack([row1,row2])
        disp_img(final_image)

    except ValueError:
        print("The shape of your images dont match.")
        for i in img_list:
            print(i.shape)


