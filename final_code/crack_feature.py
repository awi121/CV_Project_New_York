import cv2 
import numpy as np
import os
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import feature, img_as_bool
from skimage.morphology import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
import imutils
from utils import *

### Define Functions
def feature_preprocess(url):
    # img = imread(url)
    img = cv2.imread(url)
    # disp_img(img,"Original Image")
    # img = rgb2gray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # disp_img(img,"Grayscale")
    # Median Blur is used to remove salt and pepeer noise
    img = cv2.medianBlur(img,5)
    # disp_img(cv2.medianBlur(img,5),"Median Blur")
    # Bilaterlal filter is used to smoothen the image while preserving edges.
    # img = cv2.bilateralFilter(img, 11, 17, 17)
    # disp_img(cv2.bilateralFilter(img, 11, 17, 17),"Bilalteral Filter")
    # Gaussian Blur is kind off like making a gaussian distrivution over the image and finding the mean values in the surrounding pixels.
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # disp_img(cv2.GaussianBlur(img, (5, 5), 0),"Gaussian Filter")

    # img_edge = binary_erosion(binary_dilation(feature.canny(img, sigma =.1)))

    canny_img = auto_canny(img, sigma=0.33)
    
    # Dilation Images
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(canny_img,kernel,iterations = 1)

    # Erode Images
    erosion = cv2.erode(dilation,kernel,iterations = 1)

    # Convert the output of canny edge detection to binary image.
    # thresh, binary = cv2.threshold(erosion,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Adaptive thresholding
    # thresh, binary = cv2.adaptiveThreshold(erosion,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2)
    # print("Threshold for binarization:",thresh)    

    # Edge images
    # img_edge = binary.copy()
    img_edge = erosion.copy()

    """
    Bitwise not for binary image
    bitwise_not_binary = cv2.bitwise_not(binary)
    """

    return img, img_edge


def edge_prob(window, cut_off):
    pixels  = np.array(window.ravel())
    if ((np.count_nonzero(pixels)/len(pixels))>cut_off):
        return 1
    else:
        return 0
    

def sliding_mat(img,window_x=10,window_y=10, cut_off=0.1):
    
    arr_x = np.arange(0,img.shape[0],window_x)
    arr_y = np.arange(0,img.shape[1],window_y)

    A = np.zeros((len(arr_x),len(arr_y)))

    for i,x in enumerate(arr_x):
        for j,y in enumerate(arr_y):
            window = img[x:x+window_x,y:y+window_y]
            A[i,j] = edge_prob(window, cut_off=cut_off)
    
    return A, arr_x, arr_y


def plot_all(img,canny_edge,A):
    fig = plt.figure(figsize = (9,4))
    ax1 = fig.add_subplot(131)
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Original")
    
    ax2 = fig.add_subplot(132)
    ax2.set_title("Canny Edge Detection")
    ax2.imshow(canny_edge, cmap="gray")
    
    ax3 = fig.add_subplot(133)
    ax3.set_title("Mask")
    ax3.imshow(A,cmap="gray")
    plt.tight_layout()
    plt.show()

# Function to get canny image after applying threshold. 
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged


def get_crack_percentage(ip_image):
    # ip_image should be the location to input image
    percentage = 0
    og_img,edge_img = feature_preprocess(ip_image)
    # make_image_collage([og_img,edge_img])

    mask, arr_x, arr_y = sliding_mat(edge_img, window_x=10, window_y=10, cut_off=0.1)

    print(mask)
    percentage = np.sum(mask)/mask.size*100
    # print("Estimate of crack : {:.2f}%".format(percentage))

    # plot_all(og_img,edge_img,mask)
    # make_image_collage([og_img,edge_img])
    return percentage

### Define Path Images
# image_path = "concrete_crack_images"
image_path = code_path 
image_name_list = os.listdir(image_path)

### Define Lambda Functions
get_img_loc = lambda x : os.path.join(image_path,x)
get_img_bgr = lambda x : cv2.cvtColor(x,cv2.COLOR_GRAY2BGR)

### Remove .DS_Store from the code directory and the image directory
# os.system("rm -f {}") {} = image directory 
os.system("rm -f {}/.DS_Store".format(image_path))
os.system("rm -f DS_Store")

### Iterate Over Images

# for image_name in image_name_list[:2]:
#     cv2.destroyAllWindows()
#     print("Current image_name: ",image_name)
#     percentage = get_crack_percentage(get_img_loc(image_name))
#     print("Estimate of crack : {:.2f}%".format(percentage))


    