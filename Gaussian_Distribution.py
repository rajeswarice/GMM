import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from skimage.color import rgb2hsv, hsv2rgb
import cv2 as cv
import glob
import os
import os.path
from PIL import Image

def gaussian(p,mean,std):
    return np.exp(-(p-mean)**2/(2*std**2))*(1/(std*((2*np.pi)**0.5)))

def getParameters():

    # read the rgb & labelled image
    image_path_list = glob.glob("train/vsi/*.jpg") 

    # Empty dataframe to hold hsv values
    data = pd.DataFrame([], columns=['h','s','v'])

    for i in range(0,7):

        fn_prefix, _ = os.path.splitext(image_path_list[i])        
        rgb_path = os.path.join(fn_prefix.split('\\')[0] + '/'+ fn_prefix.split('\\')[1]+ ".jpg")
        lbl_path = os.path.join(fn_prefix.split('\\')[0].split('/')[0] + '/' + "lbl" + '/'+ fn_prefix.split('\\')[1]+ "_L.png")
        
        rgb_img = imread(rgb_path)
        lbl_img = imread(lbl_path)

        # convert rgb to hsv
        hsv_img = rgb2hsv(rgb_img)

        # get the shape of the image
        rows,cols,d = rgb_img.shape 

        # create a mask based on labeled image
        binary_mask = lbl_img[:,:,1] == 255
        mask = np.ones(hsv_img.shape, np.uint8)
        mask = mask[:,:,0]*binary_mask

        # overlay mask over hsv image
        masked_img = hsv_img * binary_mask[...,None]

        # load all sunlit areas hsv values
        for r in range(rows):
            for c in range(cols):
                if(masked_img[r,c] != [0,0,0]).all():
                    nr = {'h':masked_img[r,c][0], 's':masked_img[r,c][1], 'v':masked_img[r,c][2]}
                    data = data.append(nr,ignore_index=True)    

    # get mean values of hsv
    mean_h = np.mean(data['h'])
    mean_s = np.mean(data['s'])
    mean_v = np.mean(data['v'])

    # get standard deviation values of hsv
    std_h = np.std(data['h'])
    std_s = np.std(data['s'])
    std_v = np.std(data['v'])

    # save the parameters
    np.save('parameters/mean_h', mean_h)
    np.save('parameters/mean_s', mean_s)
    np.save('parameters/mean_v', mean_v)

    np.save('parameters/std_h', std_h)
    np.save('parameters/std_s', std_s)
    np.save('parameters/std_v', std_v)

def getCroppedRGBImage(file_name):
    cropped_rgb_suffix = '_rgb_image_cropped.jpg'
    fn_prefix, _ = os.path.splitext(file_name)
    cropped_rgb_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + cropped_rgb_suffix)
    return cropped_rgb_filename

def imageSegment(image_path):
    # print(image_path)
    rgb_img = imread(getCroppedRGBImage(image_path))
    print(getCroppedRGBImage(image_path))
    hsv_img = rgb2hsv(rgb_img)

    mean_h = np.load('parameters/mean_h.npy')
    mean_s = np.load('parameters/mean_s.npy')
    mean_v = np.load('parameters/mean_v.npy')

    std_h = np.load('parameters/std_h.npy')
    std_s = np.load('parameters/std_s.npy')
    std_v = np.load('parameters/std_v.npy')

    # use the mean and standard deviation in the gaussian distribution
    masked_image_h = gaussian(hsv_img[:,:,0], mean_h, std_h)
    masked_image_s = gaussian(hsv_img[:,:,1], mean_s, std_s)
    masked_image_v = gaussian(hsv_img[:,:,2], mean_v, std_v)

    # get the final mask by combining all
    final_mask = masked_image_h * masked_image_s * masked_image_v

    # create a binarized mask having clustering probability above 0.9
    binarized_mask = final_mask > 0.9

    final_img = rgb_img*binarized_mask[...,None]

    for i in range(final_img.shape[0]):    # for every col:
        for j in range(final_img.shape[1]):    # For every row
            # set the colour accordingly
            if(final_img[i,j] > 0).all():
                final_img[i,j] = [0,255,0]
            else:
                final_img[i,j] = [165,42,42]

    # final_img[final_img > 0] = 255
    # final_img[final_img == 0] = [165,42,42]

    fn_prefix, _ = os.path.splitext(image_path)
    gmm_filename = os.path.join(fn_prefix.split('\\')[0] + '/' + fn_prefix.split('\\')[1] + "_gmm.jpg")
    imshow(final_img)
    plt.show()
    final = Image.fromarray(final_img)
    final.save(gmm_filename)

if __name__ == '__main__':    

    if not(os.path.exists("parameters/mean_h.npy")):
        getParameters()

    image_path_list = glob.glob("test/*.jpg")

    for image_path in image_path_list:
        imageSegment(image_path)
        break
        
   