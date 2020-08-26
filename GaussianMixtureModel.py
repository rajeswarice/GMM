import numpy as np
import cv2 as cv
import glob
from PIL import Image
from scipy.stats import multivariate_normal as mvn
import os
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def EMalgo(xtrain, K, iters, epsilon=1e-10):
    n, d = xtrain.shape
    mean = xtrain[np.random.choice(n, K, False), :]
    Sigma = [80 * np.eye(d)] * K
    for i in range(K):
        Sigma[i] = np.multiply(Sigma[i], np.random.rand(d, d))

    w = [1. / K] * K
    z = np.zeros((n, K))

    # EM algorithm 
    log_likelihoods = []
    prev_log_likelihood = None
    while len(log_likelihoods) < iters:
        for k in range(K):   
            pdf = mvn.pdf(xtrain, mean[k], Sigma[k], allow_singular = True) 
            tmp = w[k] * pdf 
            z[:, k] = tmp.reshape((n,))     

        log_likelihood = np.sum(np.log(np.sum(z, axis=1)))
        
        print('{0} -> {1}'.format(len(log_likelihoods), -log_likelihood))

        log_likelihoods.append(log_likelihood)
        # E_Step 
        z = (z.T / np.sum(z, axis=1)).T
        N_ks = np.sum(z, axis=0)

        # M_Step 
        for k in range(K):
            mean[k] = 1. / N_ks[k] * np.sum(z[:, k] * xtrain.T, axis=1).T
            x_mean = np.matrix(xtrain - mean[k])
            Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mean.T, z[:, k]), x_mean))
            w[k] = 1. / n * N_ks[k]

        # convergence check
        if prev_log_likelihood != None and abs(log_likelihood - prev_log_likelihood) < epsilon:
            break

        prev_log_likelihood = log_likelihood

    # log_likelihoods
    plt.plot(log_likelihoods)
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()

    # save the parameters
    np.save('GMM/parameters/weights', w)
    np.save('GMM/parameters/sigma', Sigma)
    np.save('GMM/parameters/mean', mean)    

def getCroppedRGBImage(file_name):
    cropped_rgb_suffix = '_rgb_image_cropped.jpg'
    fn_prefix, _ = os.path.splitext(file_name)
    cropped_rgb_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + cropped_rgb_suffix)
    return cropped_rgb_filename

def getLabeledImage(file_name):
    labeled_suffix = '_L.png'
    fn_prefix, _ = os.path.splitext(file_name)
    labeled_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + labeled_suffix)
    return labeled_filename

def train(image_path_list, K = 13):
    train_fileNames = []
    train_TrueImg_fileName = []
    train_data = np.array([])
    for i in range(0,1):
        file_name = getCroppedRGBImage(image_path_list[i])        
        true_file_name = getLabeledImage(image_path_list[i])
        #--------------------------------------------
        img = Image.open(file_name)
        img.load()
        rgb_img = np.asarray( img, dtype="int32" )
        h, w, ch = rgb_img.shape
        image_pixels = np.reshape(rgb_img, (-1, ch))
        # --------------------------------------------
        img1 = Image.open(true_file_name)
        img1 = img1.convert('RGBA')
        data = np.array(img1)   # "data" is a height x width x 4 numpy array
        red, green, blue, alphas = data.T # Temporarily unpack the bands for readability
        
        # Replace noise with brown
        brown_areas = (red == 165) & (blue == 42) & (green == 42)
        data[..., :-1][brown_areas.T] = (0, 0, 0) # Transpose back needed        
        img1 = Image.fromarray(data)
        img1 = img1.convert('1')
        img1.load()
        true_img = np.asarray(img1, dtype="int32")
        true_image_pixels = np.reshape(true_img, (-1))
        # --------------------------------------------
        tmp = image_pixels[true_image_pixels > 0]
        if len(train_data)==0:
            train_data = tmp.copy()
        else:
            train_data = np.concatenate((train_data, tmp),axis=0)

    EMalgo(train_data, K, 1000)

def imageSegment(file_name):
    gmm_suffix = "_gmm.jpg"
    downscaled_suffix = "_gmm_downscaled.jpg"
    cropped_img_suffix = '_rgb_image_cropped.jpg'
    csv_suffix = '_thermal_values.csv'
    subfolder_path = os.path.splitext(file_name)[0]
    image_name_without_suffix = file_name.split("\\")[1].split('.')[0]
    cropped_rgb_img_filename = subfolder_path + '\\' + image_name_without_suffix + cropped_img_suffix
    csv_filename = subfolder_path + '\\' + image_name_without_suffix + csv_suffix

    #region rgb segmentation

    # load the parameters of the trained GMM 
    w = np.load('GMM/parameters/weights.npy')
    Sigma = np.load('GMM/parameters/sigma.npy')
    mean = np.load('GMM/parameters/mean.npy')

    # read the image to segment 
    img = Image.open(cropped_rgb_img_filename);   
    img.load()
    rgb_img = np.asarray(img, dtype="int32")

    # transform the rgb image into flatten image to segment the pixels
    nr, nc, d = rgb_img.shape
    n = nr * nc
    xtest = np.reshape(rgb_img, (-1, d))

    # initialize the likelihood value
    likelihoods = np.zeros((K, n))
    log_likelihood = np.zeros(n)

    # find the likelihood value for each pixels 
    for k in range(K):
        likelihoods[k] = w[k] * mvn.pdf(xtest, mean[k], Sigma[k],  allow_singular=True)
        log_likelihood = likelihoods.sum(0)

    # segment the pixels into object and non-object according with likelihood 
    log_likelihood = np.reshape(log_likelihood, (nr, nc))
    log_likelihood[log_likelihood > np.max(log_likelihood) / 100] = 255
    log_likelihood[log_likelihood <= np.max(log_likelihood) / 100] = 0

    # get the segmented rgb image from original image  
    segmented_img = np.uint8(np.zeros(rgb_img.shape))
    for n in range(nr):
        for m in range(nc):
            if log_likelihood[n, m] == 255:
                segmented_img[n, m, :] = [0,255,0]
            else:
                segmented_img[n, m, :] = [165,42,42]
    
    #endregion

    # ------- initialize the likelihood value --------------
    # plt.figure(figsize=(15,5))
    # plt.subplot(1,3,1)
    # plt.imshow(rgb_img)
    # plt.title('Original Image')

    fn_prefix, _ = os.path.splitext(file_name)

    gmm_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + gmm_suffix)
    downscaled_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + downscaled_suffix)

    # plt.subplot(1, 3, 2)
    # plt.imshow(segmented_img)
    segmented = Image.fromarray(segmented_img)
    segmented.save(gmm_filename)
    # plt.imshow(segmented)
    # plt.show()
    # plt.title('Segmented Image')
    
    # plt.subplot(1, 3, 3)
    new_mask = cv.resize(segmented_img, dsize=(80, 60), interpolation=cv.INTER_AREA)
    downscaled = Image.fromarray(new_mask)
    downscaled.save(downscaled_filename)
    downscaled_np = np.asarray(downscaled, dtype="int32")

    data = pd.read_csv(csv_filename, sep=',', parse_dates=False)

    # Create a new column that has rounded temperatures
    data['Temp_rounded(c)'] = data['Temp(c)'].map(lambda tempc: round(tempc))
    templ = []    
    temps = data['Temp_rounded(c)'] 

    row, col, d = downscaled_np.shape
    for r in range(row):
        for c in range(col):
            if (downscaled_np[r,c] != [165,42,42]).all():
                temp = data.loc[((data['x'] == r) & (data['y'] == c)),'Temp_rounded(c)'].values.tolist()
                if temp != []:
                    templ.append(temp[0]) 
            
    t_dry = max(templ)
    t_wet = min(templ)
    t_c = np.mean(templ) 
    CWSI = (t_c - t_wet)/(t_dry - t_wet)
    # print (CWSI)
    return t_c
           
if __name__ == '__main__':
    start_train = datetime.now()
    print("Training start : ", start_train)
    K = 7

    image_path_list = glob.glob("train/*.jpg") 

    if not(os.path.exists("GMM/parameters/weights.npy")):
        train(image_path_list)
    end_train = datetime.now()
    print("Training end : ", end_train)
    print('Train Duration: {}'.format(end_train - start_train))   

    start_test = datetime.now()
    print("Testing start : ", start_test)
    image_path_list = glob.glob("test/*.jpg")
    temps = []

    for image_path in image_path_list:
        temp = imageSegment(image_path)
        temps.append(temp)   
        break     

    end_test = datetime.now()
    print("Testing end : ", end_test)
    print('Test Duration: {}'.format(end_test - start_test)) 