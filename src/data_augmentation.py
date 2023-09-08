import os

import cv2
import numpy as np


def blur_pic(path_img, scale):
    """
    Blur the picture
    parameters:
        path_img: path of the image to blur
        scale: scale of the blur
    """

    img = cv2.imread(path_img)
    blur = cv2.blur(img, (scale, scale))
    cv2.imwrite(path_img, blur)


def darken_pic(path_img, scale):
    """
       darken the picture
       parameters:
           path_img: path of the image to darken
           scale: scale of parameter to darken
       """

    img = cv2.imread(path_img)
    darken = cv2.addWeighted(img, scale, img, 0, 0)
    cv2.imwrite(path_img, darken)


def noise_pic(path_img, scale):
    """
       noise the picture
       parameters:
           path_img: path of the image to add noise
           scale: scale of parameter to add noise
       """

    img = cv2.imread(path_img)
    # add a bit of noise to the image, to make it harder to detect
    noise = np.random.randint(0, scale, img.shape)
    noise = noise.astype('uint8')
    noise_img = cv2.add(img, noise)
    cv2.imwrite(path_img, noise_img)


def random_augmentation(path_img):
    """
    Randomly augment the image, with blur, darken and noise, or not.
    Augmentation can be done multiple times on the same image.
    parameters:
        path_img: path of the image to augment
    """
    is_blur = np.random.randint(0, 2)
    is_darken = np.random.randint(0, 2)
    is_noise = np.random.randint(0, 2)

    if is_blur:
        print("blur")
        scale = np.random.randint(4, 10)
        blur_pic(path_img, scale)
    if is_darken:
        print("darken")
        # scale random between 0.1 and 0.9
        scale = np.random.randint(20, 60) / 100
        darken_pic(path_img, scale)
    if is_noise:
        print("noise")
        scale = np.random.randint(50, 150)
        noise_pic(path_img, scale)


def augment_folder(path_folder, path_folder_augmented):
    """
    Augment all the images in the folder
    parameters:
        path_folder: path of the folder to augment
        path_folder_augmented: path of the folder to save the augmented images
    """

    # create copy of the folder
    os.system("cp -r " + path_folder + " " + path_folder_augmented + "_augmented")
    path_folder = path_folder_augmented + "_augmented/"

    for img in os.listdir(path_folder):
        if img.endswith(".jpg") or img.endswith(".png"):
            print(path_folder + img)
            random_augmentation(path_folder + img)
    print("Augmentation done")
