# File for methods related to converting masks into bounding boxes, may not be used in the final version
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_segmentation_masks
from torchvision.ops import masks_to_boxes
import cv2 as cv


def non_max_suppression_fast(boxes, overlapThresh):
    '''
    This method allows for non-max suppression of boxes allowing us to get the best boxes
    :param boxes: Array of boxes to be processed
    :param overlapThresh: Threshold for overlap
    :return: Array of boxes after non-max suppression from best to worst
    '''
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def plot_result(img, mask, img2):
    '''
    Method to plot the result of the process_images method, plots the original image, the mask and the image with the bounding box
    :param img: original image
    :param mask: mask of the image
    :param img2: image with the bounding box
    :return: Nothing
    '''
    # plot img, mask and img2
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(img2)
    plt.show()


def process_images(image_path, masks_path):
    '''
    Method still to be fully made
    :param image_path: Path to the images folder
    :param masks_path: Path to the masks folder
    :return: Creates a folder with the images and masks drawn on them
    '''
    plt.rcParams["savefig.bbox"] = "tight"

    # for every image in the folder
    for filename in os.listdir(image_path):
        img_path = os.path.join(image_path, filename)
        mask_path = os.path.join(masks_path, filename)
        img = read_image(img_path, mode=ImageReadMode.RGB)
        mask = read_image(mask_path, mode=ImageReadMode.GRAY)
        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set of boolean masks.
        # Note that this snippet would work as well if the masks were float values instead of ints.
        masks = mask == obj_ids[:, None, None]
        drawn_masks = []
        for mask in masks:
            drawn_masks.append(draw_segmentation_masks(img, mask, alpha=0.8, colors="blue"))
        boxes = masks_to_boxes(masks)
        # convert to (x, y, w, h) format
        boxes = boxes.numpy()
        # int boxes
        boxes = boxes.astype(np.int32)
        # convert all boxes
        for box in boxes:
            box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        box = non_max_suppression_fast(boxes, 0.9)
        print(len(box))

        x = box[0][0]
        y = box[0][1]
        w = box[0][2]
        h = box[0][3]
        h = int(h / 2)
        # open image using open cv
        img1 = cv.imread(img_path)
        # draw rectangle on image
        img2 = cv.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plot_result(img, mask, img2)

        # create output folder if it doesn't exist
        if not os.path.exists('output'):
            os.makedirs('output')
        cv.imwrite('output/' + filename, img2)
