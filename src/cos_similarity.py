import json
import os
import shutil
import clip
import cv2
import torch
import numpy as np
from PIL import Image
from src.yolo import numerical_sort

with open("config.json", "r") as f:
    config = json.load(f)


def crop_folder(path, path_out):
    """
    Crop all images in folder corresponding to the bounding box in .txt file
    :param path: path to the folder
    :param path_out: path to the output folder
    """

    # empty path_out
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    os.mkdir(path_out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    num_line = 0
    files = os.listdir(path)
    for file in files:
        if file.endswith(".txt"):
            file_name = file.split(".")[0]
            img = cv2.imread(path + file_name + ".png")
            h, w, _ = img.shape

            # copy file_name_original
            img_original = cv2.imread(config["paths"]["cn_img"] + file_name + "_original.png")
            print(config["paths"]["cn_img"] + file_name + "_original.png", img_original)
            cv2.imwrite(path_out + file_name + "full_img/_original.png", img_original)

            with open(path + file, "r") as file_reader:
                lines = file_reader.readlines()
                i = 0
                for line in lines:
                    print(" line : ", line)
                    line = line.split(" ")
                    x = int(float(line[1]) * w)
                    x = x - int(float(line[3]) * w / 2)
                    y = int(float(line[2]) * h)
                    y = y - int(float(line[4]) * h / 2)
                    width = int(float(line[3]) * w)
                    height = int(float(line[4]) * h)

                    img_crop = img[y:y + height, x:x + width]
                    img_original_crop = img_original[y:y + height, x:x + width]

                    cv2.imwrite(path_out + file_name + "_" + str(i) + ".png", img_crop)
                    cv2.imwrite(path_out + file_name + "_" + str(i) + "_original.png", img_original_crop)

                    # encode image  with CLIP
                    with torch.no_grad():
                        img_encode = preprocess(Image.open(path_out + file_name + "_" + str(i) + ".png")).unsqueeze(
                            0).to(device)
                        image_features = model.encode_image(img_encode)
                        print("image_features : ", image_features.shape)
                        print("image_features : ", type(image_features))
                        # save image features
                        torch.save(image_features, path_out + file_name + "_" + str(i) + "_crop.pt")

                    with torch.no_grad():
                        img_encode = preprocess(Image.open(config["paths"]["cn_img"] +
                                                           file_name + "_original.png")).unsqueeze(
                            0).to(device)
                        image_features = model.encode_image(img_encode)
                        # save image features
                        torch.save(image_features, path_out + file_name + "_" + str(i) + "_original.pt")

                    num_line += 1
                    i += 1
                    print(num_line)


def cos_sim(path_out):
    """
    Compute cosine similarity between two tensors, and average of all tensors, the result is plot for each detection
    in an image.
    path_out: path to the folder containing the tensors
    """

    array = []
    for tensor in sorted(os.listdir(path_out), key=numerical_sort):
        if tensor.endswith("crop.pt"):
            print("tensor : ", tensor)

            tensor_original_name = (tensor[:-7] + "original.pt")

            tensor_original = torch.load(path_out + tensor_original_name)
            tensor_crop = torch.load(path_out + tensor)

            cos = torch.nn.CosineSimilarity(dim=-1)

            output = cos(tensor_crop, tensor_original)

            print("cosine similarity : ", output.item())
            array.append(output.item())

    # plot points in a graph
    import matplotlib.pyplot as plt
    plt.plot(array)
    plt.ylabel('cosine similarity')
    plt.xlabel('Detection boxes')
    # draw average line
    plt.axhline(y=np.mean(array), color='r', linestyle='-')

    plt.show()
