# imports
import json
import os
import random
import shutil
import cv2 as cv
from src.yolo import numerical_sort

with open("config.json", "r") as f:
    config = json.load(f)


# yolo
def train_yolo(batch_size, epochs, data, name):
    """
    Function to train yolo
    :param batch_size:  int, batch size
    :param epochs:      int, number of epochs
    :param data:        str, path to data.yaml
    :param name:        str, name of the model
    """
    data += "/data.yaml"
    print(batch_size, epochs, data, name)
    os.system("python yolo/yolov7/train.py --workers 8 --device 0 --batch-size " + str(
        batch_size) + " --data " + data + "--img 512 512 --cfg src/yolo/yolov7/cfg/training/yolov7.yaml --weights "
                                          " --name "
              + name + "--hyp src/yolo/yolov7/data/hyp.scratch.p5.yaml --epoch " + str(
        epochs))


# add random files from a folder to another folder
def add_random(path_to_folder, number_of_images):
    """
    Function to add random files from a folder to another folder

    Parameters
    ----------
    path_to_folder : str, path to folder with files
    number_of_images : int, number of files to get from path
    ----------
    """

    # in case it is a string
    number_of_images = int(number_of_images)
    # in case path_to_folder doesn't end with /, add it
    if path_to_folder[-1] != "/":
        path_to_folder = path_to_folder + "/"

    # loop to get random files
    for i in range(number_of_images):
        random_file = random.choice(os.listdir(path_to_folder))

        file_img = random_file[:-3] + "jpg"
        file_txt = random_file[:-3] + "txt"

        path_img = path_to_folder + file_img
        path_txt = path_to_folder + file_txt

        # check if path_img exist
        if not os.path.exists(path_img):
            # delete 3 last characters of path_img
            path_img = path_img[:-3] + "png"

        # copy files to img/
        # if txt is empty, don't copy, it's useless
        if os.path.getsize(path_txt) > 0:
            shutil.copy(path_img, config['paths']['bp'] + file_img)
            shutil.copy(path_txt, config['paths']['bp'] + file_txt)
        else:
            # decrease to get the good number of images
            print("file empty")
            i -= 1


# Repartition of classes in dataset
def class_repartition(path_to_folder):
    """
    Function to count how many car, pedestrian and others are in a folder (repartition of classes)
    Parameters
    ----------
    path_to_folder : str, path to folder with files
    ----------
    """

    # in case path_to_folder doesn't end with /, add it
    if path_to_folder[-1] != "/":
        path_to_folder = path_to_folder + "/"

    car, pedestrian, others = 0, 0, 0
    # for all files in train folder
    for file in os.listdir(path_to_folder):
        # if file is a txt
        if file.endswith(".txt"):
            # read txt
            print(file)
            with open(path_to_folder + file, 'r') as file_reader:
                data = file_reader.readlines()
                # for each line in txt
                for line in data:
                    # get class
                    cl = line[0]
                    # count classes
                    if cl == "0":
                        car += 1
                    elif cl == "1":
                        pedestrian += 1
                    else:
                        others += 1

    print("cars : " + str(car))
    print("pedestrians : " + str(pedestrian))
    print("others : " + str(others))

    out = [car, pedestrian, others]
    return out


# merge 2 folders
def add_quantity(path_to_get, path_to_add):
    """
    Function to add quantity of files from a folder to another folder, and rename them to match the number of files
    in the folder
    Parameters
    ----------
    path_to_get : str, path to folder with files to get
    path_to_add : str, path to folder with files to add
    -----------
    """

    print(path_to_get, path_to_add)
    # if path don't end with / add it
    if path_to_get[-1] != "/":
        path_to_get = path_to_get + "/"
    if path_to_add[-1] != "/":
        path_to_add = path_to_add + "/"

    num_of_txt = 0
    for file in os.listdir(path_to_get):
        if file.endswith(".txt"):
            num_of_txt += 1
    print("num of txt : " + str(num_of_txt))

    num = int(len(os.listdir(path_to_add)) / 2)
    i = 0
    # merge path_to_get and path_to_add into path_to_add
    for file in os.listdir(path_to_get):
        if file.endswith(".txt"):
            name = file[:-4]
            name = int(name) + int(num)
            name = str(name)
            name = name + ".txt"
            shutil.copy(path_to_get + file, path_to_add + name[:-4] + "merged.txt")
            if os.path.exists(path_to_get + name[:-4] + ".jpg"):
                shutil.copy(path_to_get + file[:-3] + "jpg", path_to_add + name[:-4] + "merged.jpg")
                i += 1
            else:
                shutil.copy(path_to_get + file[:-3] + "png", path_to_add + name[:-4] + "merged.png")
                i += 1

    # rename every file in path from 0 to total_files, with their txt
    i = 0
    for file in sorted(os.listdir(path_to_add), key=numerical_sort):
        print(file)

        if file.endswith(".txt"):
            os.rename(f"{path_to_add}{file}", f"{path_to_add}{i}.txt")
            print("rename " + file + " to " + f" {i}.txt")
            if os.path.exists(f"{path_to_add}{file[:-4]}.jpg"):
                os.rename(f"{path_to_add}{file[:-4]}.jpg", f"{path_to_add}{i}.jpg")
                print("rename " + file[:-4] + ".jpg" + " to " + f" {i}.jpg")
            else:
                os.rename(f"{path_to_add}{file[:-4]}.png", f"{path_to_add}{i}.png")
                print("rename " + file[:-4] + ".png" + " to " + f" {i}.png")
            i += 1
    print("added " + str(i) + " files")


def display_bbox(path):
    """
    Display the bounding box of images in a folder (1 txt for 1 img), (yolo format)
    :param path: Path of the folder containing the images and txt
    """

    path = f"{path}/"
    # for each img in the folder, find the corresponding txt, and display the bounding box on the img
    for img in sorted(os.listdir(path), key=numerical_sort):
        if img.endswith(".png") or img.endswith(".jpg"):
            # get the name of the img
            name = img.split(".")[0]
            # get the txt file
            txt = f"{name}.txt"
            print(txt)
            print(img)
            print(f"{path}{img}")
            print(f"{path}{txt}")
            img2 = cv.imread(f"{path}{img}")
            # open the txt file
            if os.path.isfile(f"{path}{txt}"):
                with open(f"{path}{txt}", "r") as file:
                    # read the lines
                    lines = file.readlines()
                    # for each line

                    for line in lines:
                        # get the class
                        class_name = line.split(" ")[0]
                        # get the coordinates and convert to display with rectangle
                        x_center = int(float(line.split(" ")[1]) * img2.shape[1])
                        y_center = int(float(line.split(" ")[2]) * img2.shape[0])

                        w = int(int(float(line.split(" ")[3]) * img2.shape[1]) / 2)
                        h = int(int(float(line.split(" ")[4]) * img2.shape[0]) / 2)

                        x_min = x_center - w
                        y_min = y_center - h
                        x_max = x_center + w
                        y_max = y_center + h

                        # draw the bounding box..............
                        cv.rectangle(img2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        # put the class name
                        cv.putText(img2, class_name, (x_min, y_min), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # display the img
            cv.imshow("img", img2)
            # if x is pressed, close the window, if other key is pressed, continue
            if cv.waitKey(0) == ord('x'):
                cv.destroyAllWindows()
                print("Display stopped")
                break
            else:
                continue
                
    print("Display finished")
