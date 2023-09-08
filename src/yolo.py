import shutil
import os
import time
import json
import re


with open("config.json", "r") as f:
    config = json.load(f)


def numerical_sort(value):
    """
    Sorter for numerical values
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def yolo_run(weight, path_images, conf):
    """
    Run yolo on config path, manage the files and folders
    :param weight: weight to use, bestCOWC.pt or yolov7.pt (for UDACITY)
    :param path_images: path of images to use (from config)
    :param conf: confidence of yolo
    """

    # Data preparation
    # delete content of yolo/yolov7/temp_copy
    # EMPTY FOLDER
    print("Careful, yolo is adapted to run for COWC and UDACITY, and lines are adapted so that cars are 0, pedestrians 1 and others 2")
    print("using weight : " + weight)
    print("using path : " + path_images)

    for file in os.listdir("yolo/yolov7/temp_copy"):
        os.remove(f"yolo/yolov7/temp_copy/{file}")

    # Empty output of yolo for cn and sd, and txt for stable diffusion
    if path_images == config["paths"]['sd']:
        new_name_img = 0

        # Empty file output of yolo for sd
        with open(config['paths']['yolo_sd'], "w") as yolo_sd_emptier:
            yolo_sd_emptier.write("")

        # rename every images in path from 0 to total_files, delete txt
        for file in sorted(os.listdir(path_images), key=numerical_sort):
            print(file)

            if file.endswith(".png"):
                os.rename(f"{path_images}{file}", f"{path_images}{new_name_img}.png")
                print("rename " + file + " to " + f" {new_name_img}.png")
                new_name_img += 1
            else:
                print("delete" + file)
                os.remove(f"{path_images}{file}")
    # Empty file output of yolo for cn
    else:
        with open(config['paths']['yolo_cn'], "w") as yolo_cn_emptier:
            yolo_cn_emptier.write("")
        with open(config['paths']['yolo_cn_lines'], "w") as yolo_cn_emptier:
            yolo_cn_emptier.write("")

    # copy images from config path to yolo/yolov7/temp_copy
    print("copy of : " + path_images)
    for file in os.listdir(path_images):
        if file.endswith(".png"):
            shutil.copy(f"{path_images}{file}", "yolo/yolov7/temp_copy")
    time.sleep(3)

    # run yolov7 on pictures in temp_copy
    os.chdir("yolo/yolov7")
    os.system(f"python detect.py --weights {weight} --conf {conf} --img-size 512 --source temp_copy/ --save-txt")
    # copy content of runs/detect/exp/labels to config path
    os.chdir("../..")

    # Copy labels of yolo to config stable diffusion path
    if path_images == config["paths"]['sd']:
        for file in os.listdir("yolo/yolov7/runs/detect/exp/labels"):
            shutil.copy(f"yolo/yolov7/runs/detect/exp/labels/{file}", config["paths"]['sd'])

    # If COWC,  update yolo txt, if it's sd, also copy labels to img generated
    if weight == "bestCOWC.pt":
        # replace 1st character of each file by 0, because COWC contains only cars
        if path_images == config["paths"]['sd']:
            for file in os.listdir(path_images):
                if file.endswith(".txt"):
                    with open(f"{path_images}{file}", "r+") as f_reader:
                        lines = f_reader.readlines()
                    for i in range(len(lines)):
                        lines[i] = "0" + lines[i][1:]
                    with open(f"{path_images}{file}", "w") as f_writer:
                        f_writer.writelines(lines)

        else:
            for file in sorted(os.listdir("yolo/yolov7/runs/detect/exp/labels"), key=numerical_sort):
                # count number of line in file
                with open(f"yolo/yolov7/runs/detect/exp/labels/{file}", "r") as f_reader_num:
                    lines = f_reader_num.readlines()
                    i = len(lines)
                    # copy content in config path yolo_cn
                with open(config['paths']['yolo_cn'], "a") as f_writer_num:
                    f_writer_num.writelines(f"{file} : {i}\n")
    # Weight is for ground view
    else:

        size = 0
        directory_img = path_images[:-1]
        for elem in os.listdir(directory_img):
            if elem.endswith(".png"):
                size += 1
        print("size : ", size)

        i = 0
        while i < size:
            print("i = " + str(i))
            if str(i) + ".txt" in os.listdir("yolo/yolov7/runs/detect/exp/labels/"):
                print("file ", str(i), ".txt", " found")
                with open(config['paths']['yolo_cn_lines'], "a") as f_writer1:
                    f_writer1.writelines(f"{i}.png\n")

                count = 0
                with open(f"yolo/yolov7/runs/detect/exp/labels/{i}.txt", "r+") as f_reader:
                    lines = f_reader.readlines()
                    lines_to_add = []

                for z in range(len(lines)):
                    # split lines
                    lines[z] = lines[z].split(" ")

                    if lines[z][0] == "0":
                        lines[z][0] = "1"
                    elif lines[z][0] == "2":
                        lines[z][0] = "0"

                    else:
                        # delete line
                        lines[z] = ""
                    count += 1
                    # join lines to lines_to_add
                    lines_to_add.append(" ".join(lines[z]))

                if path_images == config["paths"]['sd']:
                    with open(f"{path_images}{i}.txt", "w") as f_writer:
                        f_writer.writelines(lines_to_add)
                else:
                    with open(config['paths']['yolo_cn'], "a") as f_writer:
                        f_writer.writelines(f"{i}.txt : {count}\n")
                    with open(config['paths']['yolo_cn_lines'], "a") as f_writer1:
                        f_writer1.writelines(lines_to_add)
                i += 1
            else:
                print("file " + str(i) + ".txt not found")
                with open(f"{config['paths']['yolo_cn']}", "a") as f_writer:
                    f_writer.writelines(f"{i}.txt : 0\n")
                i += 1
        print("i ended at ", str(i))

    # Deletion of images with no txt for stable diffusion
    if path_images == config["paths"]['sd']:
        # Remove images with no txt
        print("aff img to txt ")
        for file in sorted(os.listdir(path_images), key=numerical_sort):
            if file.endswith(".txt"):
                with open(f"{path_images}{file}", "r") as file_reader:
                    lines = file_reader.readlines()
                with open(config['paths']['yolo_sd'], "a") as file_append:
                    file_append.write(f"{file} : {len(lines)}\n")

    # remove folder exp
    shutil.rmtree("yolo/yolov7/runs/detect/exp")
