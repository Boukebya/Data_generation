import os
import json

global config
with open("config.json", "r") as f:
    config = json.load(f)


def run_yolact(path):
    """
    Run yolact on the images in config sd and save the results in config box, train.py is modified
    to save the results in results.txt
    :return:
    """
    # Create result.txt
    with open("result.txt", "w") as file:
        file.write("")
    # delete all txt files in config sd
    for file in os.listdir(path):
        if file.endswith(".txt"):
            os.remove(f"{path}/{file}")
    os.system(
        f"python yolact/eval.py --trained_model=yolact/weights/yolact_base_54_800000.pth --score_threshold=0.5 "
        f"--top_k=15 --images={path}:{config['paths']['yolact']}")


def run_yolact_cn(path):
    """
    Run yolact on the images in config and save the results in config box, train.py is modified
    to save the results in results.txt
    :return:
    """
    # Create result.txt
    with open("result.txt", "w") as file:
        file.write("")
    os.system(
        f"python yolact/eval.py --trained_model=yolact/weights/yolact_base_54_800000.pth --score_threshold=0.5 "
        f"--top_k=15 --images={path}:{config['paths']['yolact']}")


def create_txt():
    """
    Create txt files for each image in config sd, the txt files contain the results of yolact
    :return:
    """
    # get number of img in config sd
    num_files = len([f for f in os.listdir(config['paths']['sd'])]) - 1
    print(num_files)

    # read the results.txt file
    with open("result.txt", "r") as res:
        # read lines
        lines = res.readlines()

    while True:
        # if line empty or does not exist, break
        if not lines:
            break
        # read the first line
        line = lines[0]
        # get only name.png
        name = line.split("/")[2]
        print(name)
        name = name.split(".")[0]
        lines.pop(0)

        # create txt file with name
        with open(f"{config['paths']['sd']}{name}.txt", "w") as file:
            # if line empty or does not exist, break
            if not lines:
                break
            # while lines[0] doesn't start with path
            while not lines[0].startswith(config['paths']['sd']):
                # write the line in the txt file
                file.write(lines[0])
                # remove the line from the list
                lines.pop(0)
                # if the list is empty, break
                if not lines:
                    break
