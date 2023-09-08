import json
import os

with open("config.json", "r") as f:
    config = json.load(f)


def init(name):
    if not os.path.exists(name):
        os.makedirs(name)
        print("Directory ", name, " Created ")
    else:
        print("Directory ", name, " already exists")


def init_directory():
    # init the directory from config.json

    folders = config["paths"].values()

    for element in folders:
        init(element)


init_directory()
