import os
import subprocess
import time
import requests
from src.generation import run_stable_diffusion, run_controlnet
import json
from src.yolo import numerical_sort

global process
with open("config.json", "r") as f:
    config = json.load(f)


def number_of_files(directory):
    """
    Get the number of files in a directory based on the number of .png files or .jpg files
    :param directory: directory to check
    """
    # get the highest number in dir
    files = os.listdir(directory)
    files.sort()
    if len(files) == 0:
        return 0
    else:
        nb_files = 0
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                nb_files += 1
        return nb_files


def api_close():
    """
    Cancel the API, by killing the process
    """
    process.kill()
    print("Canceled")


def api_monitor():
    """
    Monitor the API, if it is not up, launch it
    """

    is_up = True
    # send a ping request to the API until it is up, allow us to wait for the server to start
    try:
        re = requests.get(url="http://127.0.0.1:7860/sdapi/v1/ping")
        print(re)
        print("ping...")
    except:
        is_up = False

    print(is_up)

    # If the API is not up, launch it
    if is_up is False:
        print("API is not up")
        print("Launching API...")
        global process
        process = subprocess.Popen(["python", "stable-diffusion-webui/launch.py", "--api"])
    else:
        print("API is up")

    # send a ping request to the API until it is up, allow us to wait for the server to start
    while True:
        # send a ping request to the API until it is up, allow us to wait for the server to start
        try:
            re = requests.get(url="http://127.0.0.1:7860/sdapi/v1/ping")
            print(re)
            print("ping...")
            break
        except:
            print("Wait for API to start...")
        time.sleep(3)


def print_images_info(path_of_generated_images, num_img):
    # print
    number_of_images_in_folder = number_of_files(path_of_generated_images)
    print("Number of images in folder: ", number_of_images_in_folder)
    number_of_images_to_generate = number_of_images_in_folder
    print("Number of images to generate: ", num_img - number_of_images_to_generate)
    print("Generating images...")
    return number_of_images_to_generate


def load_api(num_img, payload_webui, is_controlnet, choice, hypernetwork, embedding, model, lora):
    """
    Load the API, generate images based on config
    parameters:
        num_img: number of images to generate
        payload_webui: payload from webui
        is_controlnet: boolean, true if we use controlnet, false if we use stable diffusion
        choice: 1 for COWC, 0 for UDACITY
        hypernetwork: string, name of the hypernetwork
        embedding: string, name of the embedding
        model: string, name of the model
    """

    api_monitor()
    time.sleep(2)
    num_img = int(num_img)

    # paths of generated images
    if is_controlnet:
        print("generate using controlnet...")
        path_of_generated_images = config['paths']['cn']
        directory_img_control = config['paths']['bp']

        print_images_info(path_of_generated_images, num_img)
        run_controlnet(payload_webui, num_img, directory_img_control, hypernetwork, embedding, model, lora)

    else:
        print("generate using stable diffusion...")
        path_of_generated_images = config['paths']['sd']

        number_of_images_to_generate = print_images_info(path_of_generated_images, num_img)

        path = config["paths"]["sd"]
        # rename every file in path from 0 to total_files, with their txt
        i = 0
        for file in sorted(os.listdir(path), key=numerical_sort):
            print(file)

            if file.endswith(".png"):
                os.rename(f"{path}{file}", f"{path}{i}.png")
                print("rename " + file + " to " + f" {i}.png")
                i += 1
            else:
                print("delete" + file)
                os.remove(f"{path}{file}")

        run_stable_diffusion(payload_webui, number_of_images_to_generate, choice, num_img, hypernetwork,
                             embedding, lora)

    print("generation done")
    print("shutting down API...")
    api_close()
