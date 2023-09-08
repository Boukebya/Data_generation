import base64
import io
import os
import shutil
import subprocess
import time
import cv2 as cv
import requests
from PIL import Image

global process


def display_bbox(path):
    """
    Display the bounding box of images in a folder (1 txt for 1 img), (yolo format)
    :param path: Path of the folder containing the images and txt
    """

    # for each img in the folder, find the corresponding txt, and display the bounding box on the img
    for img in sorted(os.listdir(path)):
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
            cv.waitKey(0)
            cv.destroyAllWindows()


def cancel():
    """
    Cancel the API
    """
    process.kill()
    print("Canceled")


def yolo_run():
    """
    Run yolo on gen_api_out/img, manage the files and folders
    """
    for file in os.listdir("yolo/yolov7/temp_copy"):
        os.remove(f"yolo/yolov7/temp_copy/{file}")
    # copy gen_api_out/img to yolo/yolov7/temp_copy
    for file in os.listdir("gen_api_out/img"):
        shutil.copy(f"gen_api_out/img/{file}", "yolo/yolov7/temp_copy")
    os.chdir("yolo/yolov7")
    os.system(f"python detect.py --weights bestCOWC.pt --conf 0.25 --img-size 512 --source temp_copy/ --save-txt")
    # copy content of runs/detect/exp/labels to gen_api_out/img
    os.chdir("../../..")
    for file in os.listdir("yolo/yolov7/runs/detect/exp/labels"):
        shutil.copy(f"yolo/yolov7/runs/detect/exp/labels/{file}", "gen_api_out/img")
    # replace 1st character of each file by 0
    for file in os.listdir("gen_api_out/img"):
        if file.endswith(".txt"):
            with open(f"gen_api_out/img/{file}", "r+") as f:
                lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = "0" + lines[i][1:]
            with open(f"gen_api_out/img/{file}", "w") as f:
                f.writelines(lines)
    # remove folder exp
    shutil.rmtree("yolo/yolov7/runs/detect/exp")


def gen_api(payload_webui, quantity_to_generate, is_random):
    """
    Generate images with the API, the images are saved in gen_api_out/img
    :param payload_webui: parameters for API generation
    :param quantity_to_generate: number of images to generate
    :param is_random: True if random, False if not
    :return:
    """
    is_up = True

    # send a ping request to the API until it is up, allow us to wait for the server to start
    try:
        requests.get(url="http://127.0.0.1:7860/sdapi/v1/ping")
        print("ping...")
    except:
        is_up = False

    print(is_up)

    if is_up is False:
        print("API is not up")
        print("Launching API...")
        global process
        process = subprocess.Popen(["python", "stable-diffusion-webui/launch.py", "--api"])
    else:
        print("API is up")

    # send a ping request to the API until it is up, allow us to wait for the server to start
    while True:
        try:
            requests.get(url="http://127.0.0.1:7860/sdapi/v1/ping")
            print("ping...")
            break
        except:
            pass
        time.sleep(1)


    # get number of images in gen_api_out/img .png
    num_files = len([f for f in os.listdir("gen_api_out/img") if f.endswith(".png")])
    print("number of images already created: ", num_files)

    z = num_files
    while z < quantity_to_generate + num_files:
        payload_webui['prompt'] = "pictures of cars *aerial , with style of <hypernet:aerial_style:1>"

        response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload_webui)

        for i in range(len(response.json()['images'])):
            img = base64.b64decode(response.json()['images'][i])
            img = Image.open(io.BytesIO(img))
            # save
            img.save(f"gen_api_out/img/{z}.png")
            z += 1
    cancel()
