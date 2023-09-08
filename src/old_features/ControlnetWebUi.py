import base64
import io
import os
import subprocess
import time
import cv2
import requests
from PIL import Image
import src.tools as v
import src.old_features.yolact as yo
from src.prompt_generator import *

global process


def number_of_files(directory):
    # get the highest number in dir
    files = os.listdir(directory)
    files.sort()
    if len(files) == 0:
        return 0
    else:
        nb_files = 0
        for file in files:
            if file.endswith(".png"):
                nb_files += 1
        return nb_files


def cancel():
    # kill p
    process.kill()
    print("Canceled")


def generate_images(quantity_to_generate, steps_in, directory_img_control, choose_prompt):
    starting_nb = number_of_files("gen_api_out/img_controlnet/")
    print("starting_nb: ", starting_nb)
    print("sum: ", quantity_to_generate + starting_nb)
    data = []

    # loop to generate images
    for i in range(quantity_to_generate):
        print("i: ", i)
        print("num_img: ", quantity_to_generate)

        url = "http://127.0.0.1:7860"

        # get random file in img/
        randf = random.choice(os.listdir(directory_img_control))

        file = f"{directory_img_control}{randf[:-4]}.png"
        # if file exist
        if os.path.isfile(file):
            print("file exist")
        else:
            file = f"{directory_img_control}{randf[:-4]}.jpg"
        file_txt = f"{directory_img_control}{randf[:-4]}.txt"
        print("file: ", file)
        print("file_txt: ", file_txt)

        # count number of detection in file_txt
        num_detection_true = len(open(file_txt).readlines())
        print("num_detection: ", num_detection_true)

        # Read Image in RGB order
        img = cv2.imread(file)[:, :, ::-1]

        # Encode into PNG and send to ControlNet
        retval, bytes_encode = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(bytes_encode).decode('utf-8')

        # payload
        payload = {
            "init_images": [encoded_image],
            "prompt": 'a picture of a street with cars and people, realistic',
            "steps": 20,
            "negative_prompt": "bad proportion, distortion, bad quality, bad focus, blurry, bad lighting,"
                               "bad compression, bad artifact, bad pixel,deformed iris, deformed pupils, "
                               "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, "
                               "close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, "
                               "duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, "
                               "poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
                               "bad proportions, extra limbs, cloned face, disfigured, gross proportions, "
                               "malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, "
                               "too many fingers, long neck, no cars, no people, no buildings, no street",
            "sampler_name": "Euler",
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "module": "canny",
                            "model": "control_v11p_sd15_canny [d14c016b]",
                        }
                    ]
                }
            }
        }


        img_info = []
        # We change prompt to random and steps to input user input
        payload['prompt'] = prompt_generator(choose_prompt)
        payload['steps'] = steps_in

        # Request Generation
        response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
        print("response: ", response)
        # Read results
        r = response.json()
        # r['images'] [0] is the image,[1] represent the edge
        result = r['images'][0]

        image = Image.open(io.BytesIO(base64.b64decode(result)))

        # count number of files in gen_api_out/img_controlnet/
        num_files = len([f for f in os.listdir("gen_api_out/img_controlnet/") if f.endswith(".png")])

        # copy txt in folder
        os.system(f"cp {file_txt} gen_api_out/img_controlnet/{num_files}.txt")
        # save image
        image.save(f'gen_api_out/img_controlnet/{num_files}.png')

        img_info.append(payload['prompt'])
        img_info.append(num_detection_true)

        # count number of detection in result.txt
        path_result = f"result.txt"
        # open and read number of line
        num_final = 0
        with open(path_result, "r") as f:
            for line in f:
                "if line start with 0 or 1"
                if line[0] == "0" or line[0] == "1":
                    num_final += 1
        print("/////////////////////////")
        print("num_final: ", num_final)
        img_info.append(num_final)
        num_in = str(num_files) + ".png"
        img_info.append(num_in)

        print("num_detection: ", num_detection_true)
        print("/////////////////////////")
        # if num_detection  is at least 0.8 of num_final
        if num_final >= int(0.8 * num_detection_true):
            print("kept")
            img_info.append("kept")
        else:
            print("removed")
            img_info.append("removed")

        # write in test.txt
        with open("test.txt", "a") as f:
            # write img_info as 1 string in 1 line
            f.write(str(img_info) + "\n")

        print("//////////////////////////////// \n")
        i += 1

    cancel()
    time.sleep(3)
    yo.run_yolact_cn("gen_api_out/img_controlnet")
    return data


def use_api(num_img, steps_in):
    """
    Launch API if not already launched and generate images, at the end of the generation generate image so that
    car = pedestrians

    Parameters
    ----------
    num_img : int, number of images to generate
    steps_in : int, number of steps to generate an image
    -------

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

    # print(type(payload))
    steps_in = int(steps_in)
    num_img = int(num_img)

    data_evaluation = generate_images(num_img, steps_in, "img/", 1)

    repartition = v.class_repartition("gen_api_out/img_controlnet/")

    while repartition[0] > repartition[1]:
        repartition = v.class_repartition("gen_api_out/img_controlnet/")
        # int
        print("repartition: ", repartition)
        # repartition to int
        repartition = [int(i) for i in repartition]

        print("repartition: ", repartition)
        generate_images(1, steps_in, "pedestrianscompensation/", 2)
    print(data_evaluation)
