import base64
import io
from src.interface import *
import cv2
from PIL import Image
from src.prompt_generator import *
from src.yolo import *
from src.api_manager import *
import json
import requests


with open("config.json", "r") as f:
    config = json.load(f)


def run_stable_diffusion(payload_webui, number_of_images_to_generate, choice, num_img, hypernetwork, embedding, lora):
    """
    Run stable diffusion on config path, manage the files and folders
    parameters:
        payload_webui: payload from webui
        number_of_images_to_generate: number of images input by the user
        choice: 1 for COWC, 0 for UDACITY
        num_img: number of images to generate
        hypernetwork: hypernetwork to use
        embedding: embedding to use
        lora: lora to use
    """
    print("model : ", payload_webui["model"])
    # Loop to generate images
    while number_of_images_to_generate < num_img:
        print("Image number : ", number_of_images_to_generate)

        # Choose generation for COWC or UDACITY
        if choice.get() == 1:
            print("COWC")
            print("better not use realistic vision weight for this,"
                  " result are not good compared to v1-5-pruned-emaonly")
            if embedding != "":
                payload_webui["prompt"] += f" {embedding}"
            if hypernetwork != "":
                payload_webui["prompt"] += f", with style of {hypernetwork}"
            if lora != "":
                payload_webui["prompt"] += f", {lora}"
        else:
            print("UDACITY")
            payload_webui["prompt"] = prompt_generator(1)
            if embedding != "":
                payload_webui["prompt"] += f" {embedding}"
            if hypernetwork != "":
                payload_webui["prompt"] += f", with style of {hypernetwork}"
            if lora != "":
                payload_webui["prompt"] += f", {lora}"

        print(payload_webui["prompt"])

        # Save prompt and number of images to generate
        with open(config['paths']['generation_prompt_sd'], "a") as file_append:
            file_append.write(f"{payload_webui['prompt']}, {number_of_images_to_generate}\n")

        # Send request to API
        try:
            response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload_webui)
        except:
            print("Error")

        print("response: ", response)
        for i in range(len(response.json()['images'])):
            img = base64.b64decode(response.json()['images'][i])
            img = Image.open(io.BytesIO(img))
            # save
            img.save(f"{config['paths']['sd']}{number_of_images_to_generate}.png")

        number_of_images_to_generate += 1
        print("\n")
    print("done")


def run_controlnet(payload_webui, quantity_to_generate, directory_img_control, hypernetwork, embedding, model, lora):
    """
    Run controlnet on config path, manage the files and folders
    parameters:
        payload_webui: payload from webui
        quantity_to_generate: number of images to generate
        directory_img_control: directory of images to use for controlnet
        hypernetwork: hypernetwork to use
        embedding: embedding to use
        model: model to use
        lora: lora to use
    """

    print("hypernetwork: ", hypernetwork)
    print("embedding: ", embedding)
    print("model: ", model)
    print("lora: ", lora)

    path = config["paths"]["cn"]
    # rename every file in path from 0 to total_files, with their txt
    i = 0
    for file in sorted(os.listdir(path), key=numerical_sort):
        print(file)

        if file.endswith(".png"):
            os.rename(f"{path}{file}", f"{path}{i}.png")
            print("rename " + file + " to " + f" {i}.png")
            os.rename(f"{path}{file[:-4]}.txt", f"{path}{i}.txt")
            i += 1

    print("running controlnet...")
    path = config["paths"]["cn"]

    # get the highest number in dir
    files = os.listdir(path)
    files.sort()

    starting_nb = 0
    for file in files:
        if file.endswith(".png"):
            starting_nb += 1

    print("starting_nb: ", starting_nb)
    print("quantity_to_generate: ", quantity_to_generate)

    data = []

    # loop to generate images
    while starting_nb < quantity_to_generate:
        print("num_img: ", quantity_to_generate)

        # get random file in img/
        randf = random.choice(os.listdir(directory_img_control))

        file = f"{directory_img_control}{randf[:-4]}.png"
        # if file exist
        if os.path.isfile(file):
            print(".png file exist")
        else:
            file = f"{directory_img_control}{randf[:-4]}.jpg"
            print("convert to jpg")

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

        payload_webui['init_images'] = [encoded_image]

        payload_cn = {
            "init_images": [encoded_image],
            "prompt": prompt_generator(2),
            "steps": config["steps"],
            "negative_prompt": config["negative_prompt"],
            "sampler_name": "Euler",
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "module": "canny",
                            "model": model,
                        }
                    ]
                }
            }
        }

        print(embedding, hypernetwork, lora)

        print("prompt : ", payload_cn["prompt"])
        print("model :", payload_cn["alwayson_scripts"]["controlnet"]["args"][0]["model"])

        payload_cn["prompt"] += f" {embedding}"
        payload_cn["prompt"] += f", with style of {hypernetwork}"
        payload_cn["prompt"] += f", {lora}"

        img_info = []

        # Request Generation
        response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/img2img', json=payload_cn)
        print("response: ", response)
        # Read results
        r = response.json()
        # r['images'] [0] is the image,[1] represent the edge
        # if r['images'][0] exist
        if r['images'][0] is not None:
            result = r['images'][0]
        else:
            print("Error: image not generated, retrying..")
            result = ""
            pass

        image = Image.open(io.BytesIO(base64.b64decode(result)))

        # count number of files in config
        num_files = len([f2 for f2 in os.listdir(config['paths']['cn']) if f2.endswith(".png")])

        # copy txt in folder
        os.system(f"cp {file_txt} {config['paths']['cn']}{num_files}.txt")
        # save image
        image.save(f"{config['paths']['cn']}{num_files}.png")
        # shutil copy img in original folder
        # if folder not exist, create it config["paths"]["cn_img"]
        if not os.path.exists(config["paths"]["cn_img"]):
            os.makedirs(config["paths"]["cn_img"])
        shutil.copy(file, f"{config['paths']['cn_img']}{num_files}_original.png")

        img_info.append(payload_cn['prompt'])
        img_info.append(num_detection_true)
        img_info.append(num_files)
        # delete []
        img_info = str(img_info).replace("[", "").replace("]", "")

        # save in test.txt
        with open(f"{config['paths']['generation_prompt_cn']}", "a") as file_info:
            file_info.write(f"{img_info} \n")
        starting_nb += 1
        print("//////////////////////////////// \n")
    return data
