import tkinter as tk
from threading import Thread
from tkinter import filedialog
from tkinter import ttk
from src import tools as tools
from src.api_manager import *
from src.convert import convert_file
from src.cos_similarity import crop_folder, cos_sim
from src.data_augmentation import augment_folder
from src.yolo import *
from src.statistics_displayer import *
from src.yolo_iou_manager import *
from src.diffuser_training import train_stable

with open("config.json", "r") as f:
    config = json.load(f)


# use filedialog to create a research in folder button
def get_path():
    """
    Get path from filedialog
    :return: path
    """
    path = filedialog.askdirectory()
    print(path)
    return path


# update button for console
def update_terminal(root, temp, out):
    """
    Update the console in the interface
    :param root: root of the interface
    :param temp: temp of the interface
    :param out: out of the interface
    """
    # if change in sys.stdout then update out
    if temp.getvalue() != out.get("1.0", "end-1c"):
        out.delete("1.0", "end-1c")
        out.insert("1.0", temp.getvalue())
        out.see("end")

    root.after(100, update_terminal, root, temp, out)


# create button class to get value
class Butt:
    """
    Create a button with text and command
    :param master: master of the button
    :param text: text of the button
    :param command: command of the button
    :param pos_x: position x of the button
    :param posy: position y of the button
    """

    def __init__(self, master, text, command, pos_x, posy):
        self.master = master
        self.text = text
        self.command = command
        self.position = (pos_x, posy)
        self.button = ttk.Button(self.master, text=self.text, command=self.command)
        self.button.grid(column=self.position[0], row=self.position[1], padx=30, pady=30)
        self.value = 0

    def set_value(self):
        self.value = filedialog.askdirectory()
        print(self.value)
        return self.value

    def get_value(self):
        return self.value


# create input class with text and input box
class Input:
    """
    Create an input with text and input box, .get_value() to get the value of the input
    :param master: master of the input
    :param text: text of the input
    :param pos_x: position x of the input
    :param pos_y: position y of the input
    """

    def __init__(self, master, text, pos_x, pos_y):
        self.master = master
        self.text = text
        self.position = (pos_x, pos_y)
        self.input = ttk.Entry(self.master, width=12)
        self.input.grid(column=self.position[0], row=(self.position[1] + 1), padx=30, pady=10)

        # text
        self.label = ttk.Label(self.master, text=self.text)
        self.label.grid(column=self.position[0], row=(self.position[1]), padx=30, pady=0)
        self.value = 0

    def set_value(self):
        self.value = self.input.get()
        print(self.value)
        return self.value

    def get_value(self):
        return self.value


# create an output class with text
class Output:
    """
    Create an output with text, .set_value() to set the value of the output
    :param master: master of the output
    :param text: text of the output
    :param pos_x: position x of the output
    :param pos_y: position y of the output
    """

    def __init__(self, master, text, pos_x, pos_y):
        self.master = master
        self.text = text
        self.position = (pos_x, pos_y)
        self.label = ttk.Label(self.master, text=self.text)
        self.label.grid(column=self.position[0], row=self.position[1], padx=30, pady=0)
        self.value = 0

    def set_value(self, value):
        self.value = value
        print(self.value)
        return self.value

    def get_value(self):
        return self.value


# ////////////////////////////////////////////////////////////////////////////// TABS


def cosim_tab(tab_cosim):
    """
    Create the tab for the image generation using stable diffusion webui
    Parameters:
         tab_cosim: tab for the image generation
    """

    def crop_cn():
        crop_folder(config["paths"]["cn"], "generated_images/cropped_images/")

    def cosim():
        cos_sim("generated_images/cropped_images/")

    # btn to train
    Butt(tab_cosim, "Crop detection of controlnet images", lambda: crop_cn(), 0, 4)

    Butt(tab_cosim, "Do cosim between cropped CN images and blueprint images", lambda: cosim(), 0, 5)


def training_tab(tab_training):
    """
    Create the tab for the image generation using stable diffusion webui
    Parameters:
         tab_training: tab for the image generation
    """

    def load_convert_tab(button_path_img_to_load):
        convert_file(button_path_img_to_load.get_value(), "result.safetensors")

    def train_stablediffusion():
        print("don't forget to create a folder named from config file training_images "
              "in the root of the project, training config can be modified in diffuser_training.py")
        thread = Thread(target=train_stable, args=(config["paths"]["training_images"]))
        thread.start()

    # btn class
    button_path_img = Butt(tab_training, "Path of .bin to convert", lambda: button_path_img.set_value(), 0, 1)

    # btn to train
    Butt(tab_training, "convert to safe-tensor", lambda: load_convert_tab(button_path_img), 0, 4)

    Butt(tab_training, "train diffusion models with img in nest", lambda: train_stablediffusion(), 0, 5)


def stable_diffusion_generation_tab(tab_generation):
    """
    Create the tab for the image generation using stable diffusion webui
    Parameters:
         tab_generation: tab for the image generation
    """

    def load_gen_sd(payload_sd, quantity_sd, number_of_steps, weight_choice, hypernetwork_choice, embedding_choice,
                    model_choice, lora_choice):

        hypernetwork_choice = hypernetwork_choice.get()
        # get value of the key of the hypernetwork
        for key_hypernetwork, value_hypernetwork in config["weights"]["hyper-networks"].items():
            if key_hypernetwork == hypernetwork_choice:
                hypernetwork_choice = value_hypernetwork
            else:
                hypernetwork_choice = ""

        embedding_choice = embedding_choice.get()
        # get value of the key of the embedding
        for key_embedding, value_embedding in config["weights"]["embeddings"].items():
            if key_embedding == embedding_choice:
                embedding_choice = value_embedding
            else:
                embedding_choice = ""

        model_choice = model_choice.get()
        for key_model, value_model in config["models"].items():
            if key_model == model_choice:
                model_choice = value_model

        lora_choice = lora_choice.get()
        for key_lora, value_lora in config["lora"].items():
            if key_lora == lora_choice:
                lora_choice = value_lora
            else:
                lora_choice = ""

        payload_sd["model"] = model_choice

        print("hypernetwork : ", hypernetwork_choice)
        print("embedding : ", embedding_choice)
        print("model : ", model_choice)
        print("lora : ", lora_choice)
        quantity_sd.set_value()
        number_of_steps.set_value()

        payload_sd["steps"] = number_of_steps.get_value()
        thread = Thread(target=load_api, args=(
            quantity_sd.get_value(), payload_sd, 0, weight_choice, hypernetwork_choice, embedding_choice,
            model_choice, lora_choice))
        thread.start()

    options_list3 = []
    for key, value in config["models"].items():
        options_list3.append(key)
    value_inside3 = tk.StringVar(tab_generation)
    value_inside3.set("Select model")
    question_menu = tk.OptionMenu(tab_generation, value_inside3, *options_list3,
                                  command=lambda x: print(value_inside3.get()))
    question_menu.grid(column=3, row=1, padx=30, pady=30)

    options_list = []
    for key, value in config["weights"]["hyper-networks"].items():
        options_list.append(key)
    value_inside = tk.StringVar(tab_generation)
    value_inside.set("Select hypernetwork")
    question_menu = tk.OptionMenu(tab_generation, value_inside, *options_list,
                                  command=lambda x: print(value_inside.get()))
    question_menu.grid(column=3, row=2, padx=30, pady=30)

    options_list2 = []
    for key, value in config["weights"]["embeddings"].items():
        options_list2.append(key)
    value_inside2 = tk.StringVar(tab_generation)
    value_inside2.set("Select embedding")
    question_menu = tk.OptionMenu(tab_generation, value_inside2, *options_list2,
                                  command=lambda x: print(value_inside2.get()))
    question_menu.grid(column=3, row=3, padx=30, pady=30)

    options_list4 = []
    for key, value in config["lora"].items():
        options_list4.append(key)
    value_inside4 = tk.StringVar(tab_generation)
    value_inside4.set("Select lora")
    question_menu = tk.OptionMenu(tab_generation, value_inside4, *options_list4,
                                  command=lambda x: print(value_inside4.get()))
    question_menu.grid(column=3, row=4, padx=30, pady=30)

    # input for number of images
    nb_img = Input(tab_generation, "Number of images to generate", 0, 1)
    steps = Input(tab_generation, "Steps", 0, 3)

    # check mark for random prompt
    choice = tk.IntVar()
    choice_check = ttk.Checkbutton(tab_generation, text="Check if image generation is for COWC", variable=choice)
    choice_check.grid(column=0, row=0, padx=30, pady=30)
    # if check mark change, print it
    choice.trace("w", lambda name, index, mode, choice_cowc=choice: print(choice_cowc.get()))

    payload = {
        "prompt": "pictures of cars *aerial , with style of <hypernet:aerial_style:1>",
        "steps": config["steps"],
        "width": 512,
        "height": 512,
        "scale": 7,
        "batch_size": 1,
        "n_iter": 1,
        "seed": -1,
        "sampling": "euler",
        "model": "realisticVisionV30_VAE",
        "negative_prompt": config["negative_prompt"],
        "vae": "paragonV10_v10VAE"
    }

    # btn on click launch vis.test()
    Butt(tab_generation, "Generate",
         lambda: load_gen_sd(payload, nb_img, steps, choice, value_inside,
                             value_inside2, value_inside3, value_inside4),
         1, 5)
    # btn cancel
    Butt(tab_generation, "shutdown API", lambda: api_close(), 0, 5)


def controlnet_tab(tab_controlnet):
    """
    Create the tab for the controlnet generation
    Parameters:
            tab_controlnet: tab for the controlnet generation
    """

    def load_api_cn(number_of_images, hypernetwork_choice, embedding_choice, model_choice, lora_choice):
        number_of_images.set_value()

        hypernetwork_choice = hypernetwork_choice.get()
        # get value of the key of the hypernetwork
        for key_hypernetwork, value_hypernetwork in config["weights"]["hyper-networks"].items():
            if key_hypernetwork == hypernetwork_choice:
                hypernetwork_choice = value_hypernetwork
            else:
                hypernetwork_choice = ""

        embedding_choice = embedding_choice.get()
        # get value of the key of the embedding
        for key_embedding, value_embedding in config["weights"]["embeddings"].items():
            if key_embedding == embedding_choice:
                embedding_choice = value_embedding
            else:
                embedding_choice = ""

        model_choice = model_choice.get()
        for key_model, value_model in config["models_cn"].items():
            print(key_model, value_model)
            if key_model == model_choice:
                model_choice = value_model
                print("model choice", model_choice)

        lora_choice = lora_choice.get()
        for key_lora, value_lora in config["lora"].items():
            if key_lora == lora_choice:
                lora_choice = value_lora
            else:
                lora_choice = ""

        payload = {}

        thread = Thread(target=load_api, args=(
            number_of_images.get_value(), payload, 1, 0, hypernetwork_choice,
            embedding_choice, model_choice, lora_choice))
        thread.start()

    def delete_controlnet():
        """
        Delete the controlnet files
        """
        path = config['paths']['cn']
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        # empty yolo file
        with open(config['paths']['generation_prompt_cn'], 'w') as file:
            file.write("")

    options_list3 = []
    for key, value in config["models_cn"].items():
        options_list3.append(key)
    value_inside3 = tk.StringVar(tab_controlnet)
    value_inside3.set("Select model")
    question_menu = tk.OptionMenu(tab_controlnet, value_inside3, *options_list3,
                                  command=lambda x: print(value_inside3.get()))
    question_menu.grid(column=3, row=1, padx=30, pady=30)

    options_list = []
    for key, value in config["weights"]["hyper-networks"].items():
        options_list.append(key)
    value_inside = tk.StringVar(tab_controlnet)
    value_inside.set("Select hypernetwork")
    question_menu = tk.OptionMenu(tab_controlnet, value_inside, *options_list,
                                  command=lambda x: print(value_inside.get()))
    question_menu.grid(column=3, row=2, padx=30, pady=30)

    options_list2 = []
    for key, value in config["weights"]["embeddings"].items():
        options_list2.append(key)
    value_inside2 = tk.StringVar(tab_controlnet)
    value_inside2.set("Select embedding")
    question_menu = tk.OptionMenu(tab_controlnet, value_inside2, *options_list2,
                                  command=lambda x: print(value_inside2.get()))
    question_menu.grid(column=3, row=3, padx=30, pady=30)

    options_list4 = []
    for key, value in config["lora"].items():
        options_list4.append(key)
    value_inside4 = tk.StringVar(tab_controlnet)
    value_inside4.set("Select lora")
    question_menu = tk.OptionMenu(tab_controlnet, value_inside4, *options_list4,
                                  command=lambda x: print(value_inside4.get()))
    question_menu.grid(column=3, row=4, padx=30, pady=30)

    number_images = Input(tab_controlnet, "Number of images to generate", 0, 1)

    # btn cancel
    Butt(tab_controlnet, "Shutdown API", lambda: api_close(), 0, 5)

    Butt(tab_controlnet, "Generate",
         lambda: load_api_cn(number_images, value_inside, value_inside2, value_inside3, value_inside4), 1,
         5)

    Butt(tab_controlnet, "Delete Controlnet files", lambda: delete_controlnet(), 1, 2)


def tools_tab(tab_yolo):
    """
    Launch comparison between CN and yolo
    Parameters:
            tab_yolo: tab for the comparison between CN and yolo
    """

    def comparison_butt(threshold):
        """
        Launch comparison between CN and yolo
        """
        threshold.set_value()
        threshold = float(threshold.get_value())
        print("threshold =", threshold)

        thread = Thread(target=compare, args=(config["paths"]["cn"], config["paths"]["yolo_cn_lines"], threshold))
        thread.start()

    # input
    threshold = Input(tab_yolo, "Threshold for iou", 0, 0)

    # btn to train
    btn = ttk.Button(tab_yolo, text="Launch comparison between CN and yolo",
                     command=lambda: comparison_butt(threshold))
    btn.grid(column=0, row=2, padx=30, pady=30)


def analyze_tab(tab_analyze):
    """
    Create the tab for the class analysis, allow to see the repartition of the classes in the dataset
    Parameters:
            tab_analyze: tab for the class analysis
    """

    def analyse_butt():
        val = tools.class_repartition(button_path.get_value())
        # output
        Output(tab_analyze, f"car: {val[0]} pedestrian: {val[1]} others: {val[2]}", 0, 5)

    # Butt
    button_path = Butt(tab_analyze, "Dataset to analyse", lambda: button_path.set_value(), 0, 1)
    Butt(tab_analyze, "Analyze", lambda: analyse_butt(), 1, 1)


def merge_tab(tab_add):
    """
    Create the tab for the data merger
    Parameters:
            tab_add: tab for the data merger
    """

    def load_add_tab(button_path_img_to_load, button_path_dataset_to_load, ):
        print("img =", button_path_img_to_load.get_value(), "dataset =", button_path_dataset_to_load.get_value())
        tools.add_quantity(button_path_img_to_load.get_value(), button_path_dataset_to_load.get_value())

    # btn class
    button_path_img = Butt(tab_add, "Path to image to add", lambda: button_path_img.set_value(), 0, 1)

    button_path_dataset = Butt(tab_add, "Original dataset", lambda: button_path_dataset.set_value(), 1, 1)

    # btn to train
    Butt(tab_add, "Add", lambda: load_add_tab(button_path_img, button_path_dataset), 0, 4)


def add_random_tab(tab_get_random):
    """
    Create the tab for the random data generator
    Parameters:
            tab_get_random: tab for the random data generator

    """

    def load_add_random():
        """
        Load value for random data getter
        """
        nb_img.set_value()
        print("img =", button_path_img.get_value())
        print("nb_img =", nb_img.get_value())

        tools.add_random(button_path_img.get_value(), nb_img.get_value())

    button_path_img = Butt(tab_get_random, "Path to image to add", lambda: button_path_img.set_value(), 0, 1)

    nb_img = Input(tab_get_random, "Number of images to clone", 0, 2)

    # btn to train
    Butt(tab_get_random, "Add", lambda: load_add_random(), 0, 4)


def run_yolo(tab_run_yolo):
    """
    Create the tab for the yolo run
    Parameters:
            tab_run_yolo: tab for the yolo run
    """

    def load_yolo(choice, choice2):
        """
        Run yolo on the generated images and create txt files
        """
        if choice.get() == 1:
            weight = "bestCOWC.pt"
        else:
            weight = "yolov7.pt"
        # delete all txt files in the folder

        path = config["paths"]["sd"]
        if choice2.get() == 1:
            path = config["paths"]["cn"]

        if path == config["paths"]["sd"]:
            for file in os.listdir(path):
                if file.endswith(".txt"):
                    os.remove(os.path.join(path, file))

        nb_img.set_value()

        thread = Thread(target=yolo_run, args=(weight, path, nb_img.get_value()))
        thread.start()

    # check mark for random prompt
    check_mark_weight = tk.IntVar()
    choice_check = ttk.Checkbutton(tab_run_yolo, text="Check if image generation is for COWC",
                                   variable=check_mark_weight)
    choice_check.grid(column=0, row=0, padx=30, pady=30)

    nb_img = Input(tab_run_yolo, "Confidence for yolo", 0, 2)

    # if check mark change, print it
    check_mark_weight.trace("w", lambda name, index,
                            mode, choice_for_weight=check_mark_weight: print(choice_for_weight.get()))

    # check mark for random prompt
    check_mark_model = tk.IntVar()
    check_for_model = ttk.Checkbutton(tab_run_yolo, text="Check if image generation is for controlnet",
                                      variable=check_mark_model)
    check_for_model.grid(column=0, row=1, padx=30, pady=30)
    # if check mark change, print it
    check_mark_model.trace("w", lambda name, index, mode, choice=check_mark_model: print(check_mark_model.get()))

    Butt(tab_run_yolo, "Run yolo", lambda: load_yolo(check_mark_weight, check_mark_model), 0, 4)


def statistic_tab(tab_run_statistic):
    """
    Create the tab for the statistic
    Parameters:
            tab_run_statistic: tab for the statistic
    """

    def run_stat(choice_value, threshold):
        """
        Run the statistic on the generated images
        """
        threshold.set_value()
        choice_value = choice_value.get()
        if choice_value == 0:
            print("Running statistic on controlnet")
            path = config["paths"]["generation_prompt_cn"]
            prediction = config["paths"]["yolo_cn"]
        else:
            print("Running statistic on Stable diffusion")
            path = config["paths"]["generation_prompt_sd"]
            prediction = config["paths"]["yolo_sd"]

        threshold.set_value()
        thread = Thread(target=statistic, args=(path, prediction, float(threshold.get_value())))
        thread.start()

    # check mark for random prompt
    choice = tk.IntVar()
    choice_check = ttk.Checkbutton(tab_run_statistic, text="Check if image generation is for stable diffusion",
                                   variable=choice)
    choice_check.grid(column=0, row=0, padx=30, pady=30)
    # if check mark change, print it
    choice.trace("w", lambda name, index, mode, choice_check_stable=choice: print(choice_check_stable.get()))

    # input for threshold
    threshold = Input(tab_run_statistic, "Threshold to accept detections", 0, 1)

    # btn class
    Butt(tab_run_statistic, "display statistic", lambda: run_stat(choice, threshold), 0, 3)


def display_bbox_tab(tab_display_bbox):
    """
    Create the tab for the display bbox
    Parameters:
            tab_display_bbox: tab for the display bbox
    """

    def load_displayer(button_path_img_to_load):
        print("img =", button_path_img_to_load.get_value())
        tools.display_bbox(button_path_img_to_load.get_value())
        thread = Thread(target=tools.display_bbox, args=(button_path_img_to_load.get_value()))
        thread.start()

    # btn class
    button_path_img = Butt(tab_display_bbox, "Path to dataset", lambda: button_path_img.set_value(), 0, 1)

    # btn to train
    Butt(tab_display_bbox, "Display (x to cancel, any other key to show next image)",
         lambda: load_displayer(button_path_img), 0, 4)


def data_augmentation_tab(tab_data_augmentation):
    """
    Create the tab for the data augmentation
    Parameters:
            tab_data_augmentation: tab for the data augmentation
    """

    def load_data_augmentation(path, path_augmented_folder, choice_path):
        if choice_path == 1:
            path = config["paths"]["sd"]

        augment_folder(path, path_augmented_folder)

    path_img = config["paths"]["cn"]
    path_augmented = config["paths"]["augmented"]

    # check mark for random prompt
    choice = tk.IntVar()
    choice_check = ttk.Checkbutton(tab_data_augmentation, text="Check if image augmentation is for stable diffusion",
                                   variable=choice)
    choice_check.grid(column=0, row=0, padx=30, pady=30)
    # if check mark change, print it
    choice.trace("w", lambda name, index, mode, choice_check_stable=choice: print(choice_check_stable.get()))

    # btn to train
    Butt(tab_data_augmentation, "Data augmentation on images",
         lambda: load_data_augmentation(path_img, path_augmented, choice.get()), 0, 4)
