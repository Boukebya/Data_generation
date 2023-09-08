
'''
# Previous code
def delete_c6c2():
    """
    Delete c6c2 in PRW dataset
    :return: nothing, delete c6c2 in annotations and frames
    """
    # if file start with c6c2, delete it in prw dataset, annotations and frames
    for file in os.listdir("PRW-v16.04.20/annotations/"):
        if file.startswith("c6s1"):
            os.remove("PRW-v16.04.20/annotations/" + file)
            os.remove("PRW-v16.04.20/frames/" + file[:-4])
            print("delete " + file)


def csv_dataset(CUSTOM_DATASET_PATH, CUSTOM_DATASET_NAME, DATASET_PATH, DATASET_CSV_PATH):
    """
    Create a dataset for Yolov7 with udacity dataset, csv file
    :param CUSTOM_DATASET_PATH:  path to the folder where you want to create the dataset
    :param CUSTOM_DATASET_NAME:  name of the folder where you want to create the dataset
    :param DATASET_PATH:      path to the dataset
    :param DATASET_CSV_PATH:  path to the csv file
    :return:                nothing, create a dataset for Yolov7
    """
    # if file named custom does not exist, create it
    custom_path = CUSTOM_DATASET_PATH + CUSTOM_DATASET_NAME
    if not os.path.exists(custom_path):
        os.makedirs(CUSTOM_DATASET_PATH + CUSTOM_DATASET_NAME)

    # get _annotations.csv in yolov7/udacity to have a data file to work with
    with open(DATASET_CSV_PATH, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Variables
    nb_img = 0
    nb_data = len(data)
    img_name = data[1][0]
    img_txt = ""

    # Loop to create .txt, start at 1 because 0 is the name of the columns
    for iterations in range(1, nb_data):
        # verify if data is empty

        if data[iterations]:

            if data[iterations][0] == img_name:

                # DO THIS TO ADD A LINE IN TXT
                cl = data[iterations][3]
                if cl == "car":
                    cl = "0"
                elif cl == "pedestrian":
                    cl = "1"
                else:
                    cl = "2"

                # Convert to YOLO format : cl center_x center_y width height
                x_min = data[iterations][4]
                y_min = data[iterations][5]
                x_max = data[iterations][6]
                y_max = data[iterations][7]

                x_min = float(x_min) / 512
                y_min = float(y_min) / 512
                x_max = float(x_max) / 512
                y_max = float(y_max) / 512

                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                # img_text append all txt
                if cl != "2":
                    img_txt += (cl + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(
                        height) + "\n")
                img_name = data[iterations][0]

            # DO THIS TO CREATE A FILE
            else:
                print("image number : " + str(nb_img))
                # print(img_txt)

                # write a txt named i.txt in custom folder, class center_x center_y width height
                with open(str(nb_img) + ".txt", 'w') as file:
                    file.write(img_txt)

                # copy image in custom folder
                img = cv.imread(DATASET_PATH + img_name)
                cv.imwrite(str(nb_img) + ".jpg", img)

                # Increment nb_img and reset img_txt and img_name
                nb_img += 1
                img_txt = ""
                img_name = data[iterations][0]

    # divide data in train and val
    # create train and valis folder
    if not os.path.exists(custom_path + "/train"):
        os.makedirs(custom_path + "/train")
    if not os.path.exists(custom_path + "/val"):
        os.makedirs(custom_path + "/val")

    # get 80% of data for train and 20% for val
    nb_train = int(nb_img * 0.8)

    # put files in train or val
    for i in range(nb_train):
        # copy txt and jpg in train folder
        os.rename(str(i) + ".txt", CUSTOM_DATASET_NAME + "/train/" + str(i) + ".txt")
        os.rename(str(i) + ".jpg", CUSTOM_DATASET_NAME + "/train/" + str(i) + ".jpg")
        print("moving " + str(i) + " to train folder")
    for i in range(nb_train, nb_data):
        # if file exist
        if os.path.exists(str(i) + ".txt"):
            # copy txt and jpg in val folder
            os.rename(str(i) + ".txt", CUSTOM_DATASET_NAME + "/val/" + str(i) + ".txt")
            os.rename(str(i) + ".jpg", CUSTOM_DATASET_NAME + "/val/" + str(i) + ".jpg")
            print("moving " + str(i) + " to val folder")

    print("creation of data.yaml")
    with open(CUSTOM_DATASET_NAME + "/" + "data.yaml", 'w') as file:
        file.write("train: ../yolo/" + custom_path + "/train\n")
        file.write("val: ../yolo/" + custom_path + "/val\n")
        file.write("\n")
        file.write("nc: 2\n")
        file.write("names: ['car', 'pedestrian']\n")

    # clear memory
    torch.cuda.memory_summary(device=None, abbreviated=False)


def mat_dataset(CUSTOM_DATASET_PATH, CUSTOM_DATASET_NAME, DATASET_ANNOTATIONS_PATH, DATASET_FRAMES_PATH):
    """
    function to add pedestrian of PRW dataset to custom dataset
    PRW dataset is in format (1 folder with all images, 1 folder with all annotations .mat)
    :param CUSTOM_DATASET_PATH:  path to custom dataset
    :param CUSTOM_DATASET_NAME:   name of custom dataset
    :param DATASET_ANNOTATIONS_PATH:   path to annotations folder
    :param DATASET_FRAMES_PATH:   path to frames folder
    :return:   nothing
    """
    bboxes = []
    i = 0

    new_dataset_path = CUSTOM_DATASET_PATH + CUSTOM_DATASET_NAME + "/"
    # create folder
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # for each file in PRW-v16.04.20/annotations/
    for file in os.listdir(DATASET_ANNOTATIONS_PATH):
        # read box_new
        mat = sio.loadmat(DATASET_ANNOTATIONS_PATH + file)
        # print(mat)

        # check if keys box_new or anno_file exist
        if "box_new" in mat:
            bboxes = mat["box_new"]
        elif "anno_file" in mat:
            bboxes = mat["anno_file"]
            print("anno_file")

        txt = ""
        # for each line in bboxes
        for line in bboxes:
            x = line[1]
            y = line[2]
            w = line[3]
            h = line[4]

            x = x / 1920
            y = y / 1080
            w = w / 1920
            h = h / 1080

            center_x = x + w / 2
            center_y = y + h / 2
            width = w
            height = h

            # add to txt
            txt += ("1 " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n")

        print("image number :" + str(i) + "\n" + txt + "\n")

        # create file
        f = open(new_dataset_path + str(i) + ".txt", "w")
        # write txt in file
        f.write(txt)
        f.close()

        # add image
        img = cv.imread(DATASET_FRAMES_PATH + file[:-4])
        cv.imwrite(new_dataset_path + str(i) + ".jpg", img)

        i += 1

    nb_img = 8918
    # get 80% of data for train and 20% for val
    nb_train = int(nb_img * 0.8)

    # create train and val folder
    if not os.path.exists(new_dataset_path + "train"):
        os.makedirs(new_dataset_path + "train")
    if not os.path.exists(new_dataset_path + "val"):
        os.makedirs(new_dataset_path + "val")

    # put files in train or val
    for i in range(nb_train):
        # copy txt and jpg in train folder
        os.rename(new_dataset_path + str(i) + ".txt",
                  new_dataset_path + "train/" + str(i) + ".txt")
        os.rename(new_dataset_path + str(i) + ".jpg",
                  new_dataset_path + "train/" + str(i) + ".jpg")
        print("moving " + str(i) + " to train folder")

    for i in range(nb_train, nb_img):
        # if file exist
        if os.path.exists(new_dataset_path + str(i) + ".txt"):
            # copy txt and jpg in val folder
            os.rename(new_dataset_path + str(i) + ".txt",
                      new_dataset_path + "val/" + str(i) + ".txt")
            os.rename(new_dataset_path + str(i) + ".jpg",
                      new_dataset_path + "val/" + str(i) + ".jpg")
            print("moving " + str(i) + " to val folder")

    print("creation of data.yaml")
    with open(new_dataset_path + "data.yaml", 'w') as file:
        file.write("train: ../" + new_dataset_path + "train\n")
        file.write("val: ../" + new_dataset_path + "val\n")
        file.write("\n")
        file.write("nc: 1\n")
        file.write("names: ['pedestrian']\n")

    # clear memory
    torch.cuda.memory_summary(device=None, abbreviated=False)


def add_to_dataset(path_to_folder, path_to_dataset):
    """
    Function to get files from a folder, and add it to another file with number management (name of file will start
    at 100 if dataset is 99 images) It allow to add a dataset into another Yolo format (1 image for 1 txt) :param
    Path_to_folder:   path to folder with files to add :param Path_to_dataset:  path to dataset :return:  nothing
    """
    path_to_folder = path_to_folder + "/"
    path_to_dataset = path_to_dataset + "/"

    print(path_to_folder, path_to_dataset)
    nb_files = 0
    # count how many files are in dataset
    for file in os.listdir(path_to_dataset):
        if file.endswith(".txt"):
            nb_files += 1
    nb_files = nb_files
    print(nb_files, "files in dataset")
    nb_files2 = 0
    # count how many files are in folder
    for file in os.listdir(path_to_folder):
        if file.endswith(".txt"):
            nb_files2 += 1
    nb_files2 = nb_files2
    print(nb_files2, "files in folder")

    i = 0
    for file in os.listdir(path_to_folder):
        print(file)
        # move file to path_to_dataset with value 1+nv_files
        if file.endswith(".png"):
            shutil.copy(path_to_folder + file, path_to_dataset + str(i + 1 + nb_files) + ".png")

            file_img = file[:-3] + "txt"
            if os.path.exists(path_to_folder + file_img):
                shutil.copy(path_to_folder + file_img, path_to_dataset + str(i + 1 + nb_files) + ".txt")
            i += 1
            print("adding " + file + " to dataset")
        elif file.endswith(".jpg"):
            shutil.copy(path_to_folder + file, path_to_dataset + str(i + 1 + nb_files) + ".jpg")

            file_img = file[:-3] + "txt"
            if os.path.exists(path_to_folder + file_img):
                shutil.copy(path_to_folder + file_img, path_to_dataset + str(i + 1 + nb_files) + ".txt")
            i += 1
            print("adding " + file + " to dataset")


def change_name(Path_to_img, Path_to_txt):
    """
    Function to change name of files in 2 folders (txt and jpg) to start at 0 while keeping the same order and name
    (to merge 2 datasets and keep the link between txt and jpg)
    :param Path_to_img:  Path to folder with images
    :param Path_to_txt:  Path to folder with txt
    :return:  nothing
    """
    i = 0
    # open each file in folder Cal-tech/images/ order it by name and print
    for file in sorted(os.listdir(Path_to_img)):
        # keep only chars between 2 and .png
        name = file[:-4]
        print(file + " -> " + name)

        # verify if train+name+squared.txt exist in path_to_txt
        txt_name = name + ".txt"
        print(Path_to_txt + txt_name)
        if os.path.exists(Path_to_txt + txt_name):
            i += 1
            print(name + " exist")
            # rename both files to i.txt or i.jpg
            os.rename(Path_to_txt + txt_name, Path_to_txt + str(i) + ".txt")
            os.rename(Path_to_img + file, Path_to_img + str(i) + ".jpg")
        else:
            # delete
            print(name + " doesn't exist")
            os.remove(Path_to_img + file)
        print("------------------------------")

    # 01_V000_0
    print(i, "correct files in folder")


def change_class(path):
    """
    Function to change class of each file in a folder
    :param path: path to folder with files
    :return:    nothing
    """
    for file in os.listdir(path):
        if file.endswith(".txt"):
            # for each line, if 1st character of line is 1, replace it by 0 and if 0 replace it by 1
            with open(path + file, 'r') as text:
                lines = text.readlines()
            with open(path + file, 'w') as text:

                for line in lines:
                    # delete line that don't start with 0 or 2 (0 is pedestrian, 2 is car)
                    line = line.split(" ")
                    print(line)
                    if line[0] == "0":
                        text.write("1" + " " + line[1] + " " + line[2] + " " + line[3] + " " + line[4])
                    elif line[0] == "2":
                        text.write("0" + " " + line[1] + " " + line[2] + " " + line[3] + " " + line[4])
                    else:
                        text.write("2" + " " + line[1] + " " + line[2] + " " + line[3] + " " + line[4])


def recount(path_count, path):
    """
    Function to count how many files are in a folder, and rename file in another folder from the end of the count
    :param path_count:  path to folder to count
    :param path:      path to folder to rename
    :return:
    """
    nb_files = 0
    # count how many files are in dataset
    for file in os.listdir(path_count):
        if file.endswith(".txt"):
            nb_files += 1
    nb_files = nb_files
    print(nb_files, "files in dataset")
    i = 0
    for file in os.listdir(path):
        print(file)
        # move file to path_to_dataset with value 1+nv_files
        if file.endswith(".txt"):
            os.rename(path + file, path + str(i + nb_files) + ".txt")
            file_img = file[:-3] + "jpg"
            os.rename(path + file_img, path + str(i + nb_files) + ".jpg")
            i += 1
'''