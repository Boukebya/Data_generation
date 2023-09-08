import json
import os
import shutil

with open("config.json", "r") as f:
    config = json.load(f)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    bb1 : dict
        with keys x1, y1, x2, y2
    bb2 : dict
        with keys x1, y1, x2, y2

    ----------
    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = convert_format(bb1)
    bb2 = convert_format(bb2)

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def convert_format(line):
    """
    Convert the format of the line from yolo to x_min, y_min, x_max, y_max
    :param line: line from yolo file to convert
    :return: dict with keys x1, y1, x2, y2, corresponding to x_min, y_min, x_max, y_max
    """

    line = line.split(" ")
    for i in range(len(line)):
        line[i] = float(line[i])

    x_center = float(line[1])
    y_center = float(line[2])
    width = float(line[3])
    height = float(line[4])

    # convert to x_min, y_min, x_max, y_max
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # convert to dict
    box = {'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max}

    return box


def compare(path_img_ground_truth, yolo_file_prediction, threshold):
    """
    Compare the predictions of yolo with the ground truth.
    Create a folder with the images that have an iou > threshold.
    Copy the images in the folder, and create a txt file with the predictions of yolo that are kept with iou > threshold
    params:
        path_img: path to the folder containing the images and ground truth
        yolo_file: path to the file containing the predictions of yolo
        threshold: threshold for the iou between the predictions of yolo and the ground truth
    """

    # delete content of file comparison
    with open(config['paths']['comparison'], "w") as file_emptier:
        file_emptier.write("")

    # Read line of yolo predictions
    with open(yolo_file_prediction, "r") as yolo_reader:
        lines_yolo_predictions = yolo_reader.readlines()

    # list of img folder
    list_img_ground_truth = os.listdir(path_img_ground_truth)

    list_of_detections = []

    # loop over all lines predicted by yolo
    for i in range(len(lines_yolo_predictions)):
        # detect if it's an image name in the line
        if lines_yolo_predictions[i].endswith(".jpg\n") or lines_yolo_predictions[i].endswith(".png\n"):
            detections_one_img = []
            # delete \n
            name = lines_yolo_predictions[i].replace("\n", "")
            # replace .png or .jpg by .txt
            name = name.replace(".png", ".txt")
            name = name.replace(".jpg", ".txt")
            detections_one_img.append(name)

            # while the next line is not an image name, add line to detections_one_img
            while not lines_yolo_predictions[i + 1].endswith(".jpg\n") and not lines_yolo_predictions[i + 1].endswith(".png\n"):
                add = lines_yolo_predictions[i + 1].replace("\n", "")
                detections_one_img.append(add)
                i += 1
                if i == len(lines_yolo_predictions) - 1:
                    break

            # list of detections for each image
            list_of_detections.append(detections_one_img)
    print(list_of_detections)

    # best iou for each element in list_of_detections (list of detections for each image)
    for element in list_of_detections:
        if element[0] in list_img_ground_truth:
            with open(path_img_ground_truth + element[0], "r") as reader:
                lines_in_img = reader.readlines()
                # delete \n
                lines_in_img = [x.replace("\n", "") for x in lines_in_img]
            # prints
            print("img:  ", lines_in_img)
            print("yolo: ", element[1:])
            print("")
            print("")
            print("length: ")
            print(len(lines_in_img))
            print(len(element[1:]))

            memory_best_iou_position = 0
            # find best iou for each line in img by comparing with all lines in yolo
            for i in range(len(lines_in_img)):
                best_iou = 0
                for z in range(len(element[1:])):
                    actual_iou = get_iou(lines_in_img[i], element[1:][z])
                    if actual_iou > best_iou:
                        best_iou = actual_iou
                        memory_best_iou_position = z

                # copy the line in the comparison file if iou > threshold
                print("best iou: ", best_iou)
                with open(config['paths']['comparison'], "a") as appender_txt:
                    if best_iou > threshold:
                        appender_txt.write(element[0] + " " + lines_in_img[i] + " " + element[1:][memory_best_iou_position] + "\n")
                        element[1:][memory_best_iou_position] = "already used"

                print("best iou: ", best_iou)

    # make a folder if it doesn't exist
    if not os.path.exists(config['paths']['image_kept']):
        os.makedirs(config['paths']['image_kept'])

    # delete element in config['paths']['image_kept']
    for element in os.listdir(config['paths']['image_kept']):
        os.remove(config['paths']['image_kept'] + element)

    # Copy images
    for element in os.listdir(path_img_ground_truth):
        print(element)
        if element.endswith(".jpg") or element.endswith(".png"):
            shutil.copy(path_img_ground_truth + element, config['paths']['image_kept'])

    # read comparison
    with open(config['paths']['comparison'], "r") as reader_txt:
        lines = reader_txt.readlines()

    lines = [x.split(" ") for x in lines]

    print(lines)
    name = lines[0][0]
    print("name: ", name)
    for element in lines:
        # remove \n
        element = [x.replace("\n", "") for x in element]
        print(element)
        if name == element[0]:
            print("same name continue")
            # create a file with name
            with open(config['paths']['image_kept'] + name, "a") as file_appender:
                file_appender.write(str(element[1:]) + " \n")
        else:
            name = element[0]
            with open(config['paths']['image_kept'] + name, "a") as file_appender:
                file_appender.write(str(element[1:]) + "\n")

    # remove [ ] , ' characters in txt and write txt
    for txt in os.listdir(config['paths']['image_kept']):
        if txt.endswith(".txt"):
            with open(config['paths']['image_kept'] + txt, "r") as txt_reader:
                lines = txt_reader.readlines()
                lines = [x.replace("[", "") for x in lines]
                lines = [x.replace("]", "") for x in lines]
                lines = [x.replace("'", "") for x in lines]
                lines = [x.replace(",", "") for x in lines]
            with open(config['paths']['image_kept'] + txt, "w") as txt_rewriter:
                for element in lines:
                    txt_rewriter.write(element)

    print("done")
