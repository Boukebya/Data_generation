from collections import Counter
from src.prompt_generator import words_list
import matplotlib.pyplot as plt
import warnings


def add_word(string_words, data_prediction, list_of_words, i):
    for synonymous in string_words:
        # print(synonymous)
        for word in synonymous:
            if word in data_prediction[i]:
                print("found " + word)
                list_of_words += word + " "
                return list_of_words


# Plot statistic of a folder of generated images
def statistic(path, prediction, threshold):
    """
    Function to plot statistic of a folder of generated images
    Parameters
    ----------
    path : str, path to file with prompt, name (and predicted number of objects for controlnet)
    prediction : str, path to prediction file of yolo
    threshold : float, threshold to use
    ----------
    """

    print("Using threshold : " + str(threshold))
    print("Using path : " + path)
    print("Using prediction : " + prediction)

    # for each line in the file
    with open(path, 'r') as file_reader:
        data_prediction = file_reader.readlines()
    with open(prediction, 'r') as file_reader_prediction:
        data_yolo_prediction = file_reader_prediction.readlines()

    string_words = words_list(0)
    adj = words_list(1)
    adj = adj[0]
    print(adj)
    if adj in string_words:
        string_words.replace(adj, "")

    for i in range(len(data_prediction)):
        if adj in data_prediction[i]:
            data_prediction[i] = data_prediction[i].replace(adj, "")

    number_of_detection = len(data_prediction)
    number_of_detection_kept = 0

    print(string_words)
    all_words_removed = ""
    all_words_kept = ""

    # for each line in the file, compare the number of
    # objects predicted by yolo and the number of objects predicted by controlnet
    # and find occurrences between words in string_words and data[i]
    i = 0
    while i < len(data_prediction):

        # ground truth formatted
        ground_truth = int(data_prediction[i].split(",")[-2])

        yolo_prediction = data_yolo_prediction[i].split(":")[1]
        yolo_prediction = int(yolo_prediction.split(" ")[1])

        print(int(yolo_prediction), "/", int(ground_truth))

        comparison = float(ground_truth * threshold)

        if int(yolo_prediction) >= comparison:
            number_of_detection_kept += 1
            all_words_kept = add_word(string_words, data_prediction, all_words_kept, i)

        else:
            # find occurrences between words in string_words and data[i]
            all_words_removed = add_word(string_words, data_prediction, all_words_kept, i)
        i += 1

    # remove last space
    all_words_removed = all_words_removed[:-1]

    print("ratio:")
    print(number_of_detection_kept, "/", number_of_detection)
    print("all words kept :")
    print(all_words_kept)
    print("all words removed :")
    print(all_words_removed)

    # I don't know why, but I have a warning when I do a subplot, so I ignore it
    warnings.simplefilter("ignore", UserWarning)

    # Plot files that should be deleted
    plot_deleted(number_of_detection, number_of_detection_kept, data_prediction,
                 all_words_kept, all_words_removed, string_words)
    # Plot words that are the most present in files that should be deleted
    plot_by_synonymous(all_words_removed, string_words)

    print("Stat done")


def plot_by_synonymous(all_words_removed, string_words):
    """
    Function to plot words that are the most present in files that should be deleted
    Parameters
    ----------
    all_words_removed : str, all words removed
    string_words : list of str, synonymous
    """

    # plot a bar chart for each element in string_words, count the number of removed words from element
    all_words_removed = all_words_removed.split(" ")
    # print(all_words_removed)
    all_words_removed = Counter(all_words_removed)
    print(all_words_removed)

    # for each synonymous subplot a bar chart with corresponding number of removed words
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Number of removed words by synonymous')
    i = 0
    z = 0
    for synonymous in string_words:
        # print(synonymous)
        number_of_removed_words = []
        for word in synonymous:
            number_of_removed_words.append(all_words_removed[word])

        print(synonymous, " : ", number_of_removed_words)

        axs[i][z].bar(synonymous, number_of_removed_words)
        i += 1
        if i == 3:
            i = 0
            z += 1

    plt.show()


def plot_deleted(number_of_detection, number_of_detection_kept, data, all_words_kept, all_words_removed, string_words):
    """
    Function to plot files that should be deleted
    Parameters
    ----------
    number_of_detection : int, number of detection
    number_of_detection_kept : int, number of detection kept
    data : list of str, data
    all_words_kept : str, all words kept
    all_words_removed : str, all words removed
    string_words : list of str, synonymous
    """

    # plot in a pie chart
    labels = f'Good detection {number_of_detection_kept}', \
        f'Bad samples {number_of_detection - number_of_detection_kept}'
    sizes = [number_of_detection_kept, number_of_detection - number_of_detection_kept]

    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f"Ratio of good detection on {len(data)} images")
    plt.show()

    len_category = []
    for synonymous in string_words:
        len_category.append(len(synonymous))
    print(len_category)

    # in all_words, count the number of occurrences of each word
    all_words_removed = all_words_removed.split(" ")
    print(all_words_removed)
    all_words_removed = Counter(all_words_removed)
    print(all_words_removed)

    all_words_kept = all_words_kept.split(" ")
    print(all_words_kept)
    all_words_kept = Counter(all_words_kept)
    print(all_words_kept)

    # plot in a bar chart, order all removed words by number of occurrences
    all_words_removed = all_words_removed.most_common()

    # for each word in all_words_removed, find the category it belongs to, and multiply it by length of category
    for i in range(len(all_words_removed)):
        for synonymous in string_words:
            if all_words_removed[i][0] in synonymous:
                all_words_removed[i] = (all_words_removed[i][0], all_words_removed[i][1] / len(synonymous))
                break

    # order by number of occurrences
    all_words_removed = sorted(all_words_removed, key=lambda x: x[1], reverse=False)
    print(all_words_removed)

    # plot in a bar chart
    plt.bar(range(len(all_words_removed)), [val[1] for val in all_words_removed], align='center', color='red')
    plt.xticks(range(len(all_words_removed)), [val[0] for val in all_words_removed])
    plt.xticks(rotation=90)
    plt.title("Number of occurrences divided by numbers of synonyms for each word in removed samples")
    plt.show()
