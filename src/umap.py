import json
import os
from keras import Model
from keras.applications.densenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

with open("config.json", "r") as f:
    config = json.load(f)

# Not ended yet
def umap_displayer():
    data_path = "generated_images/cropped_images/"
    folder = os.listdir(data_path)

    # load vgg model
    from keras.applications.vgg16 import VGG16
    # load the model
    model = VGG16()
    print(model.summary())
    model = Model(inputs=model.inputs, outputs=model.layers[12].output)

    synthetic_images_features = []
    real_images_features = []
    for element in folder:

        if element.endswith("original.png"):
            # resize image to 224x224
            img = Image.open(data_path + element)

            img = img.resize((224, 224))
            img = np.asarray(img)
            img = preprocess_input(img)

            real_images_features.append(model.predict(np.expand_dims(img, axis=0)))
            break

        elif element.endswith(".png"):
            img = Image.open(data_path + element)

            img = img.resize((224, 224))
            img = np.asarray(img)
            img = preprocess_input(img)
            synthetic_images_features.append(model.predict(np.expand_dims(img, axis=0)))

    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(real_images_features[0][0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()
