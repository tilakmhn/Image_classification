from collections import OrderedDict
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import re


def buildClassifier(inputSize, outputSize, hidden_layer, dropOut):
    linear_layers = nn.ModuleList([nn.Linear(inputSize, hidden_layer[0])])
    hidden_layers = list(zip(hidden_layer[:-1], hidden_layer[1:]))
    linear_layers.extend([nn.Linear(input_size, output_size) for input_size, output_size in hidden_layers])
    hidden_to_output_layer = hidden_layer[-1] if len(hidden_layer) > 0 else hidden_layer[0]
    output_layer = nn.Linear(hidden_to_output_layer, outputSize)
    orderedDict = OrderedDict()
    i = 1
    for layer in linear_layers:
        layerName = "level" + str(i)
        droupOutName = "dropOut" + str(i)
        reluName = "relu" + str(i)
        orderedDict[layerName] = layer
        orderedDict[reluName] = nn.ReLU()
        orderedDict[droupOutName] = nn.Dropout(dropOut)
        i = i + 1

    orderedDict["output"] = output_layer;
    orderedDict["softMax"] = nn.LogSoftmax(dim=1)
    classifier = nn.Sequential(orderedDict)
    return classifier;


def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    image1 = Image.open(img)
    image1.thumbnail((256, 256))

    # Get coordinates for center crop and crop the image
    width, height = image1.size  # Get dimensions
    new_width = 224
    new_height = 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    resizedImage = image1.crop((left, top, right, bottom))
    np_image = np.array(resizedImage)

    # Convert values between 0 and 1
    np_image11 = np.divide(np_image, [255, 255, 255])

    np_image1 = np.subtract(np_image11, [0.485, 0.456, 0.406])
    np_image2 = np.divide(np_image1, [0.229, 0.224, 0.225])
    np_image3 = np_image2.transpose(2, 0, 1)

    return np_image3


def revertProcessedImage(image):
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    return image;


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    image = revertProcessedImage(image)
    ax.imshow(image)

    return ax


def get_cat_name_mapping():
    cat_to_name ={}
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def get_index_to_class_map(class_to_idx):
    classToIdxMap = class_to_idx
    return {v: k for k, v in classToIdxMap.items()}


def get_image_category_value_from_key(categoryKey):
    '''
        Get flower name from  key
        Key is the key of cat_to_name dictionary
    '''
    cat_to_name = get_cat_name_mapping()
    categoryName = cat_to_name[categoryKey]
    return categoryName


def get_image_category_key(categoryKeyList, class_to_idx):
    '''
    Create flower name array
    '''
    cat_to_name = get_cat_name_mapping()
    index_to_class_map = get_index_to_class_map(class_to_idx)
    categoryNameArray = [cat_to_name.get(index_to_class_map.get(v)) for v in categoryKeyList]
    return categoryNameArray

def get_class_from_url(image_url):
    '''
    Returns image class from image URL
    input: flowers/test/13/image_05787.jpg
    output : 13
    :param image_url:
    :return: an integer indicating image class
    '''
    result = re.search('(.*)/(.*)/(.*)$', image_url)
    return result.group(2)

def view_topN_predictions(probs, indices, img, class_to_idx, top_k, image_path):
    '''
        1. Call predict method to get predictions and images object
        2.
    '''
    print("View TOP Predictions - START")
    actualClass = get_class_from_url(image_path)
    print(actualClass)
    numpyIndices = indices.numpy().flatten()
    numpyProbs = probs.numpy().flatten()
    categoryNameArray = get_image_category_key(numpyIndices, class_to_idx)
    actualCategory = get_image_category_value_from_key(actualClass)

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 10), nrows=2)

    ax1.set_title(actualCategory)
    img.squeeze_(0)
    np_img = np.array(img)
    np_img = revertProcessedImage(np_img)
    ax1.imshow(np_img.squeeze())
    ax1.axis('off')

    ax2.barh(np.arange(top_k), numpyProbs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(categoryNameArray)
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    print("View TOP Predictions - END")