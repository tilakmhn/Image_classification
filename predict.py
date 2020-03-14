
import torch as torch
from torchvision import models
import argparse
import utility


def get_args():
    '''
    Get command line arguments

    :return:
    '''
    parser = argparse.ArgumentParser(description='Predict neural network')
    parser.add_argument('path_to_image', metavar='path_to_image', type=str, help='Path to Image')
    parser.add_argument('checkpoint', metavar='checkpoint',  type=str, help='Checkpoint File')
    parser.add_argument('--top_k', metavar='top_k',  type=int, help='Top K Values')
    parser.add_argument('--category_names', metavar='--category names',  type=float, help='category names')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    image_path = args.path_to_image
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    return image_path, checkpoint, top_k, category_names, gpu


# TODO: Write a function that loads a checkpoint and rebuilds the model
def rebuild_model(checkpoint):
    # Step 1 : Create Classifier
    checkpointLoaded = torch.load(checkpoint)

    dropOut = checkpointLoaded["dropout"]
    inputSize = 25088#checkpointLoaded["input_size"]
    hidden_layer = checkpointLoaded["hidden_layer"]
    outputSize = checkpointLoaded["output_size"]
    state_dict = checkpointLoaded["state_dict"]
    model_name = "vgg19" #checkpointLoaded["model_name"]
    classifier = utility.buildClassifier(inputSize, outputSize, hidden_layer, dropOut)

    # Step - 2 Load state_dict and assign classifier
    # It assumes that the model is either vgg19 or vgg13
    newModel = None
    if model_name == "vgg19":
        newModel = models.vgg19()
    else:
        newModel = models.vgg13()

    newModel.classifier = classifier
    print(state_dict)
    newModel.load_state_dict(state_dict)
    return newModel, checkpointLoaded;


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print("Inside Predict : START")
    model.eval()
    img = utility.process_image(image_path)
    img = torch.from_numpy(img)
    img = img.unsqueeze_(0).float()

    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output)
    probs, indices = torch.topk(ps, topk)
    print("Inside Predict : END")
    return probs, indices, img


if __name__ == '__main__':
    image_path , checkpoint , top_k , category_names , gpu = get_args()
    rebuiltModel, checkpointLoaded = rebuild_model(checkpoint)
    probs, indices, img = predict(image_path, rebuiltModel)
    class_to_idx = checkpointLoaded["class_to_idx"]
    utility.view_topN_predictions(probs, indices, img, class_to_idx, top_k, image_path)

