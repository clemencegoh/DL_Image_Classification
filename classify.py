from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
from vocparseclslabels import PascalVOC

#import skimage.io
import PIL.Image

from operator import itemgetter
import json


class CustomClassifier:
    def __init__(self, _transforms, _model):
        # transforms should be the same as the one used in training
        self.transforms = _transforms
        self.model = _model
        self.device = torch.device("cpu")

    def predictionArr(self, image):
        image_tensor = self.transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        data = output.data.cpu().numpy()[0]

        return data

    def predict(self, image):
        image_tensor = self.transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        data = output.data.cpu().numpy()[0]

        index = output.data.cpu().numpy().argmax()
        score = data[index]

        return index, score


all_labels = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']


def loadClassifier(_saved_dict='./69_resnet34.pt'):
    # model
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 20)

    model.load_state_dict(torch.load(_saved_dict))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    c = CustomClassifier(_transforms=transform, _model=model)
    return c


def ClassifyImage(_classifier,
                  _image='./static/VOCdevkit/VOC2012/JPEGImages/2010_005174.jpg'):
    """
    Classifies a single image given using classifier given
    :param _classifier: classifier from loadClassifier
    :param _image: filepath for image
    :return: class of image (string), certainty score (float)
    """

    # get image
    to_pil = transforms.ToPILImage()
    image = to_pil(np.array(PIL.Image.open(_image)))

    index, score = _classifier.predict(image)

    return all_labels[index], score


def ClassifyTest(_saved_dict='./69_resnet34.pt',
                  _image='./static/VOCdevkit/VOC2012/JPEGImages/2010_005174.jpg'):
    """
        Classifies a single image given, for demo
        :param _saved_dict: saved_dict from a trained model
        :param _image: filepath for image
        :return: class of image (string), certainty score (float)
    """
    c = loadClassifier(_saved_dict)
    return ClassifyImage(c, _image)


def saveData(_dict):
    with open('data.json', 'w') as f:
        f.write(json.dumps(_dict))

def sortData(_dict):
    for key, val in _dict.items():
        val.sort(key=itemgetter('score'), reverse=True)
    return _dict

#DEPRECATED: This should no longer be used, all data should be shown
def cutData(_dict):
    """
    Cuts from top 50
    :param _dict: dictionary input
    :return: smaller dict with 50 items or less each
    """
    for key, val in _dict.items():
        if len(val) < 50:
            _dict[key] = val[:len(val)-1]
        else:
            _dict[key] = val[:50]
    return _dict


def BuildDB(saved_dict=None):
    pv = PascalVOC('./static/VOCdevkit/VOC2012')
    if saved_dict is not None:
        c = loadClassifier(saved_dict)
    else:
        c = loadClassifier()

    overall_data = {}

    # data_arr = ['val', 'trainval']
    data_arr = ['val']
    for d in data_arr:
        print('looking at:', d)
        # cat = all_labels[0]
        for cat in all_labels:
            print('looping through cat:', cat)
            ls = pv.imgs_from_category_as_list(cat, d)
            for xml in ls:
                imagepath = pv.getImagePath(xml)
                label, score = ClassifyImage(c, _image=imagepath)
                if score > 0:
                    if label not in overall_data:
                        overall_data[label] = []
                    # print(score.item())
                    overall_data[label].append({'image':imagepath, 'score':score.item()})

    print('sorting...')
    sorted_data = sortData(overall_data)

    print("saving...")
    saveData(sorted_data)

    print('done')


def testData():
    with open('data69.json', 'r') as f:
        data = json.loads(f.read())
        print(data)


if __name__ == '__main__':
    # print(ClassifyTest())

    # todo: add new saved dict whenever there is a change
    BuildDB(saved_dict='./69_resnet34.pt')
    # testData()