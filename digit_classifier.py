import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import os
import matplotlib.pyplot as plt
from scipy import ndimage
batch_size = 512


class CNN_classifier(nn.Module):
    # initialization
    def __init__(self):
        super(CNN_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    # forward path
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.relu()

        # print(x.shape)

        x = x.view(-1, 64 * 3 * 3)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def thresholding(prediction):
    # Find label which shows highest prediction value
    _, pred_label = torch.max(prediction, 1)
    return pred_label

# Load the model
model = CNN_classifier()
PATH = "cnn_model.pth"
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))

def prepare(img):
    img = cv2.bitwise_not(img)
    h, w = 30, 30
    y, x = 10, 10

    crop_img = img[y:y + h, x:x + w]
    resized = cv2.resize(crop_img, (28, 28), interpolation=cv2.INTER_AREA)

    return resized

def predict(boxes):
    predictions = []
    i = 1
    for box in boxes:
        imgtest = box.copy()
        directory = r'C:\Users\PC\PycharmProjects\SudokuSolver'
        os.chdir(directory)
        path = "test"+ str(i) + ".png"
        i+=1
        cv2.imwrite("SudokuImage/" + path, imgtest)
        imgtest = cv2.imread("SudokuImage/" + path)


        img = cv2.bitwise_not(imgtest)
        h, w = 30, 30
        y, x = 10, 10

        crop_img = img[y:y + h, x:x + w]
        resized = cv2.resize(crop_img, (28, 28), interpolation=cv2.INTER_AREA)

        denoised_square = ndimage.median_filter(resized, 3)
        white_pix_count = np.count_nonzero(denoised_square)
        if white_pix_count > 200:
            empty_square = False
        else:
            empty_square = True
        if empty_square:
            predictions.append(-1)
            continue
        grayImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        # print(blackAndWhiteImage.shape)

        resized = blackAndWhiteImage
        # print(resized.shape)

        im_pil = Image.fromarray(resized)
        # print(im_pil.size)

        imtest = transforms.ToTensor()(im_pil).unsqueeze_(0)
        # print(imtest.shape)
        prediction = model(imtest)
        prediction = thresholding(prediction)
        predictions.append(prediction.detach().numpy()[0])
        # print(thresholding(prediction))
    return predictions
