import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

batch_size = 512
class mlp_classifier(nn.Module):

    def __init__(self):
        super(mlp_classifier, self).__init__()
        # ACTIVITY  : fill in this part
        self.layer1 = nn.Linear(28 * 28, 700)
        self.layer2 = nn.Linear(700, 500)
        self.layer3 = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.view(-1, 28 * 28)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)

        return x

def thresholding(prediction):
    # Find label which shows highest prediction value
    _, pred_label = torch.max(prediction, 1)
    return pred_label

# Load the model
model = mlp_classifier()
PATH = "model.pth"
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))

def predict(boxes):
    predictions = []
    for box in boxes:
        i = 1
        imgtest = box.copy()
        directory = r'C:\Users\PC\PycharmProjects\SudokuSolver'
        os.chdir(directory)
        path = "test"+ str(i) + ".png"
        i+=1
        cv2.imwrite("SudokuImage/" + path, imgtest)
        imgtest = cv2.imread("SudokuImage/" + path)
        img = cv2.bitwise_not(imgtest)
        h, w = 40, 40
        y, x = 10, 5

        crop_img = img[y:y + h, x:x + w]
        # print(crop_img.shape)
        resized = cv2.resize(crop_img, (28, 28), interpolation=cv2.INTER_AREA)
        grayImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        # print(blackAndWhiteImage.shape)

        resized = blackAndWhiteImage
        # print(resized.shape)

        im_pil = Image.fromarray(resized)
        # print(im_pil.size)

        imtest = transforms.ToTensor()(im_pil).unsqueeze_(0)[0][0]
        # print(imtest.shape)

        prediction = model(imtest)
        prediction = thresholding(prediction)
        predictions.append(prediction.detach().numpy()[0])
        # print(thresholding(prediction))
    return predictions

# img = cv2.imread("number3.png")
# img = cv2.bitwise_not(img)
# h,w = 40, 40
# y,x = 10, 5
# crop_img = img[y:y+h,x:x+w]
# resized = cv2.resize(crop_img, (28,28), interpolation = cv2.INTER_AREA)
# grayImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
# print(blackAndWhiteImage.shape)
#
# resized = blackAndWhiteImage
# print(resized.shape)
#
# im_pil = Image.fromarray(resized)
# print(im_pil.size)
#
# imtest = transforms.ToTensor()(im_pil).unsqueeze_(0)[0][0]
# print(imtest.shape)
#
# prediction = model(imtest)
# print(thresholding(prediction))
# # cv2.imshow("test", img)
# cv2.waitKey(0)

