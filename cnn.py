# https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import pickle
import cv2

batch_size = 512
CNN_PATH = "./checkpoints/mnist_cnn.pt"


def load_model():
    model = CNN()
    device = torch.device('cpu')
    model.load_state_dict(torch.load(CNN_PATH, map_location=device))
    model.eval()
    return model


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def get_pred(prediction):
        _, pred_label = torch.max(prediction, 1)
        return pred_label.item()

if __name__ == "__main__": 
    cnn_model = load_model()
    
    # image_path = "./img/sudoku_label1/1/54.png"
    # image_path = "./img/sudoku_label1/2/6.png"
    image_path = "./img/sudoku_label3/1/.png"
    img = Image.open(image_path).convert('L')
    img = PIL.ImageOps.invert(img)

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(40),
            transforms.Resize(28, antialias=True),
    ])


    # train_kwargs = {'batch_size': 64}
    # dataset1 = datasets.MNIST('./MNIST', train=True, download=True,
    #                 transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    # for batch in train_loader:
    #     images, labels = batch
    #     break

    input_image = transform(img).unsqueeze(0)
    # input_image = images[2].unsqueeze(0)
    transforms.ToPILImage()(input_image[0]).show()
    
    with torch.no_grad():
        output = cnn_model(input_image)
        predicted_class = get_pred(output)
    print(f"Predicted digit: {predicted_class}")