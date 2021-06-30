import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


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

