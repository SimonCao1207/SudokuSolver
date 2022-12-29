import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *

batch_size = 512
PATH = "./checkpoints/mnist_cnn.pt"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
    return pred_label


def load_model():
    model = Net()
    device = torch.device('cpu')
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    return model

if __name__ == "__main__": 
    pathImage = "./img/4.png"
    model = load_model()    
    
    # img = cv2.imread(pathImage)
    # img = cv2.bitwise_not(img)
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    # transform=transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,)),
    # transforms.Resize(28)
    # ])
    
    # img_trans = transform(blackAndWhiteImage)
    # plt.imshow(img_trans[0])
    # plt.show()
    # img_trans = img_trans.unsqueeze_(0)
    # print(img_trans.shape)
    # out = model(img_trans)

    # print(out)
    # print(get_pred(out))



