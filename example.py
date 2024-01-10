# https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from utils import load_model, get_pred

if __name__ == "__main__": 
    cnn_model = load_model()
    
    # image_path = "./img/sudoku_label1/1/54.png"
    image_path = "./img/sudoku_label1/2/6.png"
    image_path = "./img/sudoku_label1/3/5.png"
    img = Image.open(image_path).convert('L')
    img = PIL.ImageOps.invert(img)

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(40),
            transforms.Resize(32, antialias=True),
    ])

    # train_kwargs = {'batch_size': 64}
    # dataset1 = datasets.MNIST('./MNIST', train=True, download=True,
    #                 transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    input_image = transform(img).unsqueeze(0)
    input_image = torch.cat((input_image, input_image, input_image), 1)
    # input_image = images[2].unsqueeze(0)
    # transforms.ToPILImage()(input_image[0]).show()
    
    with torch.no_grad():
        output = cnn_model(input_image)
        predicted_class = get_pred(output)
    print(f"Predicted digit: {predicted_class}")