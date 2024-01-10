# https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from utils import load_model, get_pred

if __name__ == "__main__": 
    cnn_model = load_model(model_name="svhn")
    
    image_path = "./img/sudoku_label1/1/54.png"
    # image_path = "./img/sudoku_label2/2/2.png"
    # image_path = "./img/sudoku_label2/2/13.png"
    # image_path = "./img/sudoku_label2/2/32.png"
    # image_path = "./img/sudoku_label2/2/44.png"
    # image_path = "./img/sudoku_label2/3/0.png"
    # image_path = "./img/sudoku_label2/3/24.png"
    # image_path = "./img/sudoku_label2/4/1.png"
    # image_path = "./img/sudoku_label2/4/14.png"
    # image_path = "./img/sudoku_label2/4/36.png"
    # image_path = "./img/sudoku_label2/4/66.png"
    # image_path = "./img/sudoku_label2/5/21.png"
    # image_path = "./img/sudoku_label3/8/28.png"

    img = Image.open(image_path).convert('L')
    img = PIL.ImageOps.invert(img)

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.CenterCrop(40),
            transforms.Resize(32, antialias=True),
    ])

    input_image = transform(img).unsqueeze(0)
    input_image = torch.cat((input_image, input_image, input_image), 1)
    
    transforms.ToPILImage()(input_image[0]).show()
    
    with torch.no_grad():
        output = cnn_model(input_image)
        print(output)
        predicted_class = get_pred(output)
    print(f"Predicted digit: {predicted_class}")