from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image, ImageOps
import numpy as np
import model
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

model = model.DigitRecognizer()
model.load_state_dict(torch.load("MNIST.pth"))
model.eval()

def predict_digit(img):
    #resize image to 28x28 pixels
    #convert rgb to grayscale
    img = transforms.Grayscale()(img)
    img = ImageOps.invert(img)
    plt.imshow(img)
    plt.show()
    img = np.array(img)
    #reshaping to support our model input and normalizing
    #img = img.reshape(1,28,28)
    #img = img/255.0
    #predicting the class
    #img = img
    img = transforms.ToTensor()(img)
    img = img.reshape(1,1,28,28)
    res = model(img)
    probs = res.tolist()[0]
    [print(str(probs.index(x))+": " +str(round(x,2)*100)+"%") for x in probs]
    #return torch.argmax(res).item(), str(torch.max(res).item()*100)+"%"
# data = datasets.MNIST('data',train=False, download=False)
# image = data[7][0]
file = input("Enter file path: ")
predict_digit(Image.open(file))