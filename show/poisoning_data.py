import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from IPython import display
"""
Fashion-MNIST中一共包括了10个类别,分别为
        t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、
        coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）
"""

def showImg(data_sets):
    print("总长度",len(data_sets))
    batch = next(iter(data_sets))
    images, labels = batch
    grid = torchvision.utils.make_grid(images,nrow=15)
    plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(grid,(1,2,0)))
    plt.title("Ground Truth: {}".format(labels))
    plt.show()
    print(batch[1])