import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv

import torchvision.transforms as transforms
import torchvision.models as models
import copy
import numpy as numpy

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(imagePath):
    loader = transforms.Compose([transforms.resize(imsize), transforms.ToTensor()])
    image = cv.imread(imagePath)
    image = loader(image).unsqueeze(0)
    return image.to(cuda, torch.float)

class ContentLoss(nn.Module):
    def __init__(self, target):
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

#Style Loss
def gram_matrix(input):
    a, b, c, d = input.size() # a = batchsize b = # of feature maps (c, d) = dimensions of f. map (N= c*d)

    features = input.view(a * b, c * d) # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t()) # Compute gram product

    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

#Create module to normalize input image so we can easily put it in nn.sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std #Normalize image


imsize = 512 if torch.cuda.is_available() else 128
#Desired depth layers to compute style / content losses:
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3'. 'conv_4'] 

contentImage = image_loader("randomImage/blah.jpg")
styleImage = image_loader("randomImage/")

assert contentImage.size() == styleImage.size(), "Images must be same size"

unloader = transforms.ToPILImage() #Unloads from tensor to PIL for viewing image

cnn = models.vgg19(pretrained = True).features.to(cuda).eval() #Import vgg19 model and set to evaluation mode

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(cuda)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.255]).to(cuda)




