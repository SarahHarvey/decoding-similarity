import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.models as models

import dsutils

def extract_rep(model: str, trainloader) -> npt.NDArray:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')

    if model == "resnet18":
        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        resnet18.eval().to(device)

        module1 = list(resnet18.children())[0:9]
        module2 = list(resnet18.children())[9:]

        rnet18_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet18_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet18_1st(inputs)
                # y2 = rnet18_2nd(y1) 

        return y1


    if model == "resnet50":

        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        resnet50.eval().to(device)

        module1 = list(resnet50.children())[:6]
        module2 = list(resnet50.children())[6:]

        rnet50_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet50_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet50_1st(inputs)
                # y2 = rnet50_2nd(y1) 

        return y1
    

    if model == "alexnet":
        
        alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        alexnet.eval().to(device)

        module1 = list(alexnet.children())[:1]
        module2 = list(alexnet.children())[1:2]
        module3 = list(alexnet.children())[2:]

        anet_1st = nn.Sequential(*[*module1, *module2, dsutils.Flatten(), module3[0][0:6]] )
        anet_2nd = nn.Sequential(*[module3[0][6:], dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                x1 = anet_1st(inputs)
                # x2 = anet_2nd(x1) 
                
        return x1

    if model == "resnet34":
 
        resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        resnet34.eval().to(device)

        module1 = list(resnet34.children())[0:9]
        module2 = list(resnet34.children())[9:]

        rnet34_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet34_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet34_1st(inputs)
                # y2 = rnet34_2nd(y1) 

        return y1

    if model == "resnet101":

        resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        resnet101.eval().to(device)

        module1 = list(resnet101.children())[0:9]
        module2 = list(resnet101.children())[9:]

        rnet101_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet101_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet101_1st(inputs)
                # y2 = rnet101_2nd(y1) 

        return y1

    if model == "resnet152":

        resnet152 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        resnet152.eval().to(device)

        module1 = list(resnet152.children())[0:9]
        module2 = list(resnet152.children())[9:]

        rnet152_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet152_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet152_1st(inputs)
                # y2 = rnet152_2nd(y1)
                
        return y1 

    if model == "vgg16":
        vgg16 = models.vgg16(weights='DEFAULT')
        vgg16.eval().to(device)

        module1 = list(vgg16.children())[0:2]
        module2 = list(vgg16.children())[2:]

        vgg16_1st = nn.Sequential(*[*module1, dsutils.Flatten(), module2[0][0:6]] )
        vgg16_2nd = nn.Sequential(*[module2[0][6:], dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = vgg16_1st(inputs)
                # y2 = vgg16_2nd(y1) 

        return y1


    else: 
        return print("Model name not found")
