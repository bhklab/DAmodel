import torch
import torch.nn as nn
import torch.optim as optim

import json
from processing import ImageProcessor
from modelhublib.model import ModelBase


class Net(nn.Module):
    #CNN toplogy definition
    def __init__(self, output_dim):
        super(Net, self).__init__()

        self.pool = nn.MaxPool3d(2, 2)
        self.LRelu = nn.LeakyReLU(0.01)
        
        self.conv1 = nn.Conv3d(1, 4, 5, padding=2)
        self.conv1_bn = nn.BatchNorm3d(4)
        
        self.conv2 = nn.Conv3d(4, 8, 3, padding=1)
        self.conv2_bn = nn.BatchNorm3d(8)
        
        self.conv3 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv3_bn = nn.BatchNorm3d(16)

        self.conv4 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv4_bn = nn.BatchNorm3d(32)

        self.conv5 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv5_bn = nn.BatchNorm3d(64)
        
        self.avgPool = nn.AvgPool3d(2, 2)
        
        self.fc3 = nn.Linear(64 * 8 * 8 * 8, output_dim)
        
        
    def forward(self, input1):   
        input1 = self.pool(self.conv1_bn(self.LRelu(self.conv1(input1))))
        input1 = self.pool(self.conv2_bn(self.LRelu(self.conv2(input1))))
        input1 = self.pool(self.conv3_bn(self.LRelu(self.conv3(input1)))) 
        input1 = self.pool(self.conv4_bn(self.LRelu(self.conv4(input1)))) 
        input1 = self.conv5_bn(self.LRelu(self.conv5(input1)))
        input1 = self.avgPool(input1)
        input1 = input1.view(-1, 64 * 8 * 8 * 8)
        input1 = self.fc3(input1)     
           
        return input1
    
class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL model 
        nclasses = 2
        self._model = Net(output_dim = nclasses)
    
        learning_rate = 0.001
        momentum = 0.9
        weight_decay = 0.0001
    
        optimizer = optim.SGD(self._model.parameters(), lr = learning_rate, momentum=momentum, weight_decay = weight_decay) 
      
        checkpoint = torch.load('model/testCheckpoint.pth.tar, map_location=lambda storage, loc:storage)      
        optimizer.load_state_dict(checkpoint['optimizer'])

        self._model.load_state_dict(checkpoint['state_dict'])
        self._model.eval()


    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # Run inference
        results = self._model.(input)
        # postprocess results into output
        output = self._imageProcessor.computeOutput(results)
        return output
        

