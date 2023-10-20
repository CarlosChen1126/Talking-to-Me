import torch
from torch import nn

class Classifier_Vision(nn.Module):
    def __init__(self):
        super(Classifier_Vis, self).__init__()
        self.cnn_layers = nn.Sequential(

            nn.Conv2d(256, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(16, 16, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        # print(x.shape)
        x = self.fc_layers(x)
        return x

class Classifier_Audio(nn.Module):
    def __init__(self):
        super(Classifier_Aud, self).__init__()
        self.cnn_layers = nn.Sequential(

            nn.Conv2d(256, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(3, 32, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, y):
        y = self.cnn_layers(y)
        y = y.flatten(1)
        # print(y.shape)
        y = self.fc_layers(y)
        return y