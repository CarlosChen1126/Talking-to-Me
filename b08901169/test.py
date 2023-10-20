import os
import glob
import torch
from torch import nn
# import torchaudio

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# print('Device used:', device)

class Classifier_Audio_HubertLarge(nn.Module):
    def __init__(self):
        super(Classifier_Audio_HubertLarge, self).__init__()
        self.cnn_layers = nn.Sequential(
            # input: 250, 768
            nn.Conv1d(250, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2, 2),
            
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        # print(x.shape)
        x = x.flatten(1)
        print(x.shape)
        x = self.fc_layers(x)
        return x

model = Classifier_Audio_HubertLarge()
# a = torch.load("./hubert_features/0e08e1c7-3add-4e41-b28d-cd5f4413a422_36_69_1.pt", map_location=torch.device('cpu'))
# print(a.shape)

# b = model(a)
# print(b.shape)

a = torch.Tensor(64, 1000)
b = torch.Tensor(64, 500)
c = torch.cat([a,b], 1)
print(c.shape)

# m = nn.Conv1d(250, 128, 3, stride=1, padding=1)
# m2 = nn.MaxPool1d(2, 2, 0)
# c = torch.randn(1, 250, 1024)
# print(c.shape)
# b = model(c)
# print(b.shape)

