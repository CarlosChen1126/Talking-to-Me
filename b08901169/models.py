import torch
from torch import nn

# class Classifier_Vision(nn.Module):
#     def __init__(self):
#         super(Classifier_Vision, self).__init__()
#         self.cnn_layers = nn.Sequential(

#             nn.Conv2d(256, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.MaxPool2d(2, 2, 0),
            
#             nn.Conv2d(16, 32, 3, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.MaxPool2d(2, 2, 0),

#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.MaxPool2d(16, 16, 0),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(64, 16),
#             nn.ReLU(),
#             nn.Linear(16, 2)
#         )
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.flatten(1)
#         # print(x.shape)
#         x = self.fc_layers(x)
#         return x

class Classifier_Vision(nn.Module):
    def __init__(self):
        super(Classifier_Vision, self).__init__()
        self.cnn_layers = nn.Sequential(

            nn.Conv2d(3, 16, 5, 3, 1), # 3 16 5 3 1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(16, 32, 3, 2, 1), # 16 32 3 2 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1, 1), # 32 64 3 1 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(6400, 64), # 6400 64
            # nn.ReLU(),
            # nn.Linear(320, 32),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        # print(x.shape)
        x = self.fc_layers(x)
        return x


class Classifier_Vision_Feature(nn.Module):
    def __init__(self):
        super(Classifier_Vision_Feature, self).__init__()
        self.cnn_layers = nn.Sequential(

            nn.Conv2d(3, 16, 5, 3, 1), # 3 16 5 3 1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(16, 32, 3, 2, 1), # 16 32 3 2 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1, 1), # 32 64 3 1 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0)
        )
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(6400, 64), # 6400 64
        #     # nn.ReLU(),
        #     # nn.Linear(320, 32),
        #     nn.ReLU(),
        #     nn.Linear(64, 2)
        # )
    def forward(self, x):
        x = self.cnn_layers(x)
        # x = x.flatten(1)
        # print(x.shape)
        # x = self.fc_layers(x)
        return x

class Classifier_Audio(nn.Module):
    def __init__(self):
        super(Classifier_Audio, self).__init__()
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

class Classifier_Audio_HubertLarge(nn.Module):
    def __init__(self):
        super(Classifier_Audio_HubertLarge, self).__init__()
        # self.cnn_layers = nn.Sequential(
        #     # input: 250, 1024
        #     nn.Conv1d(250, 128, 3, stride=1, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.MaxPool1d(2, 2),
            
        #     nn.Conv1d(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.MaxPool1d(2, 2),

        #     nn.Conv1d(64, 32, 3, stride=1, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.MaxPool1d(2, 2),
        # )
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(4096, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # )
        self.fc = nn.Sequential(
            nn.Linear(256000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            # nn.ReLU(),
            # nn.Linear(160, 2)
        )
    def forward(self, x):
        # x = self.cnn_layers(x)
        # print(y.shape)
        x = x.flatten(1)
        # print(y.shape)
        # x = self.fc_layers(x)
        x = self.fc(x)
        return x

class Classifier_All(nn.Module):
    def __init__(self):
        super(Classifier_All, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256000 + 6400, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            # nn.ReLU(),
            # nn.Linear(160, 2)
        )
    def forward(self, x):
        x = self.fc(x)
        return x