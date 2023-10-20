import os
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from models import Classifier_Vis, Classifier_Aud
from datasets import Image_Dataset, Audio_Dataset, All_Dataset
from parsers import final_tarin_parser

NUM_EPOCHS = 5
BATCH_SIZE = 64

def Train_One_Epoch(model, train_loader, criterion):
        model.train()

        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            images, file_names, features = batch

            images = images.to(device)
            features = features.to(device)
            
            labels = []
            for i in range( len(file_names) ):
                labels.append( int( file_names[i].split('_')[-1] ) )
            labels = torch.tensor(labels, dtype=torch.long)

            labels = labels.to(device)

            if args.final_train_type == 'vision':
                logits = model(images)
            elif args.final_train_type == 'audio':
                logits = model(features)

            logits = logits.to(device)
            # print(model.cnn_layers[0].weight.data)
            # print(labels)
            # print(logits)
            # raise BaseException('stop !')

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{NUM_EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

if __name__ == "__main__":
    args = final_tarin_parser.parse_args()

    if args.final_train_type == 'vision':
        model = Classifier_Vision()
    elif args.final_train_type == 'audio':
        model = Classifier_Audio()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.1, last_epoch = -1)

    train_set = All_Dataset(
        data_root = args.final_vision_train_dir, 
        feature_root = args.final_audio_feat_dir
    )
    
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])

    print(
        'train set: ', len(train_set), 
        'valid set: ', len(valid_set)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

    # === MAIN ===
    for epoch in range(NUM_EPOCHS):
        Train_One_Epoch(model, train_loader, criterion)
        # Validate(model, valid_loader)
        torch.save(model.state_dict(), f'CKPT/{args.final_train_type}_feature_{epoch}_1e-3.ckpt')