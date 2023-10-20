import os
import logging
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from models import Classifier_Vision, Classifier_Vision_Feature, Classifier_Audio, Classifier_Audio_HubertLarge, Classifier_All
from datasets import Image_Dataset, Audio_Dataset, Audio_HubertLarge_Dataset, All_Dataset
from parsers import final_tarin_parser

NUM_EPOCHS = 10
BATCH_SIZE = 64

def Train_One_Epoch(model, train_loader, criterion, model_vis=None, optimizer_vis=None):
		if model_vis != None:
			model_vis.train()
		model.train()

		train_loss = []
		train_accs = []

		for batch in tqdm(train_loader):
			if args.final_train_type == 'vision':
				images, file_names = batch
				images = images.to(device)
			elif args.final_train_type == 'audio':
				features, file_names = batch
				features = features.to(device)
			elif args.final_train_type == 'all':
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
			elif args.final_train_type == 'all':
				# TODO
				vision_features = model_vis(images)		
				vision_features = vision_features.flatten(1).to(device)		# flattened vision feat (batch size, 6400)
				features = features.flatten(1).to(device)					# flattened audio feat (batch size, 256000)
				
				features = torch.cat([features, vision_features], 1).to(device)
				logits = model(features)

			logits = logits.to(device)

			loss = criterion(logits, labels)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			if optimizer_vis != None:
				optimizer_vis.step()
				optimizer_vis.zero_grad()

			acc = (logits.argmax(dim=-1) == labels).float().mean().item()
			train_loss.append(loss.item())
			train_accs.append(acc)

		train_loss = sum(train_loss) / len(train_loss)
		train_acc = sum(train_accs) / len(train_accs)
		print(f"[ Train | {epoch + 1:03d}/{NUM_EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
		logging.info(f"[ Train | {epoch + 1:03d}/{NUM_EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

def Validate(model, valid_loader, model_vis=None):
		if model_vis != None:
			model_vis.eval()
		model.eval()

		valid_accs = []

		with torch.no_grad():
			for batch in tqdm(valid_loader):
				if args.final_train_type == 'vision':
					images, file_names = batch
					images = images.to(device)
				elif args.final_train_type == 'audio':
					features, file_names = batch
					features = features.to(device)
				elif args.final_train_type == 'all':
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
				elif args.final_train_type == 'all':
					# TODO
					vision_features = model_vis(images)		
					vision_features = vision_features.flatten(1).to(device)		# flattened vision feat (batch size, 6400)
					features = features.flatten(1).to(device)					# flattened audio feat (batch size, 256000)

					features = torch.cat([features, vision_features], 1).to(device)
					logits = model(features)
				
				logits = logits.to(device)

				acc = (logits.argmax(dim=-1) == labels).float().mean().item()
				valid_accs.append(acc)

		valid_acc = sum(valid_accs) / len(valid_accs)
		print(f"[ Validation | {epoch + 1:03d}/{NUM_EPOCHS:03d} ], acc = {valid_acc:.5f}")
		logging.info(f"[ Validation | {epoch + 1:03d}/{NUM_EPOCHS:03d} ], acc = {valid_acc:.5f}")

if __name__ == "__main__":
	parser = final_tarin_parser()
	args = parser.parse_args()

	if not os.path.exists('./CKPT'):
		os.makedirs('./CKPT', exist_ok=True)

	logging.basicConfig(filename=f'final_{os.getpid()}.log', level=logging.INFO)

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print('Device used:', device)

	if args.final_train_type == 'vision':
		model = Classifier_Vision().to(device)
	elif args.final_train_type == 'audio':
		model = Classifier_Audio_HubertLarge().to(device)
		state = torch.load('./CKPT/audio_feature_2_1e-3.ckpt')
		model.load_state_dict(state)
	elif args.final_train_type == 'all':
		# TODO
		model_vis = Classifier_Vision_Feature()
		# load and freeze vision model
		model_vis_state = torch.load("./VIS_CKPT/A2_vis_feature_30_1e-4_one_vis.ckpt")
		model_vis.load_state_dict(model_vis_state, strict=False)	# need features only
		# for param in model_vis.parameters():
		# 	param.requires_grad = False
		model_vis = model_vis.to(device)
		optimizer_vis = torch.optim.Adam(model_vis.parameters(), lr=1e-3, weight_decay=1e-4)
		scheduler_vis = torch.optim.lr_scheduler.MultiStepLR(optimizer_vis, milestones=[5], gamma=0.2, last_epoch = -1)

		model = Classifier_All()

	model = model.to(device)
	print(model)
	logging.info(model)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.2, last_epoch = -1)

	if args.final_train_type == "all":
		train_set = All_Dataset(
			data_root = args.final_vision_train_dir, 
			feature_root = args.final_audio_feat_dir,
			mode="train"
		)
	elif args.final_train_type == "audio":
		train_set = Audio_HubertLarge_Dataset(
			data_root = args.final_audio_feat_dir,
			mode='train'
		)
	elif args.final_train_type == "vision":
		train_set = Image_Dataset(
			data_root = args.final_vision_train_dir,
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
		if args.final_train_type == "all":
			Train_One_Epoch(model, train_loader, criterion, model_vis=model_vis, optimizer_vis=optimizer_vis)
			Validate(model, valid_loader, model_vis=model_vis)
			scheduler.step()
			scheduler_vis.step()

			torch.save(model_vis.state_dict(), f'./CKPT/{args.final_train_type}_vis_feat_{epoch}_1e-3.ckpt')
			torch.save(model.state_dict(), f'./CKPT/{args.final_train_type}_classifier_{epoch}_1e-3.ckpt')
		else:
			Train_One_Epoch(model, train_loader, criterion)
			Validate(model, valid_loader)
			scheduler.step()

			torch.save(model.state_dict(), f'./CKPT/{args.final_train_type}_feature_{epoch}_1e-3.ckpt')