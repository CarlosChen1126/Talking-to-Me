import os
import glob
import torch
import numpy as np
from PIL import Image
# from sklearn.decomposition import PCA
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

class Image_Dataset(Dataset):
    def __init__(self, data_root_path='./', mode='train', transform=None):

        self.root = data_root_path
        self.mode = mode

        self.file_name = [] # {video_id}_{person_id}_{start}_{end}_{1/0}

        for i in glob.glob(f'{self.root}/*'):
            video_id = i.split('/')[-1]

            for j in glob.glob(f'{self.root}/{video_id}/*'):
                range_id = j.split('/')[-1]

                # self.file_name.append(f'{video_id}_{range_id}')
                # === !!! ===
                if len(glob.glob(f'{j}/*')) != 0 : self.file_name.append(f'{video_id}_{range_id}')
                # === !!! ===

        if transform != None:
            self.tfm = transform
        else:
            # self.tfm = transforms.Compose([
            #     transforms.Resize((128, 128)),
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # ])
            self.tfm = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
    
    def __getitem__(self, idx):

        file_name = self.file_name[idx]
        video_id, person_id, start, end, tf = file_name.split('_')
        img_num = 0
        one_img_size = []
        total_img_size = []

        for frame_path in glob.glob(f'{self.root}/{video_id}/{person_id}_{start}_{end}_{tf}/*'):
            one_img = Image.open(frame_path).convert("RGB")
            one_img = self.tfm(one_img)

            if img_num == 0:
                one_img_size =  [int(x) for x in one_img.shape] # C * H * W
                image = one_img

            else:
                image = torch.cat( [image, one_img], 0 )
            
            img_num += 1

        # === !!! ===
        # if img_num == 0:
        #     image = torch.zeros([900, 450, 450])
        # === !!! ===

        total_img_size = [int(x) for x in image.shape] # C * H * W
        ch = total_img_size[0]
        ext = image

        # === !!! ===

        while ch < 256:
            if 256 - ch >= total_img_size[0]:
                ext = torch.cat( [ext, image], 0 )
                ch = ch + total_img_size[0]
            else:
                ext = torch.cat( [ext, image[:(256-ch), :, :]], 0 )
                ch = 256
        image = ext

        if ch > 256:
            image = ext[:256, :, :]

        # print(image.size())
        # print(file_name)
        # print(img_num, one_img_size, total_img_size)
        # === !!! ===

        return image, file_name

    def __len__(self):

        return len(self.file_name)

class Audio_Dataset(Dataset):
    def __init__(self, data_root='./', mode='train', transform=None):
        
        self.data_root = data_root
        self.mode = mode

        if transform != None:
            self.transform = transform

        self.file_name = []

        for i in glob.glob(f'{self.data_root}/*'):
            self.file_name.append(i.split('/')[-1]) # {video_id}_{start}_{end}.npy

    def __getitem__(self, idx):

        file_name = self.file_name[idx]
        # whole_feature = np.load(f'{self.root}/{file_name}', allow_pickle=True)
        # feature = self.pca.fit_transform(whole_feature)
        feature = np.load(f'{self.data_root}/{file_name}', allow_pickle=True)
        feature = torch.from_numpy(feature)
        # feature = self.f_tfm(feature)

        # === !!! ===

        total_size = [int(x) for x in feature.shape] # C * H * W
        frm = total_size[0]
        ext = feature

        while frm < 256:
            if 256 - frm >= total_size[0]:
                ext = torch.cat( [ext, feature], 0 )
                frm = frm + total_size[0]
            else:
                ext = torch.cat( [ext, feature[:(256-frm), :, :]], 0 )
                frm = 256
        feature = ext

        if frm > 256:
            feature = ext[:256, :, :]

        # === !!! ===

        return feature, file_name

    def __len__(self):

        return len(self.file_name)

### TODO
class Audio_HubertLarge_Dataset(Dataset):
    def __init__(self, data_root='./', mode='train', transform=None):
        
        self.data_root = data_root
        self.mode = mode

        if transform != None:
            self.transform = transform

        self.file_names = []

        for fp in glob.glob(os.path.join(self.data_root, self.mode, "*")):
            self.file_names.append(fp.split('/')[-1])   # {video_id}_{start}_{end}_{ttm}.pt (tensor on cuda)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        feature = torch.load(os.path.join(self.data_root, self.mode, file_name)).squeeze(0)    # (H, W) = (250, 1024)

        return feature, file_name[:-3]

    def __len__(self):

        return len(self.file_names)



class All_Dataset(Dataset):
    def __init__(self, data_root='./', feature_root='./', mode='train', transform=None):

        self.data_root = data_root
        self.feature_root = feature_root

        self.mode = mode

        self.file_names = [] # {video_id}_{person_id}_{start}_{end}_{1/0}

        for i in glob.glob(os.path.join(self.data_root, "*")):
            video_id = i.split('/')[-1]

            for j in glob.glob(os.path.join(self.data_root, video_id, "*")):
                range_id = j.split('/')[-1]

                # self.file_name.append(f'{video_id}_{range_id}')
                # === !!! ===
                if len(glob.glob(f'{j}/*')) != 0 : self.file_names.append(f'{video_id}_{range_id}')
                # === !!! ===

        if transform != None:
            self.tfm = transform
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        if self.mode == 'train': 
            video_id, person_id, start, end, ttm = file_name.split('_')
        elif self.mode == 'test': 
            video_id, person_id, start, end = file_name.split('_')

        image = None

        if self.mode == 'train': 
            whole_root = os.path.join(
                self.data_root, 
                video_id, 
                f"{person_id}_{start}_{end}_{ttm}", "*"
            )
        elif self.mode == 'test': 
            whole_root = os.path.join(
                self.data_root, 
                video_id, 
                f"{person_id}_{start}_{end}", "*"
            )

        files = glob.glob(whole_root)
        mid = 0
        if len(files) % 2 == 0:
            mid = len(files) / 2
        elif len(files) % 2 == 1:
            mid = (len(files)+1) / 2

        for i, frame_path in enumerate(files, 1):
            if i == mid:
                one_img = Image.open(frame_path).convert("RGB")
                image = self.tfm(one_img)

        if self.mode == 'test' and image == None: 
            image = torch.div(torch.ones([3, 512, 512]), 1e8)

        # ThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThree
        # if Three:
        #     first = 1
        #     last = len( glob.glob(whole_root) )
        #     for i, frame_path in enumerate( glob.glob(whole_root) ):
        #         idx = i + 1
        #         if idx == first:
        #             one_img = Image.open(frame_path).convert("RGB")
        #             image = self.tfm(one_img)
        #             image = torch.div(image, 3) # 1/3 !
        #         if idx == mid:
        #             one_img = Image.open(frame_path).convert("RGB")
        #             one_img = self.tfm(one_img)
        #             one_img = torch.div(one_img, 3) # 1/3 !
        #             # image = torch.cat( [image, one_img], 0 )
        #             image = image + one_img # 1/3 !
        #         if idx == last:
        #             one_img = Image.open(frame_path).convert("RGB")
        #             one_img = self.tfm(one_img)
        #             one_img = torch.div(one_img, 3) # 1/3 !
        #             # image = torch.cat( [image, one_img], 0 )
        #             image = image + one_img # 1/3 !

        #     if self.mode == 'test' and image == None : image = torch.div(torch.ones([3, 512, 512]), 1e8)
        # ThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThreeThree


        # === Audio Feature ===
        try:
            if self.mode == 'train': 
                feature = torch.load(
                    os.path.join(
                        self.feature_root, 
                        self.mode, 
                        f"{video_id}_{start}_{end}_{ttm}.pt"
                    )
                ).squeeze(0).cpu()
            elif self.mode == 'test':
                feature = torch.load(
                    os.path.join(
                        self.feature_root, 
                        self.mode, 
                        f"{video_id}_{start}_{end}.pt"
                    )
                ).squeeze(0).cpu()
        except:
            feature = torch.div(torch.ones([250, 1024]), 1e8)

        return image, file_name, feature

    def __len__(self):

        return len(self.file_names)
