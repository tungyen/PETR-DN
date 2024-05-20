import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

mp = {
    'Car': 0,
    'Van': 1,
    'Pedestrian': 2,
    'Cyclist': 3,
    'Truck': 4,
    'Misc': 5,
    'Tram': 6,
    'Person_sitting': 7
}

class KittiDataset(Dataset):
    def __init__(self, width, height, objectNum=20, transform=None, mode="train"):
        self.transform = transform
        self.mode = mode
        self.width = width
        self.height = height
        self.objectNum = objectNum
            
        if self.mode == 'train':
            self.files = "train.txt"
            self.folderName = "training"
        elif self.mode == 'eval':
            self.files = 'eval.txt'
            self.folderName = "training"
        else:
            self.files = 'test.txt'
            self.folderName = "testing"

        self.names = []
        with open(self.files, "r") as file:
            for line in file:
                self.names.append(line.strip())
                
        self.cams = [folder for folder in os.listdir(os.getcwd()) if len(folder.split('_'))==2]

    def __len__(self):
        return len(self.names)
    
    def getData(self, index, cam):
        filename = self.names[index]
        
        # Image
        root = os.path.join(cam, self.folderName)
        folder = os.listdir(root)[0]
        root = os.path.join(root, folder)
        imgPath = os.path.join(root, filename+".png")
        
        # Calibration
        calibPath = os.path.join("calib", self.folderName, "calib", filename+".txt")
        with open(calibPath, 'r') as file:
            for index, line in enumerate(file):
                values = line.split()[1:]
                values = [float(v) for v in values]
                matrix = np.array(values).reshape(3, -1)
                if index == 2 and cam == "image_left":
                    intrin = matrix
                elif index == 3 and cam == "image_right":
                    intrin = matrix
                if index == 4:
                    rectRot = matrix
        file.close()
        rot = np.eye(3)
        if cam == "image_left":
            t = np.array([-0.06, 0, 0])
        else:
            t = np.array([0.48, 0, 0])
            
        # Bounding boxes
        if self.mode == "test":
            return imgPath, intrin, rectRot, rot, t
        else:
            bboxPath = os.path.join("label", "training", "label_2", filename+".txt")
            bboxes = []
            categories = []
            with open(bboxPath, 'r') as file:
                for line in file:
                    values = line.split()
                    if values[0] == "DontCare":
                        continue
                    categories.append(mp[values[0]])
                    boxInfo = [float(info) for info in values[8:]]
                    bboxes.append(boxInfo)
            return imgPath, intrin, rectRot, rot, t, bboxes, categories
    

    def __getitem__(self, index):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in self.cams:
            if self.mode == "train" or self.mode == 'eval':
                imgPath, intrin, rectRot, rot, t, bboxes, categories = self.getData(index, cam)
            else:
                imgPath, intrin, rectRot, rot, t = self.getData(index, cam)
            img = Image.open(imgPath) # (w, h)
            imgW, imgH = img.size
            img = img.resize((self.width, self.height), resample=Image.NEAREST)
            intrin[0, :] *= ( self.width / imgW)
            intrin[1, :] *= ( self.height / imgH)
            imgs.append(self.transform(img))
            rots.append(torch.Tensor(rot))
            trans.append(torch.Tensor(t))
            intrins.append(torch.Tensor(intrin))
        rectRot = torch.tensor(rectRot)

        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        intrins = torch.stack(intrins)
        trans = torch.stack(trans)
        
        if self.mode == "train" or self.mode == 'eval':
            box3d = torch.zeros(self.objectNum, 7)
            labels = torch.ones(self.objectNum) * -1
            labels[:len(categories)] = torch.tensor(categories)
            box3d[:len(bboxes), :] = torch.tensor(bboxes)
            return imgs, rots, trans, intrins, rectRot, box3d, labels
        else:
            return imgs, rots, trans, intrins, rectRot
                
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = KittiDataset(1280, 640, transform=transform)
    # for data in dataset:
    #     imgs, rots, trans, intrins, rectRot, box3d, labels = data
    #     print(imgs.shape)
    #     print(rots.shape)
    #     print(trans.shape)
    #     print(intrins.shape)
    #     print(rectRot.shape)
    #     break
    
    B = 2
    nw = 0
    trainDataloader = DataLoader(dataset, batch_size=B, shuffle=True, pin_memory=True, num_workers=nw)
    
    for imgs, rots, trans, intrins, rectRot, box3d, labels in trainDataloader:
        print(imgs.shape)
        print(rots.shape)
        print(trans.shape)
        print(intrins.shape)
        print(rectRot.shape)
        break
        
        