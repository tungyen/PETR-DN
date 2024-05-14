import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class KittiDataset(Dataset):
    def __init__(self, transform=None, mode="train"):
        self.transform = transform
        self.mode = mode
            
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
            trans = np.array([-0.06, 0, 0])
        else:
            trans = np.array([0.48, 0, 0])
            
        # Bounding boxes
        if self.mode == "test":
            return imgPath, intrin, rectRot, rot, trans
        else:
            bboxPath = os.path.join("label", "training", "label_2", filename+".txt")
            bboxes = []
            categories = []
            with open(bboxPath, 'r') as file:
                for line in file:
                    values = line.split()
                    categories.append(values[0])
                    bboxes.append([values[3]]+values[8:-1])
            return imgPath, intrin, rectRot, rot, trans, bboxes, categories
    

    def __getitem__(self, index):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in self.cams:
            if self.mode != "test":
                imgPath, intrin, rectRot, rot, trans, bboxes, categories = self.getData(index, cam)
                
                
                
            else:
                imgPath, intrin, rectRot, rot, trans = self.getData(index, cam)
                
        return imgPath        
                
if __name__ == '__main__':
    dataset = KittiDataset()
    for data in dataset:
        print(data)
        
        
        