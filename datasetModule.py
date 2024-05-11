#TODO delete imports
import os, json, cv2, numpy as np, matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch.utils.data import Dataset, DataLoader


from torchvision.transforms import functional as F



class ClassDataset(Dataset):
   def __init__(self, root, transform=None, demo=False):              
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "labels"))) #os.path.join(root, "labels")
    
   def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "labels", self.annotations_files[idx]) #self.root

        img_original = mpimg.imread(img_path)
        imgHeight, imgWidth, _ = img_original.shape
        #img_original = mpimg.cvtColor(img_original, mpimg.COLOR_BGR2RGB)        
        
        with open(annotations_path, 'r') as file:
            anotationDataString = file.read().split()
            anotationData = [float(x) for x in anotationDataString]
            #boundry rectangle 
            boundryBoxStartX = (imgWidth * anotationData[1]) - (imgWidth * anotationData[3])/2 #(middle of rectangel) - size/2 
            boundryBoxStartY = (imgHeight * anotationData[3]) + (imgHeight * anotationData[4])/2
            boundryBoxSizeX =  imgWidth * anotationData[3]
            boundryBoxSizeY = imgHeight * anotationData[4]

            bboxes_original =[boundryBoxStartX,boundryBoxStartY-boundryBoxSizeY,boundryBoxStartX+boundryBoxSizeX,boundryBoxStartY+boundryBoxSizeY]
            bboxes_original = np.array(bboxes_original) #convert to numpy array 


            startingKeypointX = imgWidth * anotationData[5]
            startingKeypointY = imgHeight * anotationData[6]

            centerKeypointX = imgWidth * anotationData[8]
            centerKeypointY = imgHeight * anotationData[9]

            topKeypointX = imgWidth * anotationData[23]
            topKeypointY = imgHeight * anotationData[24]

            keypoints_original = [[startingKeypointX,startingKeypointY,2],[centerKeypointX,centerKeypointY,2],[topKeypointX,topKeypointY,2]]

            # All objects are glue tubes
        #bboxes_labels_original = ['Glue tube' for _ in bboxes_original]            

        img, bboxes, keypoints = img_original, bboxes_original, keypoints_original        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        #changing size of tensor for [4] to [1,4] for compatibility with the model 
        bboxes = bboxes.unsqueeze(0)  

        #bboxes = [1,bboxes]      
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        """
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64) # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)        
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:"""


        return img, target
    
   def __len__(self):
        return len(self.imgs_files)
def tuple_batch(batch):
    return tuple(zip(*batch))
