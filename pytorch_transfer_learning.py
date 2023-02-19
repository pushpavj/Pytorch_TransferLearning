import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import urllib.request as request
# ROOT=""
# os.chdir(ROOT)
a=torch.Tensor([1,2,3])
print(a)
print("Cuda available: ", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
#Donwload the dataset

#the above data set contains the images of ants and bees. we need to classify them 

#ROOT_DATA_DIR="hymenoptera_data"


class Config:
    def __init__(self):
        self.ROOT_DATA_DIR="hymenoptera_data"
        self.EPOCH=10
        self.BATCH_SIZE=32
        self.LEARNING_RATE=0.001
        self.IMAGE_SIZE=(224,224)
        self.DEVICE="cuda" if torch.cuda.is_available() else "cpu"
        print(f"This code runs on {self.DEVICE}")
        self.SEED=2022
        
    def create_dir(self,dir_path):
        os.makedirs(dir_path,exist_ok=True)
        print(f"Directory {dir_path} created")
        

#Donwload the dataset
data_URL="https://download.pytorch.org/tutorial/hymenoptera_data.zip"
config=Config()
config.create_dir(config.ROOT_DATA_DIR)

data_zip_file="data.zip"
data_file_path=os.path.join(config.ROOT_DATA_DIR,data_zip_file)

request.urlretrieve(data_URL,data_file_path)

#Unzip the data
from zipfile import ZipFile

def unzip_data(source:str,dest:str):
    with ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(dest)
unzip_data(data_file_path,config.ROOT_DATA_DIR)

#Create the data loaders
train_path=os.path.join(config.ROOT_DATA_DIR,config.ROOT_DATA_DIR,"train")
test_path=os.path.join(config.ROOT_DATA_DIR,config.ROOT_DATA_DIR,"val")

#Normalizing the image with mean of the each channel
#for 28 X 28 image size
# mean=sum(value of the pixels)/784
# std=
# data-mean/std 
#this is how the normalization is done in most of the algorithms
mean=torch.tensor([0.5,0.5,0.5])
std=torch.tensor([0.5,0.5,0.5])

train_transforms=transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)])
           

test_transforms=transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)]) 
                #in testing we do not need to do rotation.

train_data=datasets.ImageFolder(train_path,transform=train_transforms)
test_data=datasets.ImageFolder(test_path,transform=test_transforms)

label_map=train_data.class_to_idx #gives the numerical label of each class
train_loader=DataLoader(train_data,batch_size=config.BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_data,batch_size=config.BATCH_SIZE,shuffle=False)

data=next(iter(train_loader))

print(len(data))
images,labels=data
print(images.shape)
print(labels.shape)

#Visalize one of the images
img=images[0]
print(img.shape)
plt.imshow(img.permute(1,2,0)) # the permute is to convert the image from 
                                #HWC to CHW i.e. from 3D to 2D. i.e exchanging
                                # the shape from (3,224,224) to (224,224,3)
                                #  
plt.show()

#Download and use pre_trained model for transfer learning
from torchvision import models
models.alexnet(pretrained=True)

#Alexnet is a pre-trained neural network used for transfer learning. 
#i.e. the model is pre-trained on the ImageNet dataset.
#i.e. the model is trained on the CIFAR10 dataset.
#i.e.the model is able to classify the images of the CIFAR10 dataset.
#i.e. the model is able to classify the images of the ImageNet dataset.
#i.e the ImangeNet dataset is able to classify the images of the various
#kind of images.
#Here we will use the pre-Trained model and make alteration to it.
model= models.alexnet(pretrained=True)
print("model",model)

#count the number of trainable parameters in the model
def count_parameters(model):
    model_parameters = {"Modules":list(),"Parameters":list()}
    total={"trainable":0,"non_trainable":0}
    for name, parameters in model.named_parameters():
        param=parameters.numel()
        if not parameters.requires_grad:
            total["non_trainable"]+=param
            continue
        model_parameters["Modules"].append(name)
        model_parameters["Parameters"].append(param)
        total["trainable"]+=param
    df=pd.DataFrame(model_parameters)
    print(f"Total number parameters: {total}")
    return df

print("count_parameters",count_parameters(model))

#freeze all the layers
for parameters in model.parameters():
    parameters.requires_grad=False #means we don't want to train the 
      # these parameters

print("count_parameters",count_parameters(model)) # now the trainable parameters are 0


print("modelclassifer", model)

model.classifier=nn.Sequential(nn.Linear(
    in_features=9216,out_features=100, bias=True),nn.ReLU(inplace=True),
nn.Dropout(0.5,inplace=False),
nn.Linear(100,2,bias=True))
print("model", model)      

print("Count parameters",count_parameters(model))

#transfer learning model is created as above
#Now we need to train the transfer learning model.

model.to(config.DEVICE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

for epoch in range(config.EPOCH):
    with tqdm(train_loader) as tqdm_epoch:
        for images,labels in tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch {epoch+1}/{config.EPOCH}")

            images=images.to(config.DEVICE)
            labels=labels.to(config.DEVICE)

            #forward pass
            outputs=model(images)
            loss=criterion(outputs,labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_epoch.set_postfix(loss=f"{loss.item():.4f}")
          #  tqdm_epoch.update(1)

#save the model  
os.makedirs("model_dir_2",exist_ok=True)
model_file_path=os.path.join("model_dir_2","CNN_AlexNet.pth")
torch.save(model.state_dict(),model_file_path)


#evaluate the model
pred=np.array([])
target=np.array([])

with torch.no_grad():
    for batch, data in enumerate(test_loader):
        images=data[0].to(config.DEVICE)
        labels=data[1].to(config.DEVICE)
        
        y_pred=model(images)

        pred=np.concatenate((pred,torch.argmax(y_pred,dim=1).cpu().numpy()))
        target=np.concatenate((target,labels.cpu().numpy()))
cm=confusion_matrix(target,pred)

sns.heatmap(cm,annot=True,fmt="d",xticklabels=list(label_map.keys()),
    yticklabels=list(label_map.keys()),cmap="Blues",cbar=False)
plt.show()
         
#prediction on our model

data=next(iter(test_loader))

print(len(data))
print(data[0].shape)

images,labels=data

print(images.shape)
print(labels.shape)

img=images[0]
print(img.shape)

plt.imshow(img.permute(1,2,0)) # the permute is to convert the image from

print(img.unsqueeze(0).shape)

img_on_gpu=img.unsqueeze(0).to(config.DEVICE)
prd_prob=F.softmax(model(img_on_gpu),dim=1)
print("prd_prob",prd_prob)

argmax=torch.argmax(prd_prob).item()
print("argmax",argmax)


inb_label_map={val:key for key, val in label_map.items()}
print("inb_label_map",inb_label_map)

print(inb_label_map[argmax], inb_label_map[labels[0].item()])


  








