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

#This module is to perform the classification of the image data through the 
# AlexNet CNN architecture transfer learning using PyTorch.


# ROOT=""
# os.chdir(ROOT)

#Below code is to check if the cuda is available or not.

a=torch.Tensor([1,2,3])
print(a)
print("Cuda available: ", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)



#ROOT_DATA_DIR="hymenoptera_data"

# The below class is used to define the constants and also to create the
# directory for the given path.

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
#the above data set contains the images of ants and bees. we need to
#  classify them 

data_zip_file="data.zip"
data_file_path=os.path.join(config.ROOT_DATA_DIR,data_zip_file)

request.urlretrieve(data_URL,data_file_path) # downloads the zipped file from
# the given url and saves it in the given directory path. 
# i.e to D:\user\jupyternotes\Praketh\CV-Computer vision\CV-5_pytorch
# \Pytorch_TransferLearning\hymenoptera_data\data.zip



#Unzip the data
from zipfile import ZipFile

def unzip_data(source:str,dest:str):
    with ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(dest)
unzip_data(data_file_path,config.ROOT_DATA_DIR) #unzip the zipped file into 
 # the same root directory as 
 # D:\user\jupyternotes\Praketh\CV-Computer vision\CV-5_pytorch\
 # Pytorch_TransferLearning\hymenoptera_data\hymenoptera_data 
 # and this has the sub directories of ants and bees as train and val data sets
 #D:\user\jupyternotes\Praketh\CV-Computer vision\CV-5_pytorch\
 # Pytorch_TransferLearning\hymenoptera_data\hymenoptera_data\train
 #
 #D:\user\jupyternotes\Praketh\CV-Computer vision\CV-5_pytorch\
 #Pytorch_TransferLearning\hymenoptera_data\hymenoptera_data\val



#Create the data loaders
train_path=os.path.join(config.ROOT_DATA_DIR,config.ROOT_DATA_DIR,"train")
test_path=os.path.join(config.ROOT_DATA_DIR,config.ROOT_DATA_DIR,"val")
#get the train and val directory paths

#Normalizing the image with mean of the each channel
#for 28 X 28 image size i.e. 28 X 28 = 784

# mean=sum(value of the pixels)/784
# std=
# data-mean/std 
#this is how the normalization is done in most of the algorithms
mean=torch.tensor([0.5,0.5,0.5])
std=torch.tensor([0.5,0.5,0.5])
#build the transforms which will be applied to the data as a preprocess
#in this case we will resize the image to 224 X 224 X 3
#Then do data argumentation on the image i.e. random rotation with 30 degrees
#Then do the normalization using the mean and standard deviation


# normalize the image with mean and std
#
train_transforms=transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)])
           
#similarly we create the transforms for test data.
test_transforms=transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)]) 
                #in testing we do not need to do rotation.
#apply the transforms to the data and get the transformed train and test data


train_data=datasets.ImageFolder(train_path,transform=train_transforms)
test_data=datasets.ImageFolder(test_path,transform=test_transforms)

label_map=train_data.class_to_idx #gives the numerical label of each class
#create the train and test data loaders

train_loader=DataLoader(train_data,batch_size=config.BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_data,batch_size=config.BATCH_SIZE,shuffle=False)

#get the first batch of the train data

data=next(iter(train_loader))

print(len(data)) #gives the length of data as 2 one for input pixels and 
# one for corresponding label.


images,labels=data
print("images.shape",images.shape)
print("labels.shape",labels.shape)

#Visalize one of the images
img=images[0]
print("img.shape",img.shape)
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
print("model",model) #Alexnet is a pre-trained neural network used for 
#transfer learning. 

#model AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))        
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))        
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))       
#     (7): ReLU(inplace=True)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))       
#     (9): ReLU(inplace=True)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))      
#     (11): ReLU(inplace=True)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
#   (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )

#count the number of trainable and non trainable parameters in the model
def count_parameters(model):
    model_parameters = {"Modules":list(),"Parameters":list()}
    total={"trainable":0,"non_trainable":0}
    for name, parameters in model.named_parameters():
       # print("name, Parameters",name,parameters)
        param=parameters.numel() #numel is the number of elements 
         #in the tensor

        if not parameters.requires_grad: #not requires grad means that the 
            # tensor is not trainable

            total["non_trainable"]+=param 
            continue
        model_parameters["Modules"].append(name)
        model_parameters["Parameters"].append(param)
        total["trainable"]+=param
    df=pd.DataFrame(model_parameters)
    print(f"parameters of the model are {df}")

    print(f"Total number parameters: {total}")
    return df

print("count_parameters",count_parameters(model))

#freeze all the layers. 
for parameters in model.parameters():
    parameters.requires_grad=False #means we don't want to train the 
      # these parameters

print("count_parameters",count_parameters(model)) # now the trainable
# parameters are 0


print("modelclassifer", model)
#update the model classifier with the new model as below
model.classifier=nn.Sequential(nn.Linear(
    in_features=9216,out_features=100, bias=True),nn.ReLU(inplace=True),
nn.Dropout(0.5,inplace=False),
nn.Linear(100,2,bias=True))
print("model", model) 
print("model.classifier", model.classifier)
# odel AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))   
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))   
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
#     (7): ReLU(inplace=True)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
#     (9): ReLU(inplace=True)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
#     (11): ReLU(inplace=True)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
#   (classifier): Sequential(
#     (0): Linear(in_features=9216, out_features=100, bias=True)
#     (1): ReLU(inplace=True)
# model.classifier Sequential(
#   (0): Linear(in_features=9216, out_features=100, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=100, out_features=2, bias=True)
# )     

print("Count parameters",count_parameters(model))

#transfer learning model is created as above
#Now we need to train the transfer learning model.

model.to(config.DEVICE) #put the model on the device i.e.GPU
criterion=nn.CrossEntropyLoss() #loss function

optimizer=torch.optim.Adam(model.parameters()) #Adam optimizer is used to 
                                #train the model


for epoch in range(config.EPOCH): #loop over the number of epochs 10

    with tqdm(train_loader) as tqdm_epoch: #tqdm is a progress bar
        #tqdm_epoch contains a train_loader batch data  
        # i.e. each batch contains 32 images
        #i.e. number of train_loader will run for number of epochs i.e. 10
        #i.e. total 10 epoch will run for 32 images 


        for images,labels in tqdm_epoch: #this loop runs for 32 images
            tqdm_epoch.set_description(f"Epoch {epoch+1}/{config.EPOCH}")

            images=images.to(config.DEVICE) #put the images on the device 
            #i.e.GPU

            labels=labels.to(config.DEVICE) #put the labels on the device GPU

            #forward pass
            outputs=model(images) #train the model with the images
            loss=criterion(outputs,labels) #calculate the loss

            #backward pass
            optimizer.zero_grad() #zero the gradient

            loss.backward() #calculate the gradient

            optimizer.step() #update the parameters with the gradient


            tqdm_epoch.set_postfix(loss=f"{loss.item():.4f}") #update the 
            #tqdm progress bar

          #  tqdm_epoch.update(1)

#save the model  
os.makedirs("model_dir_2",exist_ok=True) #create the directory to 
                                            #save the model

model_file_path=os.path.join("model_dir_2","CNN_AlexNet.pth")
torch.save(model.state_dict(),model_file_path) #save the model



#evaluate the model using confusion matrix
pred=np.array([])
target=np.array([])

with torch.no_grad(): # no_grad is used to preven the gradient calculation 

    for batch, data in enumerate(test_loader): # enumerate is used to
                                            # iterate over the test_loader
 

        images=data[0].to(config.DEVICE) #take the 32 images and put it on
                                        # the GPU device
        labels=data[1].to(config.DEVICE)#take the corresponding labels and 
                                        #put it on the GPU device


        y_pred=model(images) #predict the label of the 32 images using 
                             # trained model
                            #since the model is a classification model,
                            #the prediction will have a probobility of each 
                            # class
        #torch.argmax(y_pred,dim=1) is used to get the index of the label 
        # which has the highest probability
        print("y_pred",y_pred)
        print("y_pred.shape",y_pred.shape)

# for each image in the test_loader get the predicted label and the 
# corresponding label details to form the confusion matrix



        pred=np.concatenate((pred,torch.argmax(y_pred,dim=1).cpu().numpy()))
        print("pred",pred)

        target=np.concatenate((target,labels.cpu().numpy()))
#form the confusion matrix to evaluate the accuracy of the model


cm=confusion_matrix(target,pred)
#display the confusion matrix

sns.heatmap(cm,annot=True,fmt="d",xticklabels=list(label_map.keys()),
    yticklabels=list(label_map.keys()),cmap="Blues",cbar=False)
plt.show()
         


#prediction on our model using the softmax function

data=next(iter(test_loader)) #test_loader is a iterator over the test data


print("len(data)",len(data))

print("data[0].shape",   data[0].shape)

images,labels=data #gets the 32 images and the corresponding labels


print("images.shape",images.shape)

print("labels.shape",labels.shape)


img=images[0] #gets the first image

print("img.shape",img.shape)


plt.imshow(img.permute(1,2,0)) # the permute is to exchange the order of the 
# channels and the order of the images i.e 1st and 3rd dimensions are inter
# changed 


print("img.unsqueeze(0).shape",img.unsqueeze(0).shape) 
                    #unsqueeze is used to add a new dimension to the image 




img_on_gpu=img.unsqueeze(0).to(config.DEVICE) #put the image on the GPU device

prd_prob=F.softmax(model(img_on_gpu),dim=1) 
# the softmax function is used to get the probability of each class

print("prd_prob",prd_prob)

argmax=torch.argmax(prd_prob).item() #argmax is used to get the index of 
            #the label which has the highest probability


print("argmax",argmax) #argmax is the highest probability



inb_label_map={val:key for key, val in label_map.items()}
print("inb_label_map",inb_label_map)

print(f"inb_label_map[argmax] {inb_label_map[argmax]}," 
           f" inb_label_map[labels[0].item()] {inb_label_map[labels[0].item()]}",)

#Create the function to predict the class of an image
def predict_class(data, model,label_map,device,idx=0):
    images,labels=data
    img=images[idx]
    label=labels[idx]
    
    plt.imshow(img.permute(1,2,0)) # the permute is to convert the image from
    img_on_gpu=img.unsqueeze(0).to(device)

    pred_prob=F.softmax(model(img_on_gpu),dim=1)
    argmax=torch.argmax(pred_prob).item()

    pred_label=inb_label_map[argmax]
    actual_label=label_map[label.item()]

    plt.title(f"Predicted: {pred_label}, Actual: {actual_label}")
    plt.axis("off")
    plt.show()
    return pred_label,actual_label

predict_class(data, model,inb_label_map,config.DEVICE,idx=1)

#sub sampling means we are going to use a subset of the data


  








