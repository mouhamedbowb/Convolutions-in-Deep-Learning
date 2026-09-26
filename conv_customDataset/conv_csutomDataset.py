# import libraries
import numpy as np
import torch

# import transformations and dataset/loader
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader

import matplotlib.pyplot as plt



#import dataset

mnist = torchvision.datasets.MNIST(root='.data' , train = True , download=True)

#volontairement un tout petit échantillon, car l'objectif est pédagogique (visualiser l'effet des transformations), pas d'entraîner un modèle.

labels = mnist[:8,0]
data= mnist[:8,1:]



# reshape to 2D!
dataNorm = data.reshape(data.shape[0],1,28,28)
print(dataNorm.shape)
print(labels.shape)


#convert to  torch format


dataT = torch.tensor(dataNorm).float()
labelsT=torch.tensor(labels).long() 



class customDataset(Dataset):
  def __init__(self,tensors,transform=None):
     assert all (tensors[0].size(0) == t.size(0) for t in tensors), "size mismmatch"
     #On stocke simplement les entrées comme attributs de l'objet,
     # pour pouvoir y accéder plus tard dans les autres méthodes.
     self.tensors=tensors
     self.transform = transform
     
     
  def __getitem__(self,index):
    if self.transform:
       x=self.transform((self.tensors[0][index]))
    
    else:
         x=self.tensors[0][index]
         
    y=self.tensors[1][index]  
    
    return x,y
  
  def __len__(self):
    return self.tensors[0].size(0)
  
  # Note: several transforms work only on PIL-format data, so it's common to transformo PIL, apply transformations, then transform back to tensor.
  
  
imgTransf = T.Compose(
    T.ToPILImage(),
    T.RandomVerticalFlip(p=0.5),
    #T.RandomRotation()
  ) 


# now convert the data into datasets and then dataloaders

# convert into PyTorch Datasets
# NOTE: we have no test data here, but you should apply the same transformations to the test data
train_data = customDataset((dataT,labelsT),imgTransf)


# translate into dataloader objects
dataloaded = DataLoader(train_data , batch_size=8 , shuffle=False)
  

X,y = next(iter(dataloaded))
# create a figure
fig,axs = plt.subplots(2,8,figsize=(16,4))


# loop over images in the dataset
for i in range(8):

  # draw images
  axs[0,i].imshow(dataT[i,0,:,:].detach(),cmap='gray')
  axs[1,i].imshow(X[i,0,:,:].detach(),cmap='gray')

  # some niceties
  for row in range(2):
    axs[row,i].set_xticks([])
    axs[row,i].set_yticks([])

# row labels
axs[0,0].set_ylabel('Original')
axs[1,0].set_ylabel('torch dataset')
  
  
  
  
  
     
        
      
    
       
     
     
     
     
     
    




