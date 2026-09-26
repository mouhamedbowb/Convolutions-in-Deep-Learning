import torch
import torchvision
import torchvision.transforms as T

import numpy as np

import matplotlib.pyplot as plt


cdata =  torchvision.datasets.CIFAR10(root='cifar10', download=True)

print(cdata)

print(cdata.data.shape)

#the unique categories
print(  cdata.classes )

# .targets is a list of targets converted to ints
print(cdata.targets)

fig , axs =plt.subplots(5,5 , figsize=(10,10))


for ax in axs.flatten():
  # select a random picture
  randidx = np.random.choice(len(cdata.targets))
  #extract that image
  pic= cdata.data[randidx,:,:,:] 
  
  #and its label
  label = cdata.classes[cdata.targets[randidx]]
  
  # and show!
  # and show!
  ax.imshow(pic)
  ax.text(16,0,label,ha='center',fontweight='bold',color='k',backgroundcolor='y')
  ax.axis('off')
  
plt.tight_layout()
plt.show() 



#Apply some Transformation

Ts = T.Compose([ T.ToTensor(),
                 T.Resize(32*4),
                 T.Grayscale(num_output_channels=1)  ])


# include the transform in the dataset
cdata.transform = Ts

# you can also apply the transforms immediately when loading in the data
# cdata = torchvision.datasets.CIFAR10(root='cifar10', download=True, transform=Ts)


# Important! Adding a transform doesn't change the image data:
print(cdata.data[123,:,:,:].shape)

# apply the transform

# option 1a: apply the transform "externally" to an image
img1 = Ts( cdata.data[123,:,:,:] )

# option 1b: use the embedded transform
img2 = cdata.transform( cdata.data[123,:,:,:] )

# let's see what we've done!
fig,ax = plt.subplots(1,3,figsize=(10,3))
ax[0].imshow(cdata.data[123,:,:,:])
ax[1].imshow(torch.squeeze(img1))
ax[2].imshow(torch.squeeze(img2),cmap='gray')

plt.show()





