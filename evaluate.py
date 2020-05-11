import torch
import matplotlib.pyplot as plt
from model import VDSR
import cv2
import torchvision.transforms as T
import numpy as np
import math




device=torch.device('cuda:0')
transform=T.ToTensor()
net=VDSR()
checkpoint=torch.load('D:/VDSR_SGD_epoch_60.pth')
net.load_state_dict(checkpoint['model_state_dict'])
net=net.to(device)
net.eval()
image_path='D:/train_data/91/000tt16.bmp'
img=cv2.imread(image_path)
img_r=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

img=cv2.resize(img,(img.shape[1],img.shape[0]//2),interpolation=cv2.INTER_CUBIC)

#img_original=img_r[200:230,300:330]
Y,Cr,Cb=cv2.split(img)

patch=Y[200:230,300:330]


plt.imshow(img_r)
plt.show()
img=transform(Y)
img=torch.unsqueeze(img,dim=0)
img=img.to(device)
output=net(img)

output=torch.squeeze(output,dim=0).detach().cpu().numpy()

Y_trans=np.squeeze(output,axis=0)*255.0
Y_trans=np.clip(Y_trans,0,255)
Y_trans=np.array(Y_trans,dtype=np.uint8)

output=cv2.merge((Y_trans,Cr,Cb))
output=cv2.cvtColor(output,cv2.COLOR_YCrCb2RGB)

#output=output[220:250,330:360]

plt.imshow(output)
plt.show()

