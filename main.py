import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from data_utils import DatasetFromFolder
from tensorboardX import SummaryWriter
from model import VDSR

device=torch.device('cuda:0')
writer=SummaryWriter('D:/VDSR')

transform=T.ToTensor()

trainset=DatasetFromFolder('D:/train_data/291',transform=transform)
trainLoader=DataLoader(trainset,batch_size=128,shuffle=True)


net=VDSR()
net=net.to(device)

optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
criterion=nn.MSELoss()
criterion=criterion.to(device)

net.train()
for epoch in range(20):

    running_cost=0.0
    for i,data in enumerate (trainLoader,0):
        input,target=data
        input,target=input.to(device),target.to(device)
        optimizer.zero_grad()
        output=net(input)
        loss=criterion(output,target)
        loss.backward()
        if optimizer=='SGD':
            nn.utils.clip_grad_norm(net.parameters(),0.4)
        optimizer.step()
        running_cost+=loss.item()
        torch.save(net.state_dict(),'VDSR.pth')


        if i%10 == 9 :
            print('epoch:%d, loss:%.8f'%(epoch,running_cost/10))
            writer.add_scalar('loss', running_cost/10,epoch*len(trainLoader)+i)
            running_cost=0.0
    scheduler.step()

writer.close()



