import os

import torch
import torchnet as tnt
import torch.nn as nn


from models import LeNet
from optim import Adaptive
import dataloader
from dataloader import mnist

##################################
## Initialization of parameters ##
##################################

## Choose batch sizes
train_batch_size = 128
test_batch_size = 1000

## Choose the data set
train_loader = mnist(train_batch_size, train = True)
test_loader = mnist(test_batch_size, train = False)

## Pull the model architecture, and set the loss function, initialize penalties
model = LeNet()
class CELog(nn.Module):
    def __init__(self):
        super().__init__()
        self.nl = nn.NLLLoss()
    def forward(self, x, y):
        return self.nl(x.log(), y)
criterion = CELog()

## Choose optimizer to be Adaptive
optimizer = Adaptive(model.parameters(), max_lr = 0.1, fstar = 0.0,
        window = 20*int(60000/train_batch_size))



##################################################
## Train and test the model over several epochs ##
##################################################

## Define the function to train the model
def train(e):
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    for batch_ix, (data, target) in enumerate(train_loader):
   
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[0], target)
        loss.backward()
          
        # Using a running average for the optimizer
        if batch_ix == 0:
            runavg_loss = loss.item()
        else:
            runavg_loss = 0.9*runavg_loss + 0.1*loss.item()

        optimizer.step(runavg_loss)
          
        if (batch_ix % 100 == 0 and batch_ix > 0):
                print('[Epoch %2d, batch %3d] penalized training loss: %.3g' %
                    (e, batch_ix, loss.data.item()))
    return loss

## Define the function to test the model
def test(e):
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():

        # Get the true training loss and error
        top1_train = tnt.meter.ClassErrorMeter()
        train_loss = tnt.meter.AverageValueMeter()
        for data, target in train_loader:
            output = model(data)
            top1_train.add(output[0].data, target.data)
            loss = criterion(output[0], target)
            train_loss.add(loss.data.item())
        t1t = top1_train.value()[0]
        lt = train_loss.value()[0]

        # Functions for tracking test loss, error
        test_loss = tnt.meter.AverageValueMeter()
        top1 = tnt.meter.ClassErrorMeter()
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output[0], target)
            top1.add(output[0], target)
            test_loss.add(loss.item())
        t1 = top1.value()[0]
        l = test_loss.value()[0]

    # Report results
    print('[Epoch %2d] Average test loss: %.3f, error: %.2f%%'
            %(e, l, t1))
    print('%28s: %.3f, error: %.2f%%\n'
            %('training loss',lt,t1t))

    return test_loss.value()[0], top1.value()

## Run the above functions

save_path = 'log/'
save_model_path = os.path.join(save_path, 'checkpoint.pth.tar')

for e in range(100):

    train_loss = train(e)
    loss, pct_err = test(e)
    
    torch.save({'epoch': e + 1,
                'state_dict':model.state_dict(),
                'pct_err': pct_err,
                'loss': loss,
                'optimizer': optimizer.state_dict()}, save_model_path)
