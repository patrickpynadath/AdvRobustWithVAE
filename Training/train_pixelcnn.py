'''
Code by Hrituraj Singh
Indian Institute of Technology Roorkee
'''

import sys
import os
import time
import torch
from torch import optim
from torch.utils import data
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from Models.PixelCNN import PixelCNN, discretized_mix_logistic_loss
from tqdm import tqdm


def train_pixel_cnn(epochs, net : PixelCNN, device : str, trainloader : DataLoader):


    optimizer = optim.Adam(net.parameters())


    loss_overall = []
    time_start = time.time()
    print('Training Started')

    for i in range(epochs):

        net.train(True)
        step = 0
        loss_ = 0
        datastream = tqdm(enumerate(trainloader, 1), total=len(trainloader), position=0, leave=True)
        for _, (images, labels) in datastream:

            images = images.to(device)

            optimizer.zero_grad()

            output = net(images)
            loss = discretized_mix_logistic_loss(images, output)
            loss.backward()
            optimizer.step()

            loss_ += loss
            step += 1

            if (step % 100 == 0):
                print('Epoch:' + str(i) + '\t' + str(step) + '\t Iterations Complete \t' + 'loss: ',
                      loss.item() / 1000.0)
                loss_overall.append(loss_ / 1000.0)
                loss_ = 0


    print('Training Finished! Time Taken: ', time.time() - time_start)
