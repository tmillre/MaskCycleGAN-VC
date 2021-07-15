"""Headers"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import numpy as np
import os.path
import sys
import librosa
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import random, time



np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)

import torch.nn as nn
import torch.nn.functional as F

'''
Below, we define a transformation to normalize parameters within layers
'''

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

'''
Below, we define a 1-D resblock
'''

class ResBlock(nn.Module):
    def __init__(self, in_features):
        super(ResBlock, self).__init__()

        conv_block = [  nn.Conv1d(in_features, 4 * in_features, 3, padding = 1),
                        nn.InstanceNorm1d(4 * in_features),
                        nn.GLU(dim = 1),
                        nn.Conv1d(2 * in_features, in_features, 3, padding = 1),
                        nn.InstanceNorm1d(in_features)  ]
        
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        y = self.conv_block(x)
        return torch.add(y, x)

'''
Below, we define our generative network.
Arguments: numBlocks -- specify how many resBlocks to use in the network.

The dimensions are hard-coded at the moment to work with melgan.

We split our network into three parts, which are split at the 2D-1D and 1D-2D transitions.
'''


class GenNet(nn.Module):
    def __init__(self, numBlocks = 6):
        super(GenNet, self).__init__()
        
        #initial processing layers
        #init layers includes the mask!
        inLayers = [  nn.Conv2d(2, 2 * 128, (5, 15), padding = (2, 7)),
                    nn.GLU(dim = 1),
                    nn.Conv2d(128, 2 * 256, 5, stride = 2, padding = 2),
                    nn.InstanceNorm2d(2 * 256),
                    nn.GLU(dim = 1),
                    nn.Conv2d(256, 2 * 512, 5, stride = 2, padding = 2),
                    nn.InstanceNorm2d(2 * 512),
                    nn.GLU(dim = 1)
        ]
        
        self.inLayers = nn.Sequential(*inLayers)
        
        self.reshape1 = nn.Conv1d(5120 * 2, 256, 1)
        self.reshapeNorm1 = nn.InstanceNorm1d(256)

        
        midLayers = []
        for i in range(numBlocks):
            midLayers.append(ResBlock(256))
            
        self.ResBlocks = nn.Sequential(*midLayers)
            
        
        self.reshape2 = nn.Conv1d(256, 5120, 1)
        self.reshapeNorm2 = nn.InstanceNorm1d(5120)
        
        outLayers = [  nn.Conv2d(256, 2 * 1024, 5, stride = 1, padding = 2),
                    nn.PixelShuffle(upscale_factor = 2),
                    nn.InstanceNorm2d(2 * 256),
                    nn.GLU(dim = 1),
                    nn.Conv2d(256, 2 * 512, 5, stride = 1, padding = 2),
                    nn.PixelShuffle(upscale_factor = 2),
                    nn.InstanceNorm2d(2 * 128),
                    nn.GLU(dim = 1),
                    nn.Conv2d(128, 1, (5, 15), padding = (2, 7))
        ]
        
        self.outLayers = nn.Sequential(*outLayers)


    def forward(self, x):
        
        x = self.inLayers(x)
        x = x.view(x.size(0), 5120 * 2, 1, -1)
        x = x.squeeze(2)
        x = self.reshapeNorm1(self.reshape1(x))
        x = self.ResBlocks(x)
        x = self.reshapeNorm2(self.reshape2(x))
        x = x.unsqueeze(2)
        #Here, we specify height of 20. It will increase by factor of 4 to 80 (number of mels)
        x = x.view(x.size(0), 256, 20, -1)
        x = self.outLayers(x)
        return x

'''
The discriminator network, which is a simple patchGAN.
'''
class DiscNet(nn.Module):
    def __init__(self):
        super(DiscNet, self).__init__()
        
        inLayers = [  nn.Conv2d(1, 2 * 128, 3, padding = 2),
                    nn.GLU(dim = 1),
                    nn.Conv2d(128, 2 * 256, 3, stride = 2, padding = 2),
                    nn.InstanceNorm2d(2 * 256),
                    nn.GLU(dim = 1),
                    nn.Conv2d(256, 2 * 512, 3, stride = 2, padding = 2),
                    nn.InstanceNorm2d(2 * 512),
                    nn.GLU(dim = 1),
                    nn.Conv2d(512, 2 * 1024, 3, stride = 2, padding = 2),
                    nn.InstanceNorm2d(2 * 1024),
                    nn.GLU(dim = 1),
                    nn.Conv2d(1024, 2 * 1024, (1, 5), stride = 1, padding = 1),
                    nn.InstanceNorm2d(2 * 1024),
                    nn.GLU(dim = 1),
                    nn.Conv2d(1024, 1, (1, 5), stride = 1)
        ]
        
        self.inLayers = nn.Sequential(*inLayers)

    def forward(self, x):

        return self.inLayers(x)

class DiscNet1D(nn.Module):
    def __init__(self):
        super(DiscNet1D, self).__init__()
        
        inLayers = [  nn.Conv2d(1, 2 * 128, (5, 15), padding = (2, 7)),
                    nn.GLU(dim = 1),
                    nn.Conv2d(128, 2 * 256, 5, stride = 2, padding = 2),
                    nn.InstanceNorm2d(2 * 256),
                    nn.GLU(dim = 1),
                    nn.Conv2d(256, 2 * 512, 5, stride = 2, padding = 2),
                    nn.InstanceNorm2d(2 * 512),
                    nn.GLU(dim = 1)
        ]

        outLayers = [nn.Conv1d(256, 2 * 512, 5, stride = 2, padding = 2),
                        nn.InstanceNorm1d(2 * 512),
                        nn.GLU(dim = 1),
                     nn.Conv1d(512, 2 * 1024, 5, stride = 2, padding = 2),
                        nn.InstanceNorm1d(2 * 1024),
                        nn.GLU(dim = 1),
                     nn.Conv1d(1024, 1, 5, stride = 1)
                     #   nn.InstanceNorm1d(2 * 2048),
                     #   nn.GLU(dim = 1),
                     #nn.Conv1d(2048, 1, 5, stride = 1),

            ]
        
        self.inLayers = nn.Sequential(*inLayers)
        self.outLayers = nn.Sequential(*outLayers)

        self.reshape1 = nn.Conv1d(5120, 256, 1)
        self.reshapeNorm1 = nn.InstanceNorm1d(256)

    def forward(self, x):

        x = self.inLayers(x)
        x = x.view(x.size(0), 5120, 1, -1)
        x = x.squeeze(2)
        x = self.reshapeNorm1(self.reshape1(x))
        return self.outLayers(x)

        

class MaskCycleGANVC(nn.Module):
    def __init__(self, IS_GPU = False, n_blocks = 6, use1D = False):
        super(MaskCycleGANVC, self).__init__()

        self.use1D = use1D

        #below, we initialize each network with gaussian normalized parameters.
        self.agen = GenNet(numBlocks = n_blocks)
        self.bgen = GenNet(numBlocks = n_blocks)

        self.agen.apply(weights_init_normal)
        self.bgen.apply(weights_init_normal)

        self.aNet = DiscNet()
        self.bNet = DiscNet()

        self.aNet.apply(weights_init_normal)
        self.bNet.apply(weights_init_normal)

        if self.use1D:
          self.a1DNet = DiscNet1D()
          self.b1DNet = DiscNet1D()

          self.a1DNet.apply(weights_init_normal)
          self.b1DNet.apply(weights_init_normal)

        self.aNet.apply(weights_init_normal)
        self.bNet.apply(weights_init_normal)

        self.a2Net = DiscNet()
        self.b2Net = DiscNet()

        self.a2Net.apply(weights_init_normal)
        self.b2Net.apply(weights_init_normal)

        self.IS_GPU = IS_GPU

        if self.IS_GPU:
            self.agen = self.agen.cuda()
            self.bgen = self.bgen.cuda()
            self.aNet = self.aNet.cuda()
            self.bNet = self.bNet.cuda()
            if self.use1D:
              self.a1DNet = self.a1DNet.cuda()
              self.b1DNet = self.b1DNet.cuda()
            self.a2Net = self.a2Net.cuda()
            self.b2Net = self.b2Net.cuda()

        #lists to hold losses over training
        self.train_loss_over_epochs = []
        self.genLossOverEpochs = []
        self.aDiscLossOverEpochs = []
        self.bDiscLossOverEpochs = []
        if self.use1D:
          self.a1DDiscLossOverEpochs = []
          self.b1DDiscLossOverEpochs = []
        self.a2DiscLossOverEpochs = []
        self.b2DiscLossOverEpochs = []

        #lists to hold old generated samples
        self.oldFakeB = []
        self.oldFakeA = []
    '''
    Save the network parameters
    directory -- string giving the directory to save the files in
    datasetName -- string to define part of the filename within that directory
    Save file name will be [network name] + datasetName + .pt
    '''
    def save(self, directory, datasetName):
        torch.save(self.agen.state_dict(), os.path.join(directory,'agen' + datasetName + '.pt'))
        torch.save(self.bgen.state_dict(), os.path.join(directory,'bgen' + datasetName + '.pt'))
        torch.save(self.aNet.state_dict(), os.path.join(directory,'anet' + datasetName + '.pt'))
        torch.save(self.bNet.state_dict(), os.path.join(directory,'bnet' + datasetName + '.pt'))
        if self.use1D:
          torch.save(self.a1DNet.state_dict(), os.path.join(directory,'a1Dnet' + datasetName + '.pt'))
          torch.save(self.b1DNet.state_dict(), os.path.join(directory,'b1Dnet' + datasetName + '.pt'))
        torch.save(self.a2Net.state_dict(), os.path.join(directory,'a2net' + datasetName + '.pt'))
        torch.save(self.b2Net.state_dict(), os.path.join(directory,'b2net' + datasetName + '.pt'))

    '''
    Load the network parameters
    directory -- string giving the directory to load the files from
    datasetName -- string to define part of the filename within that directory
    Save file name assumed to be [network name] + datasetName + .pt
    '''
    def load(self, directory, datasetName):
        if not self.IS_GPU:
            self.aNet.load_state_dict(torch.load(os.path.join(directory,'anet'+datasetName+'.pt')))
            self.bNet.load_state_dict(torch.load(os.path.join(directory,'bnet'+datasetName+'.pt')))
            self.bgen.load_state_dict(torch.load(os.path.join(directory,'bgen'+datasetName+'.pt')))
            self.agen.load_state_dict(torch.load(os.path.join(directory,'agen'+datasetName+'.pt')))
            if self.use1D:
              self.a1DNet.load_state_dict(torch.load(os.path.join(directory,'a1Dnet'+datasetName+'.pt')))
              self.b1DNet.load_state_dict(torch.load(os.path.join(directory,'b1Dnet'+datasetName+'.pt')))
            self.a2Net.load_state_dict(torch.load(os.path.join(directory,'a2net'+datasetName+'.pt')))
            self.b2Net.load_state_dict(torch.load(os.path.join(directory,'b2net'+datasetName+'.pt')))
        else:
            self.aNet.load_state_dict(torch.load(os.path.join(directory,'anet'+datasetName+'.pt'), map_location=torch.device('cpu')))
            self.bNet.load_state_dict(torch.load(os.path.join(directory,'bnet'+datasetName+'.pt'), map_location=torch.device('cpu')))
            self.bgen.load_state_dict(torch.load(os.path.join(directory,'agen'+datasetName+'.pt'), map_location=torch.device('cpu')))
            self.agen.load_state_dict(torch.load(os.path.join(directory,'bgen'+datasetName+'.pt'), map_location=torch.device('cpu')))
            if self.use1D:
              self.a1DNet.load_state_dict(torch.load(os.path.join(directory,'a1Dnet'+datasetName+'.pt'), map_location=torch.device('cpu')))
              self.b1DNet.load_state_dict(torch.load(os.path.join(directory,'b1Dnet'+datasetName+'.pt'), map_location=torch.device('cpu')))
            self.a2Net.load_state_dict(torch.load(os.path.join(directory,'a2net'+datasetName+'.pt'), map_location=torch.device('cpu')))
            self.b2Net.load_state_dict(torch.load(os.path.join(directory,'b2net'+datasetName+'.pt'), map_location=torch.device('cpu')))

    '''
    Trains the network
    aLoader -- dataloader for the first dataset
    bLoader -- dataloader for the second dataset
    lam -- parameter balancing the loss between the cycle consistency loss and GAN loss
    learnRate -- learning rate for the network
    useIDLoss -- (default: False) boolean for whether the network computes the identity loss
                    (i.e. using bgen on data in the b dataset should return the original data)
    NUM_EPOCHS -- (default: 20) how many epochs to run the training for
    gan_criterion -- (default: Mean Squared Error (MSE)) Accepts BCE and MSE for arguments. Defines how the GAN loss is computed.
    beta1/beta2 -- AdaM-specific parameters
    maxMask -- longest possible mask to use for filling-in-frames task
    '''
    def train(self, aLoader, bLoader, lam = 10.0, learnRate = 0.002, useIDLoss = False, NUM_EPOCHS = 20,
              gan_criterion = 'MSE', beta1 = 0.5, beta2 = 0.999, maxMask = 20, bufferLen = 50):
        #define optimizers for backward pass
        params = list(self.agen.parameters()) + list(self.bgen.parameters())
        optimizerGen = torch.optim.Adam(params, lr=learnRate, betas=(beta1, beta2))
        optimizerADisc = torch.optim.Adam(self.aNet.parameters(), lr = learnRate / 2, betas=(beta1, beta2))
        optimizerBDisc = torch.optim.Adam(self.bNet.parameters(), lr = learnRate / 2, betas=(beta1, beta2))
        if self.use1D:
          optimizerA1DDisc = torch.optim.Adam(self.a1DNet.parameters(), lr = learnRate / 2, betas=(beta1, beta2))
          optimizerB1DDisc = torch.optim.Adam(self.b1DNet.parameters(), lr = learnRate / 2, betas=(beta1, beta2))
        optimizerA2Disc = torch.optim.Adam(self.a2Net.parameters(), lr = learnRate / 2, betas=(beta1, beta2))
        optimizerB2Disc = torch.optim.Adam(self.b2Net.parameters(), lr = learnRate / 2, betas=(beta1, beta2))

        mask = torch.ones(1,1,80,64)

        realTarget = torch.zeros(1,1,14,4)
        fakeTarget = torch.ones(1,1,14,4)

        realTarget1D = torch.zeros(1,1,4)
        fakeTarget1D = torch.ones(1,1,4)

        if self.IS_GPU:
          realTarget = realTarget.cuda()
          fakeTarget = fakeTarget.cuda()

          realTarget1D = realTarget1D.cuda()
          fakeTarget1D = fakeTarget1D.cuda()

        k = 0
        j = 0

        L1 = nn.L1Loss()
        if gan_criterion == 'MSE':
            MSELoss = nn.MSELoss()
        elif gan_criterion == 'BCE':
            MSELoss = nn.BCEWithLogitsLoss()

        for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
            t = time.time()
            a_iterator = iter(aLoader)
            runningGenLoss = 0.0
            runningADiscLoss = 0.0
            runningBDiscLoss = 0.0
            runningA1DDiscLoss = 0.0
            runningB1DDiscLoss = 0.0
            runningA2DiscLoss = 0.0
            runningB2DiscLoss = 0.0
            
            for i, data in enumerate(bLoader, 0):

                #determine our masks (current A and B have same mask)
                maskLen = np.random.randint(maxMask)
                
                maskStartA = np.random.randint(64 - maskLen)
                maskStartB = np.random.randint(64 - maskLen)
                
                stackB = torch.cat((data[0].clone(), mask.clone()), axis = 1)
                stackB[:,:,:,maskStartB: (maskStartB + maskLen)] = 0.

                if data[0].shape[0] < 1:
                  continue

                if self.IS_GPU:
                    bInputs = data[0].cuda()
                    stackB = stackB.cuda()
                else:
                    bInputs = data[0]

                # wrap them in Variable
                bInputs = Variable(bInputs.float())
                stackB = Variable(stackB.float())

                # zero the parameter gradients
                optimizerGen.zero_grad()
                optimizerADisc.zero_grad()
                optimizerBDisc.zero_grad()
                if self.use1D:
                  optimizerA1DDisc.zero_grad()
                  optimizerB1DDisc.zero_grad()
                optimizerA2Disc.zero_grad()
                optimizerB2Disc.zero_grad()

                # forward + backward + optimize
                if useIDLoss:
                  bToB = self.bgen(stackB)
                bToA = self.agen(stackB)

                if len(self.oldFakeA) < bufferLen:
                  self.oldFakeA.append(bToA.detach().cpu().clone())
                else:
                  self.oldFakeA[j] = []
                  self.oldFakeA[j] = bToA.detach().cpu().clone()
                  j = (j + 1) % bufferLen

                if self.IS_GPU:
                    recB = self.bgen(torch.cat((bToA.clone(), mask.clone().cuda()), axis = 1))
                else:
                    recB = self.bgen(torch.cat((bToA.clone(), mask.clone()), axis = 1))

                try:
                  aData = next(a_iterator)
                except StopIteration:
                  continue

                if aData[0].shape[0] < 1:
                  continue
                
                stackA = torch.cat((aData[0].clone(), mask.clone()), axis = 1)
                stackA[:,:,:,maskStartA: (maskStartA + maskLen)] = 0.

                if self.IS_GPU:
                    aInputs = aData[0].cuda()
                    stackA = stackA.cuda()
                else:
                    aInputs = aData[0]

                # wrap them in Variable
                aInputs = Variable(aInputs.float())
                stackA = Variable(stackA.float())

                # forward + backward + optimize
                if useIDLoss:
                  aToA = self.agen(stackA)
                aToB = self.bgen(stackA)

                if len(self.oldFakeB) < bufferLen:
                  self.oldFakeB.append(aToB.detach().cpu().clone())
                else:
                  self.oldFakeB[k] = []
                  self.oldFakeB[k] = aToB.detach().cpu().clone()
                  k = (k + 1) % bufferLen

                if self.IS_GPU:
                    recA = self.agen(torch.cat((aToB.clone(), mask.clone().cuda()), axis = 1))
                else:
                    recA = self.agen(torch.cat((aToB.clone(), mask.clone()), axis = 1))
                
                ''' Generator Loss '''
                genLoss = (lam * L1(recA, aInputs)
                  + lam * L1(recB, bInputs)
                  + MSELoss(self.aNet(bToA), realTarget)
                  + MSELoss(self.bNet(aToB), realTarget)
                  + MSELoss(self.a2Net(recA), realTarget)
                  + MSELoss(self.b2Net(recB), realTarget))
                if self.use1D:
                  genLoss += (MSELoss(self.a1DNet(bToA), realTarget1D)
                            + MSELoss(self.b1DNet(aToB), realTarget1D))
                if useIDLoss:
                  genLoss += (0.5 * (lam * L1(bToB, bInputs)
                           + lam * L1(aToA, aInputs)))
                  
                genLoss.backward(retain_graph = True)
                optimizerGen.step()
                runningGenLoss += genLoss.item()

                ''' Discriminator Loss '''
                if self.IS_GPU:
                    fakeASample = Variable(random.choice(self.oldFakeA).clone().cuda().float())
                    fakeBSample = Variable(random.choice(self.oldFakeB).clone().cuda().float())
                else:
                    fakeASample = Variable(random.choice(self.oldFakeA).clone().float())
                    fakeBSample = Variable(random.choice(self.oldFakeB).clone().float())

                bDiscLoss = (MSELoss(self.bNet(bInputs), realTarget) + MSELoss(self.bNet(fakeBSample), fakeTarget)) / 2.0
                bDiscLoss.backward()
                optimizerBDisc.step()
                runningBDiscLoss += bDiscLoss.item()

                aDiscLoss = (MSELoss(self.aNet(aInputs), realTarget) + MSELoss(self.aNet(fakeASample), fakeTarget)) / 2.0
                aDiscLoss.backward()
                optimizerADisc.step()
                runningADiscLoss += aDiscLoss.item()

                #1 step 1 D loss
                if self.use1D:
                  b1DDiscLoss = (MSELoss(self.b1DNet(bInputs), realTarget1D) + MSELoss(self.b1DNet(fakeBSample), fakeTarget1D)) / 2.0
                  b1DDiscLoss.backward()
                  optimizerB1DDisc.step()
                  runningB1DDiscLoss += b1DDiscLoss.item()

                  a1DDiscLoss = (MSELoss(self.a1DNet(aInputs), realTarget1D) + MSELoss(self.a1DNet(fakeASample), fakeTarget1D)) / 2.0
                  a1DDiscLoss.backward()
                  optimizerA1DDisc.step()
                  runningA1DDiscLoss += a1DDiscLoss.item()
                
                b2DiscLoss = (MSELoss(self.b2Net(bInputs), realTarget) + MSELoss(self.b2Net(recB.detach()), fakeTarget)) / 2.0
                b2DiscLoss.backward()
                optimizerB2Disc.step()
                runningB2DiscLoss += b2DiscLoss.item()

                a2DiscLoss = (MSELoss(self.a2Net(aInputs), realTarget) + MSELoss(self.a2Net(recA.detach()), fakeTarget)) / 2.0
                a2DiscLoss.backward()
                optimizerA2Disc.step()
                runningA2DiscLoss += a2DiscLoss.item()



            
            # Normalizing the loss by the total number of train batches
            self.train_loss_over_epochs.append(runningGenLoss)
            self.genLossOverEpochs.append(runningGenLoss)
            self.bDiscLossOverEpochs.append(runningBDiscLoss)
            self.aDiscLossOverEpochs.append(runningADiscLoss)
            if self.use1D:
              self.b1DDiscLossOverEpochs.append(runningB1DDiscLoss)
              self.a1DDiscLossOverEpochs.append(runningA1DDiscLoss)
            self.b2DiscLossOverEpochs.append(runningB2DiscLoss)
            self.a2DiscLossOverEpochs.append(runningA2DiscLoss)
            if self.use1D:
              print('[%d] loss:  %.3f \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f ' %
                    (epoch + 1, runningGenLoss, runningBDiscLoss, runningADiscLoss, runningB2DiscLoss, runningA2DiscLoss, runningB1DDiscLoss, runningA1DDiscLoss), '\t', 'Epoch took ', (time.time() - t) / 60, ' minutes')
            else:
              print('[%d] loss:  %.3f \t %.3f \t %.3f \t %.3f \t %.3f ' %
                  (epoch + 1, runningGenLoss, runningBDiscLoss, runningADiscLoss, runningB2DiscLoss, runningA2DiscLoss), '\t', 'Epoch took ', (time.time() - t) / 60, ' minutes')
        
        # -----------------------------



        # -------------
