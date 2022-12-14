import torch
import torchvision
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
import argparse
# Python Imaging Library
import PIL
import numpy as np
import sys as sys
import pdb


#  Global Parameters
# Automatically detect if there is a GPU or just use CPU.
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ========================================================================================================================
# Functions and Network Template
# ========================================================================================================================
def load_data(bSize = 32):
    # bundle common args to the Dataloader module as a kewword list.
    # pin_memory reserves memory to act as a buffer for cuda memcopy 
    # operations
    comArgs = {'shuffle': True,'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Data Loading -----------------------
    # ******************
    # At this point the data come in Python tuples, a 28x28 image and a label.
    # while the label is a tensor, the image is not; it needs to be converted.  
    # So we need to transform PIL image to tensor and then normalize it.
    # Normalization is quite a good practise to avoid numerical and convergence
    # problems. For that we need the dataset's mean and std which fortunately
    # can be computed!
    # ******************
    mean = 0.1307
    std  = 0.3081
    # Bundle our transforms sequentially, one after another. This is important.
    # Convert images to tensors + normalize
    transform = tTrans.Compose([tTrans.ToTensor(), tTrans.Normalize( (mean,), (std,) )])
    # Load data set
    mnistTrainset = tdata.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    mnistTestset = tdata.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Once we have a dataset, torch.utils has a very nice lirary for iterating on that
    # dataset, with shuffle AND batch logic. Very usefull in larger datasets.
    trainLoader = torch.utils.data.DataLoader(mnistTrainset, batch_size = bSize, **comArgs )
    testLoader = torch.utils.data.DataLoader(mnistTestset, batch_size = bSize, **comArgs)
    # End of DataLoading -------------------


    # Sanity Prints---
    # print(len(mnistTrainset))
    # print(type(mnist_trainset[0]))

    return trainLoader, testLoader

# --------------------------------------------------------------------------------------------------------

# Model Definition
class Net(nn.Module):

    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    # Mod init + boiler plate code
    # Skeleton of this network; the blocks to be used.
    # Similar to Fischer prize building blocks!
    def __init__(self):
        super(Net, self).__init__()
        # Declare the layers along with their dimension here!
        # NOTE: Tryuing to run the code with no layer declared and no architecture defined will 
        # Lead to an error "ValueError: optimizer got an empty parameter list"

        # ---|

        self.conv_1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv_2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fully_connec_1 = torch.nn.Linear(32*7*7, 120 )
        self.fully_connec_2 = torch.nn.Linear(120, 60)
        self.fully_connec_out = torch.nn.Linear(60, 10)

    # ------------------

    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        # Define the network architecture here.
        # Each layer would be given as input to the next. 
        # Output should be of size [batchSize, classes]
        # NOTE: After each layer, especially after convolutional layers the shape
        # of the size tensor changes. print(x.shape) might be your Samwise Gamtzee
        # in those difficult moments!

        #print(x.shape)
        #pdb.set_trace()
        x = self.conv_1(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv_2(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.pool(x)
        #x = self.pool(F.relu(self.conv_2(x)))
        #pdb.set_trace()
        x = x.view(-1, 32*7*7)
        x = F.relu(self.fully_connec_1(x))
        x = F.relu(self.fully_connec_2(x))
        x = self.fully_connec_out(x)

        return x

    # ------------------

    # Call this function to facilitate the traiing process
    # While there are many ways to go on about calling the
    # traing and testing code, define a function within
    # a "Net" class seems quite intuitive. Many examples
    # do not have a class function; rather they set up
    # the training logic as a block script layed out.
    # Perhaps the object oriented approach leads to less 
    # anguish in large projects...
    def train(self, args, device, indata, optim, verbose = True):

        loss = torch.nn.CrossEntropyLoss()
        for idx, (img, label) in enumerate(indata):
            data, label = img.to(device), label.to(device)
            # forward pass calculate output of model
            output      = self.forward(data)
            # compute loss
            #loss        = F.nll_loss(output, label)
            difference = loss(output, label)

            # Backpropagation part
            # 1. Zero out Grads
            optim.zero_grad()
            # 2. Perform the backpropagation based on loss
            difference.backward()            
            # 3. Update weights 
            optim.step()

           # Training Progress report for sanity purposes! 
            if verbose:
                if idx % 20 == 0: 
                    print("Epoch: {}->Batch: {} / {}. Loss = {}".format(args, idx, len(indata), difference.item() ))
        # Log the current train loss
        self.trainLoss = loss   
    # -----------------------

    # Testing and error reports are done here
    def test(self, device, testLoader):
        print("In Testing Function!")        
        loss = 0 
        true = 0
        acc  = 0
        # Inform Pytorch that keeping track of gradients is not required in
        # testing phase.
        with torch.no_grad():
            for data, label in testLoader:
                data, label = data.to(device), label.to(device)
                # output = self.forward(data)
                output = self.forward(data)
                # Sum all loss terms and tern then into a numpy number for late use.
                loss  += F.nll_loss(output, label, reduction = 'sum').item()
                # Find the max along a row but maitain the original dimenions.
                # in this case  a 10 -dimensional array.
                pred   = output.max(dim = 1, keepdim = True)
                # Select the indexes of the prediction maxes.
                # Reshape the output vector in the same form of the label one, so they 
                # can be compared directly; from batchsize x 10 to batchsize. Compare
                # predictions with label;  1 indicates equality. Sum the correct ones
                # and turn them to numpy number. In this case the idx of the maximum 
                # prediciton coincides with the label as we are predicting numbers 0-9.
                # So the indx of the max output of the network is essentially the predicted
                # label (number).
                true  += label.eq(pred[1].view_as(label)).sum().item()
        acc = true/len(testLoader.dataset)
        self.accuracy = acc
        self.testLoss = loss 
        # Print accuracy report!
        print("Accuracy: {} ({} / {})".format(acc, true,
                                              len(testLoader.dataset)))

    def report(self):

        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

# --------------------------------------------------------------------------------------------------------

def parse_args():
    ''' Description: This function will create an argument parser. This will accept inputs from the console.
                     But if no inputs are given, the default values listed will be used!
        

    '''
    parser = argparse.ArgumentParser(prog='Fashion MNIST Network building!')
    # Tell parser to accept the following arguments, along with default vals.
    parser.add_argument('--lr',    type = float,metavar = 'lr',   default='0.001',help="Learning rate for the oprimizer.")
    parser.add_argument('--m',     type = float,metavar = 'float',default= 0,     help="Momentum for the optimizer, if any.")
    parser.add_argument('--bSize', type = int,  metavar = 'bSize',default=32,     help="Batch size of data loader, in terms of samples. a size of 32 means 32 images for an optimization step.")
    parser.add_argument('--epochs',type = int,  metavar = 'e',    default=12   ,  help="Number of training epochs. One epoch is to perform an optimization step over every sample, once.")
    # Parse the input from the console. To access a specific arg-> dim = args.dim
    args = parser.parse_args()
    lr, m, bSize, epochs = args.lr, args.m, args.bSize, args.epochs
    # Sanitize input
    m = m if (m>0 and m <1) else 0 
    lr = lr if lr < 1 else 0.1
    # It is standard in larger project to return a dictionary instead of a myriad of args like:
    # return {'lr':lr,'m':m,'bSize':bbSize,'epochs':epochs}
    return lr, m , bSize, epochs

# ================================================================================================================================
# Execution
# ================================================================================================================================
def main():
    # Get keyboard arguments, if any! (Try the dictionary approach in the code aboe for some practice!)
    lr, m , bSize, epochs = parse_args()
    # Load data, initialize model and optimizer!
    trainLoader, testLoader = load_data(bSize=bSize)
    model = Net().to(device) # send model to appropriate computing device (CPU or CUDA)
    optim = optm.SGD(model.parameters(), lr=lr, momentum=m) # Instantiate optimizer with the model's parameters.

    print("######### Initiating Fashion MNIST network training #########\n")
    print("Parameters: lr:{}, momentum:{}, batch Size:{}, epochs:{}".format(lr,m,bSize,epochs))
    for e in range(epochs):
        print("Epoch: {} start ------------\n".format(e))
        # print("Dev {}".format(device))
        args = e
        model.train(args, device, trainLoader, optim)
        model.test(device, testLoader)

    # Final report
    model.report()

# Define behavior if this module is the main executable. Standard code.
if __name__ == '__main__':
    main()

