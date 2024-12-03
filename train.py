import torch
from vision_mamba import Vim

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser(description='Vim')



parser.add_argument('--dim', default=32, type=int, help='Vim Dim')
parser.add_argument('--depth', default=4, type=int, help='Vim Dim')
args = parser.parse_args() 
print(args.dim, args.depth, flush=True)
testtransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.0001, 0.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.2),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1024

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=testtransform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=12)



# Model
model = Vim(
    dim=args.dim,  # Dimension of the transformer model
    dt_rank=args.dim,  # Rank of the dynamic routing matrix
    dim_inner=args.dim,  # Inner dimension of the transformer model
    d_state=args.dim,  # Dimension of the state vector

    num_classes=100,  # Number of output classes
    image_size=32,  # Size of the input image
    patch_size=8,  # Size of each image patch
    channels=3,  # Number of input channels
    dropout=0.1,  # Dropout rate
    depth=args.depth,  # Depth of the transformer model
)
model = model.cuda()
def test(testloader, net):
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct/ total} %')
    return correct/ total

criterion = nn.CrossEntropyLoss()

optimizer =torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=300)
accs = []

for epoch in range(300):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for  data in tqdm(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    acc = test(testloader, model)
    accs.append(acc)

print(max(accs))
