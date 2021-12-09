'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import sys
import os
import argparse
import numpy

from models import *
from utils import progress_bar

from PIL import Image
convert_tensor = transforms.ToTensor()


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')



args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data

'''net = ResNet18()
net = net.to(device)
print(summary(net, (3, 32, 32)) )

sys.exit()
'''
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
'''
trainset = torchvision.datasets.CIFAR10(
    root='./data', train = True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train = False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

'''
cur_dir = os.getcwd()

#Edge_Image_Path = os.path.join(cur_dir,'DexiNed-master/result/BIPED2CLASSIC/CIFAR10/train/avg')
Edge_Image_Path = os.path.join(cur_dir,'DexiNed-master/enhanced_results/CIFAR10/train')
Label_Path = os.path.join(cur_dir,'DexiNed-master/data/CIFAR10/train')

edge_images = os.listdir(Edge_Image_Path)
labels = [f for f in os.listdir(Label_Path) if f.endswith('.pt')]

image_label_list_train = []
for img_name in edge_images:
  img = Image.open(Edge_Image_Path+'/'+img_name)
  input = convert_tensor(img)
  label_name = img_name.split('.')[0]+'label.pt'
  label = torch.load(Label_Path+'/'+label_name)
  image_label_list_train.append((input,label))


#Edge_Image_Path = os.path.join(cur_dir,'DexiNed-master/result/BIPED2CLASSIC/CIFAR10/test/avg')
Edge_Image_Path = os.path.join(cur_dir,'DexiNed-master/enhanced_results/CIFAR10/test')
Label_Path = os.path.join(cur_dir,'DexiNed-master/data/CIFAR10/test')

edge_images = os.listdir(Edge_Image_Path)
labels = [f for f in os.listdir(Label_Path) if f.endswith('.pt')]


image_label_list_test = []
for img_name in edge_images:
  img = Image.open(Edge_Image_Path+'/'+img_name)
  input = convert_tensor(img)
  label_name = img_name.split('.')[0]+'label.pt'
  label = torch.load(Label_Path+'/'+label_name)
  image_label_list_test.append((input,label))

trainloader = torch.utils.data.DataLoader(
    image_label_list_train, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    image_label_list_test, batch_size=100, shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck')

#classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

#classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale',
#                   'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
#		   'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
#		   'bottles', 'bowls', 'cans', 'cups', 'plates',
#		   'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
#		   'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
#		   'bed', 'chair', 'couch', 'table', 'wardrobe',
#		   'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
#		   'bear', 'leopard', 'lion', 'tiger', 'wolf',
#		   'bridge', 'castle', 'house', 'road', 'skyscraper',
#		   'cloud', 'forest', 'mountain', 'plain', 'sea',
#		   'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
#		   'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
#		   'crab', 'lobster', 'snail', 'spider', 'worm',
#		   'baby', 'boy', 'girl', 'man', 'woman',
#		   'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
#		   'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
#		   'maple', 'oak', 'palm', 'pine', 'willow',
#		   'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
#		   'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')
#
# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #print(inputs.shape,targets)
        targets = torch.reshape(targets, (len(targets),))
        #print(inputs.shape,targets)
        #sys.exit()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #print(inputs.shape,targets)
            targets = torch.reshape(targets, (len(targets),))
            #print(inputs.shape,targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
