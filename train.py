import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model import ResNet18


def prepare_cifar():
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

    trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

    return trainloader, testloader

# Training
def train(net, optimizer, criterion, epoch, trainloader, log, args):
    print('Epoch: %d' % epoch)
    print("Training")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    tot_batch = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        tot_batch = batch_idx
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

        print("epoch: %d, step: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)" % (epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    log.write("(Train) epoch: %d, loss: %.3f | Acc: %.3f%% (%d/%d)" % (epoch, train_loss/(tot_batch+1), 100.*correct/total, correct, total))


def test(net, epoch, criterion, testloader, log, best_acc, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("Testing")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print("%d" % batch_idx)

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
        torch.save(state, './checkpoint/ckpt_{}.pth'.format(args.expid))
        best_acc = acc
    elif epoch % 50 == 49:
        print("Saving..")
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_{}_{}.pth'.format(args.expid, epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('-expid', default=0, type=int)
    
    args = parser.parse_args()
    log = open("result_log_{}".format(args.expid), "w")

    trainloader, testloader = prepare_cifar()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    net = ResNet18() # kaiming init
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_{}.pth'.format(args.expid))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(start_epoch):
        scheduler.step()
    for epoch in range(start_epoch, 290):
        train(net, epoch, criterion, epoch, trainloader, log, args)
        test(net, epoch, criterion, epoch, testloader, log, args)
        scheduler.step()