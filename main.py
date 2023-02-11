import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

#epochs
#batch size
#which optimizer to run
#do we run a scheduler?

#data split between test and train
def split_data(trainset, testset, batch_size=256):
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    return trainloader, testloader

#training and test loops
def train(model, device, train_loader, train_acc, train_loss, optimizer, criterion):

    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):

        ## Get data samples
        data, target = data.to(device), target.to(device)

        ## Init
        optimizer.zero_grad()

        ## Predict
        y_pred = model(data)

        ## Calculate loss
        loss = criterion(y_pred, target)

        #train_loss.append(loss.data.cpu().numpy().item())
        train_loss.append(loss.item())

        ## Backpropagation
        loss.backward()

        optimizer.step()

        ## Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader, test_acc, test_losses, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

def start_training_process(model, device, trainloader, testloader, optimizer='Adam', learning_rate = 0.01, epochs = 20):

    # Hyperparameters


    train_acc, train_losses, test_acc, test_losses = [], [], [], []
    if optimizer == 'Adam':
      torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()


    for epoch in range(epochs):
        print("EPOCH:", epoch + 1)
        train(model, device, trainloader, train_acc, train_losses, optimizer,  criterion)
        test(model, device, testloader, test_acc, test_losses, criterion)


    return train_acc, train_losses, test_acc, test_losses
