import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

#epochs
#batch size
#which optimizer to run
#do we run a scheduler?

# data split between train and test
def split_data_set(trainset, testset, cuda = torch.cuda.is_available(), batch_size=256):
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    return trainloader, testloader

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Train loop
def train(model, device, train_loader, train_acc, train_loss, optimizer, scheduler, l_rate, criterion):

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
        scheduler.step()

        # record and update learning rate
        l_rate.append(get_lr(optimizer))


        ## Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}% learning_rate={l_rate[-1]:0.8f}')
        train_acc.append(100*correct/processed)

# Test loop
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

# Trigger training
def start_training_process(model, device, trainloader, testloader, optimizer='Adam', learning_rate = 0.01, epochs = 20):

    # Hyperparameters
    train_acc, train_losses, test_acc, test_losses, l_rate = [], [], [], [], []
    if optimizer == 'Adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = learning_rate, epochs = epochs, steps_per_epoch = len(trainloader), pct_start = 0.2) # pct_start = 0.2 (~5/24) since max_lr is required at Epoch 5

    for epoch in range(epochs):
        print("EPOCH:", epoch + 1)
        train(model, device, trainloader, train_acc, train_losses, optimizer, scheduler, l_rate, criterion)
        test(model, device, testloader, test_acc, test_losses, criterion)


    return train_acc, train_losses, test_acc, test_losses, l_rate
