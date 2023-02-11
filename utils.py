from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import cv2
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import random
import matplotlib.pyplot as plt
import numpy as np 

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

batchsize = 128

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#image transforms,
def data_param ():
    cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    data = cifar_trainset.data / 255 # data is numpy array
    mean = data.mean(axis = (0,1,2))
    std = data.std(axis = (0,1,2))
    return mean, std

def train_transform_func(mean, std):
    train_transform = A.Compose(
      [
      #RandomCrop(32, padding=4)
      #CutOut(16x16)
      #A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=1, min_width=1, fill_value=tuple(mean), mask_fill_value = None),
      # Pad 4
      A.PadIfNeeded(min_height=32+4, min_width=32+4),
      A.RandomCrop(height = 32, width = 32, always_apply=False, p=1.0),
      A.Cutout(num_holes=1, max_h_size=16, max_w_size=16,  fill_value=tuple(mean)),
      A.Normalize(mean=mean, std=std),
      ToTensorV2(),
      ]
    )
    return lambda img:train_transform(image=np.array(img))["image"]

def test_transform_func(mean, std):
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)
                                        ])
    return test_transform


#gradcam,
def gradcam_test(dataloader, model):
    running_corrects = 0
    running_loss=0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    sm = nn.Softmax(dim = 1)
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        model.eval()
        output = model(data)
        loss = criterion(output, target)
        output = sm(output)
        _, preds = torch.max(output, 1)
        running_corrects = running_corrects + torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        data = data.cpu().numpy()

        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            if(preds[i]!=target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])

    epoch_acc = running_corrects.double()/(len(dataloader)*batchsize)
    epoch_loss = running_loss/(len(dataloader)*batchsize)
    print(epoch_acc,epoch_loss)
    return true,pred,image,true_wrong,pred_wrong

# To plot the wrong predictions given by model
def gradcam_wrong_plot(n_figures,true,ima,pred,encoder, layer, model):
    print('Classes in order Actual and Predicted')
    #n_row = int(n_figures/3)

    for r in range(n_figures):
      plt.figure(figsize = (5, 5))
      a = random.randint(0,len(true)-1)

      image,correct,wrong = ima[a],true[a],pred[a]
      img_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
      image = torch.from_numpy(image)
      correct = int(correct)
      c = encoder[correct]
      wrong = int(wrong)
      w = encoder[wrong]
      f = 'Actual:'+c + ',' +'Predicted:'+w

      for i in range(image.shape[0]):
        image[i] = (image[i]*std[i])+mean[i]
      image = image.numpy().transpose(1,2,0)
      plt.subplot(1, 3, 1)
      im = plt.imshow(image)
      plt.title(f)

      cam = GradCAM(model=model, target_layers=model.layer4)

      target_category = None

      grayscale_cam = cam(input_tensor=img_tensor)

      grayscale_cam = grayscale_cam[0, :]
      #print (grayscale_cam, len(grayscale_cam))
      #print (img, len(img))

      gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
      gb = gb_model(img_tensor, target_category=None)
      cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
      cam_gb = deprocess_image(cam_mask * gb)
      gb = deprocess_image(gb)
      rgb_img = np.float32(image) / 255

      cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
      cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
      plt.subplot(1, 3, 2)
      plt.imshow(cam_gb)

      plt.subplot(1, 3, 3)
      plt.imshow(cam_image)
      #ax.axis('off')
    plt.show()


#misclassification code
def test_missclassified(dataloader, model, criterion=torch.nn.CrossEntropyLoss()):
    running_corrects = 0
    running_loss=0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    sm = nn.Softmax(dim = 1)
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        model.eval()
        output = model(data)
        loss = criterion(output, target)
        output = sm(output)
        _, preds = torch.max(output, 1)
        running_corrects = running_corrects + torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        data = data.cpu().numpy().astype(dtype=np.float32)

        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            if(preds[i]!=target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])
    return true,pred,image,true_wrong,pred_wrong

# show misclassified images
def wrong_plot(n_figures,true,ima,pred, mean, std):
    #n_row = int(n_figures/3)
    fig = plt.figure(figsize = (10, 5))
    fig.tight_layout()
    for r in range(n_figures):
      a = random.randint(0,len(true)-1)
      img,correct,wrong = ima[a],true[a],pred[a]

      f = 'Actual:'+ (classes[correct[0]]) + ',\n ' +'Predicted:'+(classes[wrong[0]])
      #plt.subplot(2,5,r+1)    # the number of images in the grid is 5*5 (25)

      fig.add_subplot(2, 5, r+1)
      fig.subplots_adjust(hspace=.25)


      for i in range(img.shape[0]):
        img[i] = (img[i]*std[i])+mean[i]

      im = plt.imshow(np.transpose(img, (1,2,0)))

      plt.title(f)
    plt.show()

#tensorboard related stuff
#advanced training policies
