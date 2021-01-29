from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

import wandb

def train_resnet50(model, dataloaders, criterion, optimizer, device,num_epochs=25,save_name='RESNET50'):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if dataloaders[phase] is None: #No val, skip validation
                continue
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)                
                wandb.log({f"{save_name} Validation Epoch Loss": epoch_loss, f"{save_name} epoch": epoch})
                wandb.log({f"{save_name} Validation Epoch ACC": epoch_acc, f"{save_name} epoch": epoch})
            else: 
                wandb.log({f"{save_name} Train Loss": epoch_loss, f"{save_name} epoch": epoch})
                wandb.log({f"{save_name} Train ACC": epoch_acc, f"{save_name} epoch": epoch})
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, req_grad):
    if req_grad == False:
        for i, param in enumerate(model.parameters()):
            if(param.shape[0] != 1000):    #Only train last layer?    
                param.requires_grad = False


def get_update_param(encoder):
    # Only update the new last layer parameter
    params_to_update = []
    for name, param in encoder.named_parameters():
        if param.requires_grad == True:  # Only new layer is true
            params_to_update.append(param)
            print("\t", name)
    return params_to_update
def train_last_layer_resnet50(encoder, dataloader_list, log_name,num_classes, args,device,num_epochs=25):
    # Concat dataloaders
    concat_dataset = torch.utils.data.ConcatDataset([x.dataset for x in dataloader_list])
    dataloader = torch.utils.data.DataLoader(
        dataset=concat_dataset,
        batch_size=args.encoder_bs,
        shuffle=True,
        num_workers=4
    )
    #Val not mentioned in paper so use ignore
    dataloader_dict = {'train':dataloader,'val':None}
    

    # Set all param grad update to false
    set_parameter_requires_grad(encoder, req_grad=False)
    num_ftrs = encoder.fc.in_features
    encoder.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    encoder.to(device)
    
    
    # Only update the new last layer parameter
    params_to_update = get_update_param(encoder)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    encoder, val_acc_history = train_resnet50(
        encoder, dataloader_dict, criterion, optimizer_ft, device,save_name=log_name, num_epochs=num_epochs)
    return encoder, val_acc_history