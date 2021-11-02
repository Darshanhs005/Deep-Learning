from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import sklearn.metrics as sk
import wandb
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

def plot_confusion_matrix(all_lbls, all_outputs, class_names):
    # TODO Task 1c - Implement plot function
    # Count confusions using sklearn.metrics.confusion_matrix
    # Normalise by true label as in the labs
    # cm = ...

    
    out_array = np.argmax(np.array(all_outputs),axis=1)
    arr= np.array(all_lbls)
    cm = sk.confusion_matrix(arr,out_array)
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(cm, class_names,class_names)
    ax = sn.heatmap(df_cm, annot=True, cmap='flare')
  # plot the resulting confusion matrix
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    # TODO Task 1c - Set axis labels and show plot
    

def count_classes(preds):
    '''
    Counts the number of predictions per class given preds, a tensor
    shaped [batch, n_classes], where the maximum per preds[i]
    is considered the "predicted class" for batch element i.
    '''
    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]
    return [(pred_classes == c).sum().item() for c in range(n_classes)]
def train_epoch(epoch, model, optimizer, criterion, loader):
    epoch_loss = 0

    model.train()
    # At the end all_outputs should store the output for each sample in the training data.
    all_outputs = []
    # At the end all_lbls should store the ground truth label for each sample in the training data.
    all_lbls = []
    for i, (inputs, lbls) in enumerate(loader):
        inputs, lbls = inputs.to(device), lbls.to(device)

        # Update model weights
        # TODO: Tasb 1b - Perform a forward pass, backward pass and 
        # update the weights of your model with the batch of data.
        # lbls = lbls.type(torch.LongTensor)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        # TODO: Task 2d - Temporarily uncomment these lines
        # print(count_classes(outputs))
        # if i > 9:
        #     assert False

        # Collect metrics
        epoch_loss += loss.item()
        all_outputs.extend(outputs.tolist())
        all_lbls.extend(lbls.tolist())

    # Calculate epoch metrics
    # TODO Task 1b - Use all_outputs and all_lbls with
    # sklearn.metrics.accuracy_score and sklearn.metrics.recall_score
    # to calculate the accuracy and unweighted average recall.
    # Note sklearn.metrics.accuracy_score and sklearn.metrics.recall_score
    # only take numpy arrays and also you need to convert all_outputs
    # to the actual predicted class for each sample.
    
    # acc = ...
    # uar = ...

    all_array = np.array(all_lbls)
    out_array = np.argmax(np.array(all_outputs), axis=1)
    acc = sk.accuracy_score(all_array,out_array)
    uar=sk.recall_score(all_array,out_array,average='macro')
    metrics_dict = {
        'Loss/train': (epoch_loss/len(loader)),
        'Accuracy/train': acc,
        'UAR/train': uar,
    }

    return all_lbls, all_outputs, metrics_dict

def val_epoch(epoch, model, criterion, loader):
    epoch_loss = 0

    model.eval()
    all_outputs = []
    all_lbls = []
    for inputs, lbls in loader:
        inputs, lbls = inputs.to(device), lbls.to(device)

        # TODO Task 1b - Perform a forward pass through your model and 
        # obtain the validation loss (use torch.no_grad())
        with torch.no_grad():
          # lbls = lbls.type(torch.LongTensor)
          outputs = model(inputs)

          loss = criterion(outputs, lbls)
          

          
          # Collect metrics
          epoch_loss += loss.item()
          all_outputs.extend(outputs.tolist())
          all_lbls.extend(lbls.tolist())

    # Calculate epoch metrics
    # TODO Task 1b - Use all_outputs and all_lbls with
    # sklearn.metrics.accuracy_score and sklearn.metrics.recall_score
    # to calculate the accuracy and unweighted average recall.
    # Note sklearn.metrics.accuracy_score and sklearn.metrics.recall_score
    # only take numpy arrays and also you need to convert all_outputs
    # to the actual predicted class for each sample.
 
    # acc = ...
    # uar = ...
    out_array = np.array(all_outputs)
    out_array = np.argmax(out_array, axis=1)
    acc = sk.accuracy_score(np.array(all_lbls),out_array)
    uar=sk.recall_score(np.array(all_lbls),out_array,average='macro')
    metrics_dict = {
        'Loss/val': (epoch_loss/len(loader)),
        'Accuracy/val': acc,
        'UAR/val': uar,
    }

    return all_lbls, all_outputs, metrics_dict


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
    model.to(device)

    # Initialise Weights and Biases project
    if ident_str is None:
      ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    run = wandb.init(project = project_name, name=exp_name)

    try:
        # Train by iterating over epochs
        for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
            _, _, train_metrics_dict = \
                train_epoch(epoch, model, optimizer, criterion, train_loader)
            val_lbls, val_outs, val_metrics_dict = \
                val_epoch(epoch, model, criterion, val_loader)
            wandb.log({**train_metrics_dict, **val_metrics_dict})
    finally:
        run.finish()

    # Create confusion matrix from results of last val epoch
    # TODO Task 1c - Implement the rest of plot_confusion_matrix and uncomment
    # print(val_lbls,"  ",val_outs)
    # print(val_lbls.size,val_outs.size)
    plot_confusion_matrix(val_lbls, val_outs, class_names)

    # Save the model weights to "saved_models/"
    # TODO Task 2b - Save model weights
