import collections
import csv
from pathlib import Path
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# TODO Task 1b - Implement LesionDataset
#                The __init__ function should have the following prototype
#                     def __init__(self, img_dir, labels_fname):
#                     - img_dir is the directory path with all the image files
#                     - labels_fname is the csv file with image ids and their corresponding labels

toTenser = transforms.Compose([
        transforms.ToTensor()
        ])
totenserAug = transforms.Compose([
    transforms.RandomGrayscale(0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
class LesionDataset():
  def __init__(self, img_dir, labels_fname,augment=False):
    self.img_dir = img_dir
    self.labels_fname = pd.read_csv(labels_fname)
    self.lab = self.labels_fname.iloc[: , 1:]
    self.lab = np.argmax(np.array(self.lab), axis=1)
    self.aug= augment
  def __len__(self):
    return len(self.labels_fname)
    
  def __getitem__(self,index):
    img_id = self.labels_fname.iloc[index, 0]
    img = Image.open(os.path.join(self.img_dir, img_id+'.jpg')).convert("RGB")
    label = self.lab[index]
    if self.aug:
      image = totenserAug(img)
    else:
      image = toTenser(img)
    
    return image, label

# TODO Task 1e - Add augment flag to LesionDataset, so the __init__ function
#                now look like this:
#                   def __init__(self, img_dir, labels_fname, augment=False):



# TODO Task 2b - Implement TextDataset
#               The __init__ function should have the following prototype
#                   def __init__(self, fname, sentence_len)
#                   - fname is the filename of the cvs file that contains each
#                     news headlines text and its corresponding label.
#                   - sentence_len the maximum sentence length you want the
#                     tokenized to return. Any sentence longer than that should
#                     be truncated by the tokenizer. Any shorter sentence should
#                     padded by the tokenizer.
#                We will be using the pretrained 'distilbert-base-uncased' transform,
#                so please use the appropriate tokenizer for it. NOTE: You will need
#                to include the relevant import statement.
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class TextDataset():
  def __init__(self,fname,sentence_len):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    self.fname = pd.read_csv(fname, header=None)
    texts = self.fname[2].str.slice(0, sentence_len).tolist()
    self.sentence_len = sentence_len
    tokens = tokenizer(texts,
      max_length=self.sentence_len,
      return_token_type_ids=True,
      padding=True,
      truncation=True,
      return_attention_mask=True
        )
    self.tokens = tokens["input_ids"]


  def __len__(self):
    return len(self.fname)

  def __getitem__(self,index):
    labels_fname = self.fname.iloc[index, 0]-1
    input_ids=torch.tensor(self.tokens[index])
    labels=torch.tensor(labels_fname)
    return input_ids,labels
   
    

  
