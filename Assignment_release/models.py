import torch
import torch.nn as nn
import torchvision
from torchvision import models
from transformers import DistilBertModel
class Printer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
# TODO Task 1c - Implement a SimpleBNConv
class SimpleBNConv(nn.Module):
  def __init__(self,device):
    super().__init__()

    self.seq = nn.Sequential(
            nn.Conv2d(3,8,3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,8,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,16,3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,16,3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64), 
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(17920,7)
            
    )
    self.to(device)

  def forward(self, x):
    return self.seq(x)

# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.
def construct_resnet18():
    # TODO: Download the pre-trained model
    model_ft = models.resnet18(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 7)
  
    return model_ft 
  


# TODO Task 1f - Create your own models
class Mymodel(nn.Module):
  def __init__(self,device):
    super().__init__()

    self.seq = nn.Sequential(
            nn.Conv2d(3,8,3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,50,3),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50,50,3),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            #Printer(),
            nn.Linear(197100, 7)
      )
    self.to(device)

  def forward(self, x):
    return self.seq(x)

# TODO Task 2c - Complete TextMLP
# class TextMLP(nn.Module):
#       # Set up the classifier
#   def __init__(self,vocab_size,sentence_len,hidden_size):
#     super().__init__()
#     self.fc = nn.Sequential(
#             nn.Embedding(vocab_size, hidden_size//2),
#             nn.Flatten(),
#             Printer(),
#             nn.Linear(128, 4)
#             )     
   
#   def forward(self, x):
#        # Embed the input sequence
#         embedded_words = self.embedding(x)
#         # Feed the embedded sequence through the LSTM
#         lstm_out, _ = self.lstm(embedded_words)
#         # Keep the LSTM outputs for only the last time-step
#         lstm_out = lstm_out[:, -1]
#         # Feed the outputs through the classifier to get class predictions
#         return self.fc(lstm_out)

# TODO Task 2c - Create a model which uses distilbert-base-uncased
#                NOTE: You will need to include the relevant import statement.
class DistilBertForClassification(nn.Module):

  def __init__(self):
    super(DistilBertForClassification, self).__init__()
    self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.model.classifier = torch.nn.Linear(768, 4)

  def forward(self, input_ids):
      output_1 = self.model(input_ids=input_ids)
      output = self.model.classifier(input_ids)
      return self.model[input_ids]
  

#   ....