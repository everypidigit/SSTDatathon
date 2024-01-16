import os
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

from google.colab import drive
drive.mount('/content/drive')

data_dir = '/content/drive/MyDrive/Colab Notebooks/sst256/data'


print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)

classes.remove('.DS_Store')

no_fire_files = os.listdir(data_dir + "/train/0")
print('No. of training examples for no-fire examples:', len(no_fire_files))
print(no_fire_files[:5])

fire_files = os.listdir(data_dir + "/train/1")
print('No. of images that contain fire:', len(fire_files))
print(fire_files[:5])

dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

img, label = dataset[0]

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0)

for x in range(10):
    show_example(*dataset[x])


# In[42]:


random_seed = 42
torch.manual_seed(random_seed);


# In[45]:


val_set_size = 200
train_size = len(dataset) - val_set_size


# In[46]:


# defining our training and validation dataset
training_ds, validation_ds = random_split(dataset, [train_size, val_set_size])
len(training_ds), len(validation_ds)


# In[47]:


from torch.utils.data.dataloader import DataLoader

batch_size=32


# In[48]:


# a couple of functions to check the default device and to transwer a torch.dataloader to NVIDIA GPU that is used in Colab

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
device = get_default_device()
device


# In[49]:


# defining our dataloaders for training and validation datasets
train_dl = DataLoader(training_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(validation_ds, batch_size*2, num_workers=4, pin_memory=True)


# In[50]:


from torchvision.utils import make_grid


# In[51]:


# a function to show a bunch of images from our datasets
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break


# In[52]:


show_batch(train_dl)


# In[55]:


import torch.nn as nn
import torch.nn.functional as F


# In[58]:


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[60]:


# our model itself


class FireModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 128 x 128

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 64 x 64

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 32 x 32
            
            
            nn.Flatten(), 
            nn.Linear(256*32*32, 32),
            nn.ReLU(),
            nn.Linear(32, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)


# In[61]:


model = FireModel()
model


# In[63]:


# sending our dataloaders to the GPU
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device);


# In[65]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[66]:


model = to_device(FireModel(), device)


# In[67]:


evaluate(model, val_dl)


# In[68]:


# defining hyperparams and the number of epochs
num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001


# In[69]:


# training our model for 10 epochs
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


# In[107]:


# saving our validation scores
scores = [d['val_acc'] for d in history]


# In[129]:


total_score = sum(scores)


# In[183]:


# our average score for the validation dataset
avg_score = total_score / len(scores)
avg_score


# Now we are gonna go and test our model on the test images

# In[132]:


# defining test files directory
test_dir = '/content/drive/MyDrive/Colab Notebooks/sst256/data/test/test'


# In[133]:


# resizing images to a 256x256 size so that we can do matrix operations on them
for file in os.listdir(test_dir):
    if not file.startswith('.'):
        f_img = test_dir+"/"+file
        img = Image.open(f_img)
        img = img.resize((256, 256))
        img.save(f_img)


# In[134]:


# defining our test dataset
test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())


# In[192]:


# a function to predict a label for an image with a given model
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


# In[207]:


test_names = []
test_predictions = []


# In[208]:


test_length = len(test_dataset)
test_length


# In[210]:


# making prediction for each of the test images and saving their name+predictedvalue in two lists for names and labels
for x in range(test_length):
  img, label = test_dataset[x]
  test_predictions.append(predict_image(img, model))
  test_names.append(test_dataset.imgs[x])


# In[211]:


results = {}


# In[212]:


# creating a dictionary from our names and predictions lists
for key in test_names:
  for value in test_predictions:
    results[key] = value
    test_predictions.remove(value)
    break


# In[216]:


import pandas as pd


# In[238]:


# converting our dictionary to a DF
final_results = pd.DataFrame.from_dict(results, orient='index')


# In[247]:


# saving our DF to a .csv fi,e
final_results.to_csv('final_results.csv')  

