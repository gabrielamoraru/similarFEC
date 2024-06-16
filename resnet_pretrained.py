import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchinfo import summary
from typing import Dict, List
from timeit import default_timer as timer 
from engine import train_step, test_step

# Create a writer with all default settings
writer = SummaryWriter('resnet18pretrained')
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ColorJitter(brightness=(0.1,0.6), contrast=1,saturation=0, hue=0.4),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path1 = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path2 = os.path.join(self.img_dir, self.img_labels.iloc[idx, 5])
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')
        label = self.img_labels.iloc[idx, 10]
       
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        if self.target_transform:
            label = self.target_transform(label)
        image = torch.cat((image1, image2), dim=0)
        return image, torch.tensor(label, dtype=torch.float32)

training_data = CustomImageDataset('train_dataset5_cropped.csv', 'cropped_train_dataset5', transform=transform)
test_data = CustomImageDataset('test_dataset5_cropped.csv', 'cropped_test_dataset5', transform=transform)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights).to(device)
old_conv_weight = model.conv1.weight
new_conv = nn.Conv2d(6, 64, (7, 7), (2, 2))
with torch.no_grad():
    new_conv.weight[:, :3, :] = old_conv_weight
    new_conv.weight[:, 3:, :] = old_conv_weight
model.conv1 = new_conv

# Freeze all base layers
for p in model.parameters():
    p.requires_grad = False

# Modify the fully connected layer and add dropout for regularization
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 1)
)
model.to(device)

# Print a summary using torchinfo
print(summary(model=model, 
              input_size=(64, 6, 224, 224), 
              col_names=["input_size", "output_size", "num_params", "trainable"],
              col_width=20,
              row_settings=["var_names"]))

# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Learning rate scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Function to train and validate the model
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

        writer.add_scalars(main_tag="Accuracy", 
                           tag_scalar_dict={"train_acc": train_acc,
                                            "test_acc": test_acc}, 
                           global_step=epoch)
        
        writer.add_graph(model=model, input_to_model=torch.randn(64, 6, 224, 224).to(device))

        # Update the learning rate
        #scheduler.step()
    
    writer.close()
    
    return results

start_time = timer()
results = train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=50,
                device=device)
end_time = timer()

def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.show()

plot_loss_curves(results)
print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")
