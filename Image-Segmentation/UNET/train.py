import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models import Unet
from utils import get_loaders, save_predictions_as_imgs
from metrics import check_accuracy

#TRAINED FOR BINARY CLASS SEGMENTATION, ACCURACY AND SAVING IMPLEMENTED THUS

# Hyperparameters 
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  
IMAGE_WIDTH = 240  
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward through model and initialize loss
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():

    # transforms for images in loader
    train_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),])

    val_transforms = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),])

    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    
    # a binary CELoss is used with logits as we haven't passed the final convolution through a sigmoid
    # for multi-class segmentation, cross entropy loss to be used
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # load data
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    # untrained accuracy
    check_accuracy(val_loader, model, device=DEVICE, MODEL=True)
    scaler = torch.cuda.amp.GradScaler()

    # training can be done using simple pytorch methods as well, used tqdm for learning and testing
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # training validation accuracy
        check_accuracy(val_loader, model, device=DEVICE, MODE=True)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE, MODE=True)

    # accuracy after training
    check_accuracy(val_loader, model, device=DEVICE, MODE=False)
