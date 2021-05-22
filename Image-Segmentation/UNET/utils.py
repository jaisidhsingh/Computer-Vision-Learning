import torch
from data import CarvanaDataset
from torch.utils.data import DataLoader
import torchvision

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True,):
    train_ds = CarvanaDataset(imageDir=train_dir, maskDir=train_maskdir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_ds = CarvanaDataset(imageDir=val_dir, maskDir=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", MODE=True):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train(mode=MODE)
