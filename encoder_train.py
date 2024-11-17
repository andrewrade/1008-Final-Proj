import tqdm 
from PIL import Image
import torch
from torchvision import transforms
from dataclasses import dataclass

from dataset import create_wall_dataloader
from configs import ConfigBase
from models import BarlowTwins

@dataclass
class EncoderConfig(ConfigBase):
    epochs: int = 100
    batch_size: int = 64
    repr_dim: int = 128
    base_lr: float = 1E-3
    lambd: float = 5E-3

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_train_data(device, batch_size):
    path = '/scratch/DL24FA/train'

    train_ds = create_wall_dataloader(
        data_path=path,
        device=device,
        probing=False,
        train=True,
        batch_size=batch_size
    )
    return train_ds


def augment_data(imgs):
    # Normalize by Mean / Std in training data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(90, interpolation=Image.BILINEAR),
                                    transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])
    return torch.stack([train_transforms(img) for img in imgs])

def train(model, data, device, epochs, base_lr, optimizer):
    
    model.to(device)
    model.train()

    optimizer = torch.optim.Adamw(model.parameters(), lr=base_lr)
    lr_schedule = torch.optim.CosineAnnealingLR(optimizer, T_max=epochs)
        
    for epoch in tqdm(range(epochs)):
        losses = []
        for batch in data:
            states, _ = batch

            Y_a = states
            Y_b = augment_data(states).to(device)

            loss = model(Y_a, Y_b)
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch} Avg Loss: {sum(losses) / len(losses):.3f}')
        lr_schedule.step()
    
    return model



if __name__ == "__main__":
    
    default_config = EncoderConfig()
    device = get_device()
    data = load_train_data(device, batch_size=256)
    
    enc = BarlowTwins(default_config.batch_size, default_config.repr_dim, default_config.lambd)
    encoder = train(enc, data, device, default_config.epochs, default_config.base_lr)
    encoder.save()