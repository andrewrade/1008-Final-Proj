from tqdm import tqdm 
import argparse

from PIL import Image
import torch
from torchvision import transforms

from dataset import create_wall_dataloader
from models import BarlowTwins


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Barlow Twins Encoder')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--repr_dim', type=int, default=128, help='Dimensionality of the representation')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--proj_lyrs', type=float, default=3, help='Number of Projection Layers for Decoder')
    parser.add_argument('--lambd', type=float, default=5e-3, help='Lambda parameter for loss')

    return parser.parse_args()

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

def train(model, data, device, epochs, base_lr):
    
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
    for epoch in tqdm(range(epochs)):
        losses = []
        for batch in data:
            states = batch.states
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

def main():
    args = parse_args()
    device = get_device()
    data = load_train_data(device, batch_size=args.batch_size)
    enc = BarlowTwins(args.batch_size, args.repr_dim, args.proj_lyrs, args.lambd)
    encoder = train(enc, data, device, args.epochs, args.base_lr)
    torch.save(encoder.state_dict(), '/home/ad3254/encoder.pth')



if __name__ == "__main__":
    main()