import argparse
from tqdm import tqdm 

import torch
from torchvision.transforms import v2

from dataset import create_wall_dataloader
from models import BarlowTwins
from normalizer import StateNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Barlow Twins Encoder')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--warmup_epochs', type=float, default=3, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--repr_dim', type=int, default=256, help='Dimensionality of the representation')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--proj_lyrs', type=float, default=2, help='Number of Projection Layers for Decoder')
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
    
    transforms = v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.RandomAffine(degrees=(0, 30), scale=(0.9, 1)),
        v2.GaussianBlur(kernel_size=3,sigma=(0.1, 1)),
    ])
    
    return torch.stack([transforms(img) for img in imgs])

def train(model, data, device, epochs, warmup_epochs, base_lr):
    """
    Encoder Pre-Training Loop
    """
    
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs)
    
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        [warmup_epochs]
    )
    
    best_loss = float('inf')
    losses = []
    normalizer = StateNormalizer()

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(data, desc=f'Epoch {epoch}'):
            
            states = batch.states.to(device)
            states = normalizer.normalize_state(states)
            
            # Un-augmented Frames
            Y_a = states
            batch_size, num_frames, channels, height, width = Y_a.shape
            Y_a = Y_a.view(batch_size * num_frames, channels, height, width)
            
            # Shuffle Indices to prevent model from exploiting ordering
            shuffled_idxs = torch.randperm(Y_a.size(0), device=device)
            Y_a = Y_a[shuffled_idxs]

            # Augmented Frame for Loss
            Y_b = augment_data(Y_a).to(device)

            # Forward pass
            loss = model(Y_a, Y_b)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f'Epoch {epoch} Avg Loss: {avg_loss:.3f}')
        lr_scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'normalizer_state': normalizer.state_dict() if hasattr(normalizer, 'state_dict') else None
            }, '/home/ad3254/checkpoints/best.pth')

    return model

def main():
    args = parse_args()
    device = get_device()
    data = load_train_data(device, batch_size=args.batch_size * 17)
    enc = BarlowTwins(args.batch_size, args.repr_dim, args.proj_lyrs, args.lambd)
    encoder = train(enc, data, device, args.epochs, args.warmup_epochs, args.base_lr)
    torch.save(encoder.state_dict(), '/home/ad3254/encoder.pth')
    


if __name__ == "__main__":
    main()