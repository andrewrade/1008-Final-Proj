from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import ViTBackbone, BarlowTwins
import glob


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model(device):
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    enc_path = r"/home/ad3254/checkpoints/best.pth"
    state_dict = torch.load(enc_path)['model_state_dict']

    # Define the ViT Backbone
    backbone = ViTBackbone(
        image_size=65,
        patch_size=5,
        in_channels=2,
        embed_dim=256,
        num_heads=4,
        mlp_dim=1024,
        num_layers=3,
        dropout=0.1,
    )

    model = BarlowTwins(backbone=backbone, batch_size=64, repr_dim=256)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
