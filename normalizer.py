import torch

class Normalizer:
    def __init__(self):
        self.location_mean = torch.tensor([31.5863, 32.0618])
        self.location_std = torch.tensor([16.1025, 16.1353])

    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return (location - self.location_mean.to(location.device)) / (
            self.location_std.to(location.device) + 1e-6
        )

    def unnormalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return location * self.location_std.to(location.device) + self.location_mean.to(
            location.device
        )

    def unnormalize_mse(self, mse):
        return mse * (self.location_std.to(mse.device) ** 2)


class StateNormalizer:
    
    def __init__(self):
        self.state_mean = torch.tensor([2.37E-4, 9.316E-3])
        self.state_std = torch.tensor([8.225E-3, 6.9441E-2])

    def normalize_state(self, states):
        return (
            (states - self.state_mean.to(states.device)) / (self.state_std.to(self.state_std.device) + 1e-6)
        )