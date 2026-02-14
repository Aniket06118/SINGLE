import torch
import torch.nn as nn
from tqdm import tqdm
from forward import Diffusion
from unet import Unet
import os

from torchvision.utils import save_image

class DDPM_sampler(nn.Module):

    def __init__(self, diffusion):
        super().__init__()

        self.beta = diffusion.beta
        self.alpha = diffusion.alpha
        self.alpha_bar = diffusion.alpha_bar
        self.timesteps = diffusion.timesteps
        self.device = diffusion.beta.device

    @torch.no_grad()
    def p_sample(self, model, x, t):

        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

        #Predict noise
        eps_theta = model(x, t_tensor)

        #Reverse mean
        model_mean = sqrt_recip_alpha*(x - (beta_t/sqrt_one_minus_alpha_bar)*eps_theta)

        if t>0:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(beta_t)*noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, model, shape):

        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(self.timesteps))):
            x = self.p_sample(model, x, t)

        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    diffusion = Diffusion(timesteps=1000).to(device)

    sampler = DDPM_sampler(diffusion)

    model = Unet(
        dim=256,    #need to be resolved
        image_size=256,
        channel=3
    )

    #Subjected to change according to the saved model during training
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()

    samples = sampler.sample(model, (1, 3, 256, 256))
    samples = samples.clamp(-1, 1)
    samples = (samples+1)/2

    folder = "D:/SINGLE/Generated_images"
    index = len(os.listdir(folder))
    filename = f"{folder}/generated_{index}.png"

    save_image(samples, filename)

    return samples


if __name__ == "__main__":
    main()




