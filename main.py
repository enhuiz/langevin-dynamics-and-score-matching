import imageio
import argparse
import yaml
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


class PixelSampler:
    def __init__(self, path, threshold):
        image = Image.open(path)
        image = image.convert("I")
        image = np.array(image)
        image = image < threshold

        points = np.stack(np.where(image), 1)
        points = (points / image.shape - 0.5) * 2  # => [-1, 1]

        self.image = image
        self.points = points

    @property
    def size(self):
        return np.array(self.image.shape)

    def plot(self, points=None):
        if points is None:
            plt.imshow(self.image, cmap="gray")
        else:
            points = points / 2 + 0.5
            points = points * (self.size - 1)
            points = points.astype(np.int)
            slot = np.zeros(self.size)
            slot[points[:, 0], points[:, 1]] = 1
            plt.imshow(slot, cmap="gray")

    def sample(self, n_samples=1):
        choices = np.arange(len(self.points))
        choices = np.random.choice(choices, n_samples)
        return self.points[choices]


class Model(nn.Module):
    def __init__(self, σ):
        super().__init__()
        # std of the noise to add
        self.σ = σ
        # score function scaled by σ (reparametrization)
        self.σψ = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def array2tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def train(self, ps, n_iters, batch_size=32, η=1e-3):
        optimizer = torch.optim.Adam(self.σψ.parameters(), lr=η)
        pbar = tqdm.trange(n_iters)
        for _ in pbar:
            x = self.array2tensor(ps.sample(batch_size))
            z = torch.randn_like(x)
            # Reparametrization: learn -z instead of the real gradient (-z / σ)
            loss = F.l1_loss(self.σψ(x + self.σ * z), -z)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Loss: {loss.item():.3g}")

    @torch.no_grad()
    def sample(self, n_samples, n_steps, Ɛ, x=None):
        sqrt_Ɛ = math.sqrt(Ɛ)
        if x is None:
            # uniformly sample from the canvas
            x = torch.rand([n_samples, 2], device=self.device) * 2 - 1
        for _ in range(n_steps):
            z = torch.randn_like(x)
            x = x + 0.5 * Ɛ * self.σψ(x) / self.σ + sqrt_Ɛ * z
        return x


def load_args(path):
    with open(path, "r") as f:
        args = argparse.Namespace(**yaml.load(f, yaml.Loader))
    args.image = Path(args.image)
    args.ckpt = args.image.with_suffix(".pth")
    args.frame_dir = args.ckpt.with_suffix("")
    args.gif = args.image.with_suffix(".gif")
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=load_args)
    parser.add_argument("action", choices=["train", "test"])
    args = parser.parse_args()
    args.config.action = args.action
    args = args.config
    print(args)

    ps = PixelSampler(args.image, args.threshold)

    model = Model(args.σ)
    model = model.cuda()

    if args.ckpt.exists():
        model.load_state_dict(
            torch.load(
                args.ckpt,
            )
        )

    if args.action == "train" and args.n_iters > 0:
        model.train(
            ps,
            args.n_iters,
            args.batch_size,
            args.η,
        )
        torch.save(
            model.state_dict(),
            args.ckpt,
        )
    elif args.action == "test":
        real_points = ps.sample(args.n_samples)
        fake_points = None
        steps_per_frame = args.n_steps // args.n_frames
        args.frame_dir.mkdir(exist_ok=True)

        for i in tqdm.tqdm(range(args.n_frames)):
            plt.subplot(121)
            ps.plot(real_points)

            plt.subplot(122)
            fake_points = model.sample(
                args.n_samples,
                steps_per_frame,
                args.Ɛ,
                fake_points,
            )
            ps.plot(fake_points.clip(-1, 1).cpu().numpy())

            plt.savefig(f"{args.frame_dir}/{i:06d}.png", bbox_inches="tight")
            plt.clf()

        images = sorted(args.frame_dir.glob("*.png"))
        images = [imageio.imread(image) for image in images]
        imageio.mimsave(args.gif, images, duration=0.05)


if __name__ == "__main__":
    main()
