from pathlib import Path
from functools import partial
from PIL import Image

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from torchmetrics.image.fid import FrechetInceptionDistance


class FID:
    def __init__(self, use_lcm: bool = True, lora_dim: int = 64):
        self.fid = FrechetInceptionDistance(normalize=True).to("cuda")
        self.annotations = pd.read_parquet("/workspace/data/30k/annotations.parquet")

        self.original_dir = Path("/workspace/data/30k/images/")
        self.generated_dir = Path(
            f"/workspace/data/30k/{lora_dim if use_lcm else 'original'}_30k/"
        )

    def __call__(self) -> float:
        for _, row in tqdm(self.annotations.iterrows(), total=len(self.annotations)):
            self.fid.update(
                torch.from_numpy(
                    np.array(
                        Image.open(self.original_dir / row["file_name"]).convert("RGB")
                    )
                )
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .to("cuda", torch.float32),
                real=True,
            )
            self.fid.update(
                torch.from_numpy(
                    np.array(
                        Image.open(self.generated_dir / row["file_name"]).convert("RGB")
                    )
                )
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .to("cuda", torch.float32),
                real=False,
            )

        return float(self.fid.compute().item())


if __name__ == "__main__":
    for lora_dim in [1, 2, 4, 8, 16, 32, 64, 128, 256, None]:
        if lora_dim is None:
            score = FID(use_lcm=False)()
        else:
            score = FID(use_lcm=True, lora_dim=lora_dim)()

        print(f"{lora_dim if lora_dim is not None else 'original'}: {score}")

        with open("/workspace/data/30k/fid.txt", "a") as f:
            f.write(f"{lora_dim if lora_dim is not None else 'original'}: {score}\n")
