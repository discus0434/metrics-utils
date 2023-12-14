from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from tqdm import tqdm


class CLIPScore:
    def __init__(self, use_lcm: bool = True, lora_dim: int = 64):
        self.clip_score_fn = partial(
            clip_score, model_name_or_path="openai/clip-vit-base-patch16"
        )
        self.annotations = pd.read_parquet("/workspace/data/30k/annotations.parquet")

        self.generated_dir = Path(
            f"/workspace/data/30k/{lora_dim if use_lcm else 'original'}_30k"
        )

    def __call__(self, batch_size: int = 32) -> float:
        clip_scores = []
        images = []
        captions = []
        batch_count = 0
        for i, (_, row) in tqdm(enumerate(self.annotations.iterrows()), total=len(self.annotations)):
            images.append(
                torch.from_numpy(
                    np.array(Image.open(self.generated_dir / row["file_name"]))
                )
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            captions.append(row["caption"])
            batch_count += 1

            if batch_count < batch_size:
                continue
            else:
                clip_score = self.clip_score_fn(
                    torch.cat(images).to("cuda"),
                    captions,
                ).detach().cpu().numpy()
                clip_scores.append(clip_score)

                images = []
                captions = []
                batch_count = 0

                print(f"CLIP Score: {np.mean(clip_scores)}")

        return np.mean(clip_scores)


if __name__ == "__main__":
    for lora_dim in [1, 2, 4, 8, 16, 32, 64, 128, 256, None]:
        if lora_dim is None:
            score = CLIPScore(use_lcm=False)()
        else:
            score = CLIPScore(use_lcm=True, lora_dim=lora_dim)()

        print(f"{lora_dim if lora_dim is not None else 'original'}: {score}")

        with open("/workspace/data/30k/clip_scores.txt", "a") as f:
            f.write(f"{lora_dim if lora_dim is not None else 'original'}: {score}\n")
