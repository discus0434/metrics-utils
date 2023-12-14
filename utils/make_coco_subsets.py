import json
import shutil
from pathlib import Path
import pandas as pd

data_file = "/workspace/data/annotations/captions_val2014.json"
data = json.load(open(data_file))


images = data["images"]
annotations = data["annotations"]
df = pd.DataFrame(images)
df_annotations = pd.DataFrame(annotations)
df = df.merge(pd.DataFrame(annotations), how="left", left_on="id", right_on="image_id")


# keep only the relevant columns
df = df[["file_name", "caption"]]


# shuffle the dataset
df = df.sample(frac=1)


# remove duplicate images
df = df.drop_duplicates(subset="file_name")


# create a random subset
n_samples = 30000
df_sample = df.sample(n_samples)

# save the sample to a parquet file
df_sample.to_parquet("/workspace/data/subset.parquet")

subset_path = Path("/workspace/data/subset")
subset_path.mkdir(exist_ok=True)
for i, row in df_sample.iterrows():
    path = "/workspace/data/val2014/" + row["file_name"]
    shutil.copy(path, "/workspace/data/subset/")
