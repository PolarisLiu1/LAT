import re
import glob
import pyarrow as pa
from tqdm import tqdm
import polars as pl
import torch
import io
from PIL import Image
from typing import List, Optional, Union


def multi_image():  # 
    name = "paper" # paper/wiki/fine-web
    DATA_ROOT = "../date/visa/{}".format(name)
    columns_to_read = ['image', 'id', 'candidates', 'pos_idx']
    
    # read parquet files
    df = (
        pl.scan_parquet(f"{DATA_ROOT}/data/*.parquet")
        .select(columns_to_read)
        .collect()
    )
    
    for item in tqdm(df.iter_rows(named=True)):
        current_id = item['id']
        current_image = item['image']['bytes']
        with open(f"{DATA_ROOT}/image/{current_id}.bin", "wb") as f:
            f.write(current_image)
        
        


if __name__ == "__main__":
    multi_image()
