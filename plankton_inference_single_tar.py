# plankton_inference_single_tar.py

import argparse
import tarfile
import tempfile
from pathlib import Path
import pandas as pd
import random
import torch
from fastai.vision.all import *
from src.utils import process_predictions_to_dataframe  # Your existing function

# -------------------------
# Create minimal dummy dataset for FastAI
# -------------------------
def ensure_dummy_dataset(model_weights, train_dataset_path):
    train_dataset_path = Path(train_dataset_path)
    if train_dataset_path.exists() and any(train_dataset_path.iterdir()):
        return  # Already populated

    # Determine class names from .csv or default
    class_names = [f"class_{i}" for i in range(2)]  # Minimal example: 2 classes
    train_dataset_path.mkdir(parents=True, exist_ok=True)
    for cname in class_names:
        class_path = train_dataset_path / cname
        class_path.mkdir(exist_ok=True)
        # Create 1 dummy image per class
        dummy_img = Image.new("L", (10, 10))
        dummy_img.save(class_path / f"dummy_0.tif")

# -------------------------
# Process a single .tar file
# -------------------------
def process_single_tar(tar_file_path, model_weights, train_dataset, cruise_name,
                       batch_size=64, density_constant=1.0):
    ensure_dummy_dataset(model_weights, train_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup FastAI DataBlock
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=RandomSplitter(),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'),
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    )
    dls = block.dataloaders(train_dataset, bs=batch_size, num_workers=0)
    learn = vision_learner(dls, resnet50, metrics=error_rate, pretrained=False)
    if torch.cuda.device_count() > 1:
        learn.model = torch.nn.DataParallel(learn.model)
    learn.model.to(device)
    learn.load(model_weights, weights_only=False)

    # Extract tar into temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with tarfile.open(tar_file_path, "r") as tar:
                tar.extractall(path=temp_dir)
        except Exception as e:
            print(f"[ERROR] Could not extract {tar_file_path}: {e}")
            return None

        imgs = get_image_files(temp_dir)
        imgs.sort()
        if len(imgs) == 0:
            print(f"[WARNING] No images found in {tar_file_path}")
            return None

        # Predictions
        try:
            dl = learn.dls.test_dl(imgs)
            preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
        except Exception as e:
            print(f"[WARNING] Prediction failed for some images in {tar_file_path}: {e}")
            return None

        # Process predictions into DataFrame
        df = process_predictions_to_dataframe(
            imgs=imgs,
            preds=preds,
            label_numeric=label_numeric,
            vocab=learn.dls.vocab,
            cruise_name=cruise_name,
            date_str="single_tar",
            time_str=Path(tar_file_path).stem,
            timestamp_path=temp_dir,
            results_dir=Path(temp_dir),
            processed_dir=Path(temp_dir),
            density_constant=density_constant,
            csv_filename=None,
            tar_file_path=tar_file_path
        )

    return df

# -------------------------
# CLI interface for Azure ML
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plankton Inference for a Single .tar file")
    parser.add_argument("--tar_file", type=str, required=True)
    parser.add_argument("--model_weights", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--cruise_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--density_constant", type=float, default=1.0)
    args = parser.parse_args()

    df_result = process_single_tar(
        tar_file_path=args.tar_file,
        model_weights=args.model_weights,
        train_dataset=args.train_dataset,
        cruise_name=args.cruise_name,
        batch_size=args.batch_size,
        density_constant=args.density_constant
    )

    if df_result is not None:
        output_path = Path("outputs/results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(output_path, index=False)
        print(f"[INFO] Results saved to {output_path}")
    else:
        print("[INFO] No results generated for this tar file.")
