import argparse
import os

import numpy as np
from tqdm import tqdm
from PIL import Image


def extract_patches(
        input_dir: str,
        output_dir: str,
        num_patches_x: int = 4,
        num_patches_y: int = 4,
) -> None:
    """
    Splits a large image into smaller patches.

    Args:
        input_dir: Directory containing the images to be split
        output_dir: Directory where the extracted patches should be saved
        num_patches_x: How many patches should be created horizontally
        num_patches_y: How many patches should be created vertically

    Returns:
        Saves the extracted patches in output dir
    """

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The input directory {input_dir} does not exist")

    if not os.path.exists(output_dir):
        print(f"Warning! The output directory {output_dir} does not exist, creating it now...")
        os.makedirs(output_dir, exist_ok=True)

    image_files = os.listdir(input_dir)

    for file_name in tqdm(image_files):
        file_path = os.path.join(input_dir, file_name)
        img_id = file_name.split(".")[0]
        img_array = np.array(Image.open(file_path))
        patch_size = (
                img_array.shape[0] // num_patches_x,
                img_array.shape[1] // num_patches_y,
        )

        for i in range(num_patches_x):
            for j in range(num_patches_y):
                patch = img_array[
                        i * patch_size[0]:i * patch_size[0] + patch_size[0],
                        j * patch_size[1]:j * patch_size[1] + patch_size[1]
                ]

                new_img_path = os.path.join(output_dir, f"{img_id}_{i}{j}.png")

                if not os.path.exists(new_img_path):
                    img = Image.fromarray(patch.astype('uint8'))
                    img.save(new_img_path)


if __name__ == '__main__':

    extract_patches(
        input_dir="/system/user/publicdata/dracch/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set",
        output_dir="/system/user/publicdata/dracch/C. Diabetic Retinopathy Grading/patches_train",
        num_patches_x=2,
        num_patches_y=2
    )
