import pandas as pd
import numpy as np
import os
from PIL import Image

IMAGE_SIZE = (768, 768)


def rle_decode(mask_rle, shape):  # (height,width)
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if not mask_rle:  # Check if mask_rle is an empty string
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def save_mask(decoded_pixels, mask_id, save_dir):
    image_mask = Image.fromarray(decoded_pixels * 255)
    image_mask.save(os.path.join(save_dir, mask_id))


df = pd.read_csv("train_ship_segmentations_v2.csv")
df["EncodedPixels"] = df["EncodedPixels"].fillna("")

mask_save_dir = "masks"
os.makedirs(mask_save_dir, exist_ok=True)

print(f'Total number of images: {df["ImageId"].nunique()}')


for image_id in df["ImageId"].unique():
    filtered_df = df.loc[df["ImageId"] == image_id].copy()
    filtered_df['DecodedPixels'] = filtered_df['EncodedPixels'].apply(lambda x: rle_decode(x, IMAGE_SIZE))
    filtered_df = filtered_df.groupby('ImageId')[['DecodedPixels']].sum().reset_index()
    filtered_df['MaskId'] = filtered_df['ImageId'].str.replace('.jpg', '.png')
    save_mask(filtered_df['DecodedPixels'].values[0], filtered_df['MaskId'].values[0], mask_save_dir)
