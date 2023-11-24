import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

df = pd.read_csv("../common/train_ship_segmentations_v2.csv")
df = df.groupby('ImageId').first().reset_index()
print(df['ImageId'].nunique())

df['HasMask'] = df['EncodedPixels'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
df["MaskId"] = df['ImageId'].str.replace("jpg", "png")

train_data, temp_data = train_test_split(df, train_size=0.01, test_size=0.01, stratify=df["HasMask"], random_state=20)
val_data, test_data = train_test_split(temp_data, train_size=0.5, stratify=temp_data["HasMask"], random_state=40)
# Now train_data contains 1%, valid_data contains 0.5%, and test_data contains  0.5%

test_data = test_data.loc[test_data["HasMask"] == 1]

print(f"train: {train_data.shape[0]}",
      f"val: {val_data.shape[0]}",
      f"test: {test_data.shape[0]}", sep='\n')

# Define the paths for the destination folders
train_folder = '../train/'
val_folder = '../validation/'
test_folder = '../test/'
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


def create_folders_for_images_and_masks(dst):
    os.makedirs(dst + "images/", exist_ok=True)
    os.makedirs(dst + "masks/", exist_ok=True)


create_folders_for_images_and_masks(train_folder)
create_folders_for_images_and_masks(val_folder)
create_folders_for_images_and_masks(test_folder)


def copy_data(dataframe, src_folder,  dst_folder):
    for index, row in dataframe.iterrows():
        shutil.copy(os.path.join(src_folder, "masks/", row["MaskId"]), os.path.join(dst_folder, "masks/"))
        shutil.copy(os.path.join(src_folder, "images/", row["ImageId"]), os.path.join(dst_folder, "images/"))


# Move images to the corresponding folders based on the split
data_folder = "../common/"

copy_data(train_data, data_folder, train_folder)
copy_data(val_data, data_folder, val_folder)
copy_data(test_data, data_folder, test_folder)
