import zipfile
import os


archive_path = 'train_v2.zip'
extract_folder = 'images'

os.makedirs(extract_folder, exist_ok=True)

# Extract the contents of the archive
with zipfile.ZipFile(archive_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
