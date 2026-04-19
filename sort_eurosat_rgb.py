import os
import shutil

# Path to your RGB folder (original downloaded images)
src_dir = r"D:\Feature_extraction_from_satellite_images final\data\EuroSAT_RGB"

# Destination folder where sorted class folders will be created
dst_dir = r"D:\Feature_extraction_from_satellite_images final\data\EuroSAT_RGB_sorted"

# Loop through images and sort them into class subfolders
for filename in os.listdir(src_dir):
    if filename.endswith(".jpg"):
        # Extract class name from filename (before underscore)
        class_name = filename.split("_")[0]
        class_folder = os.path.join(dst_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        shutil.copy(os.path.join(src_dir, filename), os.path.join(class_folder, filename))

print("Images sorted into class folders successfully.")
