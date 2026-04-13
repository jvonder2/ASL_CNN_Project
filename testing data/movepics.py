import os
import shutil

# 🔧 UPDATE THIS ROOT PATH
BASE_DIR = r"C:\Users\jvond\ML_Project\training data"

SOURCE_FOLDERS = [
    "jackson_data",
    "katie_tracy",
    "roberto_pereira",
    "subset_data"
]

DEST_FOLDER = "merged_finetune"

DEST_PATH = os.path.join(BASE_DIR, DEST_FOLDER)
os.makedirs(DEST_PATH, exist_ok=True)

print("Merging datasets...\n")

for source in SOURCE_FOLDERS:
    source_path = os.path.join(BASE_DIR, source)

    print(f"Processing: {source}")

    for class_name in os.listdir(source_path):
        class_src_path = os.path.join(source_path, class_name)

        if not os.path.isdir(class_src_path):
            continue

        # Create class folder in destination
        class_dst_path = os.path.join(DEST_PATH, class_name)
        os.makedirs(class_dst_path, exist_ok=True)

        for filename in os.listdir(class_src_path):
            src_file = os.path.join(class_src_path, filename)

            if not os.path.isfile(src_file):
                continue

            # 🔥 Avoid filename collisions
            new_filename = f"{source}_{filename}"
            dst_file = os.path.join(class_dst_path, new_filename)

            shutil.copy(src_file, dst_file)

    print(f"✅ Done: {source}\n")

print("🎉 All datasets merged successfully!")
print(f"Merged folder: {DEST_PATH}")