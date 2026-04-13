import os
import random
import shutil

# 🔧 CHANGE THESE PATHS
SOURCE_DIR = r"C:\Users\jvond\ML_Project\training data\main_data"
TEST_DIR = r"C:\Users\jvond\ML_Project\training data\subset_data"

NUM_PER_CLASS = 200

os.makedirs(TEST_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(images) < NUM_PER_CLASS:
        print(f"Skipping {class_name} (only {len(images)} images)")
        continue

    selected = random.sample(images, NUM_PER_CLASS)

    # create test class folder
    test_class_path = os.path.join(TEST_DIR, class_name)
    os.makedirs(test_class_path, exist_ok=True)

    for img in selected:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_path, img)

        shutil.copy(src, dst)  # use move() if you want to remove from train

    print(f"{class_name}: copied {NUM_PER_CLASS} images")

print("\nTest dataset created!")