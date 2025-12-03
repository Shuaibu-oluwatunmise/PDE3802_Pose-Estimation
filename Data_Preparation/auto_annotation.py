#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# GPU enabled - using CUDA
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from pathlib import Path
import shutil
import random

# ============================================
# USER CONFIGURATION
# ============================================

# Input folder containing class subfolders
input_folder = Path("../data/images")

# Map folder names to Grounding DINO text prompts
prompt_mapping = {
    "headset": "headset",
    "wallet": "wallet",
    "notebook": "notebook",
    "repair_mat": "screw mat",
    "game_box": "card box"
}

# Output folder for final YOLO dataset
output_folder = Path("../data/dataset")


# Train/Val/Test split percentages
train_split = 0.8
val_split = 0.1
test_split = 0.1

# ============================================
# STEP 1: DISCOVER CLASSES AND RENAME FILES
# ============================================

print("=" * 60)
print("STEP 1: Discovering classes and renaming files...")
print("=" * 60)

# Find all class folders
class_folders = [f for f in input_folder.iterdir() if f.is_dir()]
class_names = [f.name for f in class_folders]

print(f"Found {len(class_names)} classes: {class_names}")

# Rename files in each class folder
for class_folder in class_folders:
    class_name = class_folder.name
    image_files = sorted(list(class_folder.glob("*.*")))
    
    # Filter out non-image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in image_files if f.suffix.lower() in image_extensions]
    
    print(f"\n  Renaming {len(image_files)} images in '{class_name}'...")
    
    # Rename files
    for idx, img_file in enumerate(image_files, start=1):
        ext = img_file.suffix
        new_name = f"{class_name}{idx:04d}{ext}"
        new_path = class_folder / new_name
        
        # Only rename if different
        if img_file.name != new_name:
            img_file.rename(new_path)
    
    print(f"    ✓ Renamed to {class_name}0001{ext} ... {class_name}{len(image_files):04d}{ext}")

# ============================================
# STEP 2: GENERATE CLASS MAPPING
# ============================================

print("\n" + "=" * 60)
print("STEP 2: Generating class mapping...")
print("=" * 60)

classes = {}
for idx, class_name in enumerate(sorted(class_names)):
    if class_name not in prompt_mapping:
        print(f"WARNING: '{class_name}' not found in prompt_mapping. Using folder name as prompt.")
        prompt = class_name.lower()
    else:
        prompt = prompt_mapping[class_name]
    
    classes[class_name] = ({prompt: class_name.lower()}, idx)
    print(f"  {class_name}: prompt='{prompt}', class_id={idx}")

# ============================================
# STEP 3: ANNOTATE EACH CLASS
# ============================================

print("\n" + "=" * 60)
print("STEP 3: Annotating each class with Grounding DINO...")
print("=" * 60)

annotation_folders = []

for class_name, (ontology_dict, class_id) in classes.items():
    print(f"\n📦 Processing {class_name} (class {class_id})...")
    
    # Create ontology
    ontology = CaptionOntology(ontology_dict)
    
    # Initialize model
    base_model = GroundingDINO(ontology=ontology)
    
    # Annotate
    output_annotation_folder = Path(f"./annotations_{class_name}")
    annotation_folders.append(output_annotation_folder)
    
    base_model.label(
        input_folder=str(input_folder / class_name),
        output_folder=str(output_annotation_folder),
        extension="*"  # Accept all image extensions
    )
    
    # Remap class IDs from 0 to class_id
    print(f"  🔄 Remapping class IDs to {class_id}...")
    label_dirs = [
        output_annotation_folder / "train" / "labels",
        output_annotation_folder / "valid" / "labels"
    ]
    
    for label_dir in label_dirs:
        if label_dir.exists():
            for label_file in label_dir.glob("*.txt"):
                # Read labels
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # Remap class 0 to class_id
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = str(class_id)
                        new_lines.append(' '.join(parts) + '\n')
                
                # Write back
                with open(label_file, 'w') as f:
                    f.writelines(new_lines)
    
    print(f"  ✓ {class_name} annotation complete!")

# ============================================
# STEP 4: MERGE AND SPLIT DATASET
# ============================================

print("\n" + "=" * 60)
print("STEP 4: Merging datasets with stratified 80/10/10 split...")
print("=" * 60)

# Create output structure
for split in ["train", "val", "test"]:
    (output_folder / split / "images").mkdir(parents=True, exist_ok=True)
    (output_folder / split / "labels").mkdir(parents=True, exist_ok=True)

# Collect all images and labels from each class
for class_name in class_names:
    annotation_folder = Path(f"./annotations_{class_name}")
    
    print(f"\n  Processing {class_name}...")
    
    # Collect all images and labels (from train and valid folders created by autodistill)
    all_images = []
    all_labels = []
    
    for split_folder in ["train", "valid"]:
        img_dir = annotation_folder / split_folder / "images"
        lbl_dir = annotation_folder / split_folder / "labels"
        
        if img_dir.exists():
            for img_file in img_dir.glob("*.*"):
                label_file = lbl_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    all_images.append(img_file)
                    all_labels.append(label_file)
    
    total_images = len(all_images)
    print(f"    Found {total_images} annotated images")
    
    if total_images == 0:
        continue
    
    # Shuffle with same seed for reproducibility
    combined = list(zip(all_images, all_labels))
    random.seed(42)
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)
    
    # Calculate split indices
    train_end = int(total_images * train_split)
    val_end = train_end + int(total_images * val_split)
    
    # Split data
    train_imgs = all_images[:train_end]
    train_lbls = all_labels[:train_end]
    
    val_imgs = all_images[train_end:val_end]
    val_lbls = all_labels[train_end:val_end]
    
    test_imgs = all_images[val_end:]
    test_lbls = all_labels[val_end:]
    
    print(f"    Split: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    # Copy files to output folder
    # Copy files to output folder
    for i in range(len(train_imgs)):
        shutil.copy(train_imgs[i], output_folder / "train" / "images" / train_imgs[i].name)
        shutil.copy(train_lbls[i], output_folder / "train" / "labels" / train_lbls[i].name)
    
    for i in range(len(val_imgs)):
        shutil.copy(val_imgs[i], output_folder / "val" / "images" / val_imgs[i].name)
        shutil.copy(val_lbls[i], output_folder / "val" / "labels" / val_lbls[i].name)
    
    for i in range(len(test_imgs)):
        shutil.copy(test_imgs[i], output_folder / "test" / "images" / test_imgs[i].name)
        shutil.copy(test_lbls[i], output_folder / "test" / "labels" / test_lbls[i].name)

# ============================================
# STEP 5: CREATE DATA.YAML
# ============================================

print("\n" + "=" * 60)
print("STEP 5: Creating data.yaml...")
print("=" * 60)

# Create class names list in correct order
sorted_class_names = [name.lower() for name in sorted(class_names)]

yaml_content = f"""path: {output_folder.absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(sorted_class_names)}
names: {sorted_class_names}
"""

with open(output_folder / "data.yaml", "w") as f:
    f.write(yaml_content)

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("✅ COMPLETE! Dataset created successfully!")
print("=" * 60)

train_count = len(list((output_folder / "train" / "images").glob("*.*")))
val_count = len(list((output_folder / "val" / "images").glob("*.*")))
test_count = len(list((output_folder / "test" / "images").glob("*.*")))

print(f"\n📊 Dataset Statistics:")
print(f"   📁 Output folder: {output_folder.absolute()}")
print(f"   🏷️  Classes ({len(sorted_class_names)}): {sorted_class_names}")
print(f"   🚂 Train: {train_count} images")
print(f"   ✅ Val: {val_count} images")
print(f"   🧪 Test: {test_count} images")
print(f"   📝 Total: {train_count + val_count + test_count} images")
print(f"\n🎯 Ready for YOLOv8 training!")
print("=" * 60)