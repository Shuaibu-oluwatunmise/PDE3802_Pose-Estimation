import cv2
import os
from pathlib import Path
from tqdm import tqdm

# ============================================
# USER CONFIGURATION
# ============================================

# Input folder containing class subfolders with videos
# OLD:
# video_folder = Path("./Data_Video")
# NEW (matches your actual structure):
video_folder = Path("./data/raw_videos")

# Output folder for extracted frames
# OLD:
# output_folder = Path("./images")
# NEW:
output_folder = Path("./data/images")


# Frame extraction rate (extract every N frames)
# 1 = extract every frame (30 FPS if video is 30 FPS)
frame_skip = 1

# ============================================
# MAIN SCRIPT
# ============================================

print("=" * 60)
print("VIDEO TO IMAGES EXTRACTION")
print("=" * 60)

# Check if GPU is available for OpenCV
if cv2.ocl.useOpenCL():
    print("✓ GPU acceleration enabled for OpenCV")
else:
    print("⚠ GPU acceleration not available, using CPU")

# Create output folder
output_folder.mkdir(exist_ok=True)

# Find all class folders
class_folders = [f for f in video_folder.iterdir() if f.is_dir()]
class_names = sorted([f.name for f in class_folders])

print(f"\nFound {len(class_names)} classes: {class_names}\n")

# Process each class
total_images_extracted = 0

for class_name in class_names:
    print(f"{'=' * 60}")
    print(f"Processing class: {class_name}")
    print(f"{'=' * 60}")
    
    class_video_folder = video_folder / class_name
    class_output_folder = output_folder / class_name
    class_output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(class_video_folder.glob(f"*{ext}")))
    
    video_files = sorted(video_files)
    
    if len(video_files) == 0:
        print(f"  ⚠ No videos found in {class_video_folder}")
        continue
    
    print(f"  Found {len(video_files)} video(s)")
    
    # Counter for naming images across all videos
    image_counter = 1
    
    # Process each video
    for video_file in video_files:
        print(f"\n  📹 Processing: {video_file.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            print(f"    ✗ Error: Could not open video {video_file.name}")
            continue
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"    FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Extract frames
        frame_number = 0
        extracted_from_video = 0
        
        pbar = tqdm(total=total_frames, desc=f"    Extracting", unit="frame")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_number % frame_skip == 0:
                # Generate filename
                image_name = f"{class_name}_{image_counter:04d}.jpg"
                image_path = class_output_folder / image_name
                
                # Save frame
                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                image_counter += 1
                extracted_from_video += 1
            
            frame_number += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        print(f"    ✓ Extracted {extracted_from_video} frames from {video_file.name}")
        total_images_extracted += extracted_from_video
    
    print(f"\n  ✓ Total images for {class_name}: {image_counter - 1}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("✅ EXTRACTION COMPLETE!")
print("=" * 60)

# Count final images per class
print(f"\n📊 Final Dataset Statistics:")
for class_name in class_names:
    class_output_folder = output_folder / class_name
    if class_output_folder.exists():
        image_count = len(list(class_output_folder.glob("*.jpg")))
        print(f"   {class_name}: {image_count} images")

print(f"\n📁 Output folder: {output_folder.absolute()}")
print(f"🖼️  Total images extracted: {total_images_extracted}")
print("=" * 60)