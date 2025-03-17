import cv2
import random
import numpy as np
import os
import shutil
import zipfile

def augment_image(image):
    """Applies a series of random augmentations to the image."""
    # Horizontal Flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Rotation
    angle = random.randint(-30, 30)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    image = cv2.warpAffine(image, matrix, (w, h))

    # Scaling
    scale = random.uniform(0.8, 1.2)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Cropping (random 80-100% of image)
    h, w = image.shape[:2]  # Get new dimensions after scaling
    crop_x = random.randint(0, max(1, w//5))
    crop_y = random.randint(0, max(1, h//5))
    if crop_y < h and crop_x < w:  # Ensure valid crop dimensions
        image = image[crop_y:h-crop_y, crop_x:w-crop_x]

    # Brightness/Contrast
    alpha = random.uniform(0.7, 1.3)  # Contrast
    beta = random.randint(-30, 30)  # Brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return image

def extract_and_process_zip(zip_path, dataset_dir):
    """Extract a ZIP file and handle nested directories."""
    print(f"Extracting zip file to {dataset_dir}")

    # Check the ZIP file contents before extraction
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"ZIP contains {len(file_list)} files/folders")
        print(f"First 10 entries: {file_list[:10]}")
        
        # Extract the ZIP file
        zip_ref.extractall(dataset_dir)

    print(f"Extraction complete. Dataset directory contents: {os.listdir(dataset_dir)}")

    # Check if we need to handle a nested directory structure
    # Sometimes ZIP files contain a single root folder
    if len(os.listdir(dataset_dir)) == 1:
        first_item = os.path.join(dataset_dir, os.listdir(dataset_dir)[0])
        if os.path.isdir(first_item):
            print(f"Found single root directory: {first_item}")
            # If the ZIP contained a single folder, use its contents as our dataset
            if len(os.listdir(first_item)) > 0:
                print(f"Moving contents from {first_item} to {dataset_dir}")
                for item in os.listdir(first_item):
                    shutil.move(
                        os.path.join(first_item, item),
                        os.path.join(dataset_dir, item)
                    )
                # Remove the now-empty directory
                os.rmdir(first_item)
                print(f"After restructuring, dataset directory contains: {os.listdir(dataset_dir)}")

def process_dataset_for_training(dataset_dir, augmented_dir):
    """Process each person's folder, copy and augment images."""
    person_count = 0
    total_images = 0
    augmented_images = 0
    
    print(f"Dataset directory contents: {os.listdir(dataset_dir)}")

    for person_folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, person_folder)
        
        if not os.path.isdir(folder_path):
            print(f"Skipping {person_folder} as it's not a directory")
            continue  # Skip if not a directory
        
        person_count += 1
        output_folder_path = os.path.join(augmented_dir, person_folder)
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Copy original images to augmented directory
        images = []
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_name)
                new_img_path = os.path.join(output_folder_path, img_name)
                print(f"Copying {img_path} to {new_img_path}")
                shutil.copy2(img_path, new_img_path)
                images.append(img_name)
        
        original_count = len(images)
        total_images += original_count
        
        print(f"Processing {person_folder}: {original_count} original images")
        
        # If no images were found, print a warning
        if original_count == 0:
            print(f"WARNING: No images found for {person_folder}!")
            print(f"Folder contents: {os.listdir(folder_path)}")
            continue
        
        # Apply augmentation to create additional images
        count = original_count + 1
        target_count = min(500, max(50, original_count * 10))  # Aim for 10x but cap at 500
        
        print(f"Augmenting images for {person_folder} from {original_count} to target {target_count}")
        
        while len(images) < target_count and original_count > 0:
            # Pick a random image to augment
            img_name = random.choice(images[:original_count])  # Only choose from original images
            img_path = os.path.join(folder_path, img_name)
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading {img_path}, skipping...")
                continue
            
            # Apply augmentation
            try:
                augmented_img = augment_image(img)
                
                # Save new image in the output folder
                new_img_name = f"{person_folder}_{count}.png"
                new_img_path = os.path.join(output_folder_path, new_img_name)
                cv2.imwrite(new_img_path, augmented_img)
                
                images.append(new_img_name)  # Add new image to list
                count += 1
                augmented_images += 1
            except Exception as e:
                print(f"Error augmenting image: {str(e)}")
        
        print(f"Augmentation complete for {person_folder}, now contains {len(images)} images")
    
    return person_count, total_images, augmented_images 