
from PIL import Image
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import glob
from typing import Tuple
from natsort import natsorted, ns
import shutil

def rename_files(directory: str) -> None:
    try:
        subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        for subdir in subdirectories:
            files = os.listdir(subdir)
            files = natsorted(files, alg=ns.IGNORECASE)
            base_name = os.path.basename(subdir)

            for i, filename in enumerate(files):
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, f"{base_name}_{i+1}.jpg")

                if not os.path.exists(new_file):
                    os.rename(old_file, new_file)
                    print(f"Renamed {old_file} to {new_file}")
                else:
                    print(f"Skipped renaming {old_file} because {new_file} already exists")
    except Exception as e:
        print(f"Error renaming files: {e}")

def resize_images(original_directory: str, resize_directory: str, size: Tuple[int, int]) -> None:
    if os.path.exists(resize_directory):
        # Clear the contents of the directory
        shutil.rmtree(resize_directory)
    os.makedirs(resize_directory)

    for subdir , dir , files in os.walk(original_directory) :
        for filename in files:
            if filename.endswith('.jpg'):

                input_path = os.path.join(subdir, filename)
                relative_path =os.path.relpath(subdir, original_directory)
                output_subdir = os.path.join(resize_directory, relative_path)
                
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir, exist_ok=True)

                resize_filename = f"resized_{filename}"
                output_path = os.path.join(output_subdir, resize_filename)
                try:
                    with Image.open(input_path) as img :
                        img = img.resize(size, Image.LANCZOS)
                        img.save(output_path)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

def augment_images(images_directory: str, augmented_images_directory: str, datagen: any, num_augmented_copies: int) -> None:
    if os.path.exists(augmented_images_directory):
        # Clear the contents of the directory
        shutil.rmtree(augmented_images_directory)
    os.makedirs(augmented_images_directory)
    try:
        for subdir , dir , files in os.walk(images_directory):
            for image_name in files:
                if image_name.endswith('.jpg'):  
                    images_path = os.path.join(subdir, image_name)

                    # adjust images
                    with Image.open(images_path) as img:
                        img = img.convert('RGB')
                    img_array = np.array(img) / 255.0
                    img_batch = img_array.reshape((1,) + img_array.shape)
                    
                    # Create a directory path that mirrors the structure of the original
                    relative_path = os.path.relpath(subdir, images_directory)
                    target_subdir = os.path.join(augmented_images_directory, relative_path)
                    if not os.path.exists(target_subdir):
                        os.makedirs(target_subdir)

                    filename_without_extension = os.path.splitext(image_name)[0]
                    save_prefix = 'aug_' + filename_without_extension
                    i = 0
                    for batch in datagen.flow(img_batch, batch_size=1, save_to_dir=target_subdir, save_prefix=save_prefix, save_format='jpg'):
                        i += 1
                        if i >= num_augmented_copies:  
                            break
    except Exception as e:
        print(f"Error augmenting image {image_name}: {e}")

def main(original_directory: str, resize_images_directory: str, augmented_images_directory: str, num_augmented_copies: int, size=(224, 224)) -> None:

    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    rename_files(original_directory)

    resize_images(original_directory, resize_images_directory, size)
        
    augment_images(resize_images_directory, augmented_images_directory, datagen, num_augmented_copies)
    

if __name__ == "__main__":
    main(
        original_directory=r'C:\Users\purin\Desktop\ImageClassifier-Animal\ImageClassifier-Animal\data\animal_dataset\animals\animals',
        resize_images_directory=r'C:\Users\purin\Desktop\ImageClassifier-Animal\ImageClassifier-Animal\data\resize_images',
        augmented_images_directory=r'C:\Users\purin\Desktop\ImageClassifier-Animal\ImageClassifier-Animal\data\augmented_images',
        num_augmented_copies=3
    )
