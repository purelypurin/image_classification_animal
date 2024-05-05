import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import glob
from typing import List, Tuple
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def train_test_validate(original_images_base_path: str, augmented_images_base_path: str, animals: List[str]) -> Tuple[List[str], List[str], List[str]]:
    all_images = []

    # Loop through each animal type and gather images from both original and augmented directories
    for animal in animals:
        original_images_dir = os.path.join(original_images_base_path, animal)
        augmented_images_dir = os.path.join(augmented_images_base_path, animal)

        # Use glob to collect all images of this type of animal from both directories
        original_images = glob.glob(os.path.join(original_images_dir, '*.jpg'))  # Adjust the pattern if needed
        augmented_images = glob.glob(os.path.join(augmented_images_dir, '*.jpeg'))  # Adjust the pattern if needed

        all_images.extend(original_images)
        all_images.extend(augmented_images)

    # Splitting the dataset into training, validation, and test sets
    train_val_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)  # 20% for testing
    train_images, val_images = train_test_split(train_val_images, test_size=0.125, random_state=42)  # 12.5% of 80% = 10% for validation

    print(f"Total images: {len(all_images)}")
    print(f"Training set size: {len(train_images)}")
    print(f"Validation set size: {len(val_images)}")
    print(f"Test set size: {len(test_images)}")

    return train_images, val_images, test_images

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') 
    ])
    return model

def compile_and_train(model, train_generator, validation_generator, epochs=10):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  
                  metrics=['accuracy'])
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size
    )
    return history

def save_text(output_directory,image_list) :
    with open(output_directory, 'w') as file :
        for item in image_list:
            file.write('%s\n' % item)

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def main(original_images_directory, augmented_images_directory, train_test_valid_directory):
    animals = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly",
    "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer",
    "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly",
    "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog",
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo",
    "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse",
    "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot",
    "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat",
    "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake",
    "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale",
    "wolf", "wombat", "woodpecker", "zebra"
    ]
    num_classes = len(animals)

    # Split the dataset into training, testing, and validation sets
    train_images, test_images, val_images = train_test_validate(original_images_directory, augmented_images_directory, animals)
    train_images, val_images = shuffle(train_images), shuffle(val_images)
    
    #save test images
    train_text_path = os.path.join(train_test_valid_directory, 'train.txt')
    test_text_path = os.path.join(train_test_valid_directory, 'test.txt')
    validate_text_path = os.path.join(train_test_valid_directory, 'validate.txt')
    save_text(train_text_path,train_images)
    save_text(test_text_path,test_images)
    save_text(validate_text_path,val_images)
    
    # Create a DataFrame for the training and validation images
    train_df = pd.DataFrame({
        'filename': train_images,
        'class': [os.path.basename(os.path.dirname(x)) for x in train_images] 
    })
    val_df = pd.DataFrame({
        'filename': val_images,
        'class': [os.path.basename(os.path.dirname(x)) for x in val_images]
    })

    # Create ImageDataGenerator objects for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators using flow_from_dataframe
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'  
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'  
    )
    
    # Create the model
    model = create_model((224, 224, 3), num_classes)
    
    # Train the model
    history = compile_and_train(model, train_generator, validation_generator, epochs=10)
    
    # Plot graph
    plot_training_history(history)
    
    # Save the model
    model.save('animal_classifier_model.h5')

    # Optionally print out training and validation accuracy per epoch
    print(history.history['accuracy'])
    print(history.history['val_accuracy'])

if __name__ == "__main__":
    main(
        r'C:\Users\purin\Desktop\ImageClassifier-Animal\ImageClassifier-Animal\data\animal_dataset\animals\animals',
        r'C:\Users\purin\Desktop\ImageClassifier-Animal\ImageClassifier-Animal\data\augmented_images_backup_10',
        r'C:\Users\purin\Desktop\ImageClassifier-Animal\ImageClassifier-Animal\data\train_test_validate'
    )