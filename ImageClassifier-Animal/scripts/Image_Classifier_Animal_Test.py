import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
model = load_model(r'C:\Users\purin\Desktop\ImageClassifier-Animal\ImageClassifier-Animal\models\animal_classifier_model.h5')

# Prepare the test data
# Assuming test.txt is in the current directory and contains absolute paths
test_data_path = r'C:\Users\purin\Desktop\ImageClassifier-Animal\ImageClassifier-Animal\data\train_test_validate\test.txt'
test_images = []
with open(test_data_path, 'r') as file:
    test_images = [line.strip() for line in file.readlines()]

# Create a DataFrame for the test data
test_df = pd.DataFrame({
    'filename': test_images,
    'class': [os.path.basename(os.path.dirname(x)) for x in test_images]  # Adjust if your file paths are structured differently
})

# Create an ImageDataGenerator for testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Create a test generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),  # Match the input size of your model
    batch_size=32,           # Or another batch size that fits your system
    class_mode='categorical',  # This should match the configuration of your training setup
    shuffle=False             # No need to shuffle test data
)

# Evaluate the model on the test data
results = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")
