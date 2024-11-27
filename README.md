
# Building custom CNN model
# Cats vs Dogs Image Classification

This project implements a deep learning pipeline for classifying images of cats and dogs using TensorFlow/Keras. The dataset is organized into training, validation, and test sets, and augmentation techniques are applied to improve model performance.

# Features
Data Visualization:

- Plot class distribution for training, validation, and test datasets.
- Display example images from each class.
Custom Functions:

- Count the number of images per class.
- Plot sample images for visualization.
- Getting class weights.

Data Augmentation:

- Apply transformations such as rotation, zoom, flips, and shifts.
Data Generators:

- Use ImageDataGenerator for training, validation, and test data preparation.

# Dataset Structure
for more clarification here is how my data was organized.






```bash
data/
├── train/
│   ├── cats/
│   └── dogs/
├── valid/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/

```
# How to Use
- Dependencies: 
Install required libraries
- Downloading data:
Visit the Kaggle link: https://www.kaggle.com/datasets/d4rklucif3r/cat-and-dogs

Download and extract the dataset.
Organize Data for Training and Validation

Create a valid folder alongside the train folder.
Inside valid, create two subfolders: cats and dogs.
Move Images into Validation

Move 1k images from train/cats to valid/cats.
Move 1k images from train/dogs to valid/dogs.

renaming the folder names (optinal).
- Run the Notebook:

Debugging lines added to ease the further error resolving.
- Make a predication and evaluating the model:
You can use two provided images cat.jpg and dog.jpg to make a prediction and testing you model.
# Key Code Highlights
- Plot Class Distribution:

Visualizes the number of images in each class for all datasets.
- Sample Visualization:

Displays sample images from the training set, labeled by class.
- Data Augmentation:

Implements advanced augmentation techniques for robust model training.

# Next step
- Train and fine-tune a CNN model using the prepared generators.
- Evaluate performance on the test set.
- Experiment with transfer learning for better accuracy.

# Notes
- You can utilize it for any classificaton problems. NOT strictlly for dogs and cats. But you have to modify the code according to your needs.
- To continue trainng the model after intial number of epochs use code below:
```bash
# Define the learning rate schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=5000,
    decay_rate=0.8,
    staircase=True)

# Create a new optimizer with the learning rate schedule
opt = Adam(learning_rate=lr_schedule)

# Re-compile the model with the new optimizer
dogs_cats_cnn_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Set the callbacks again (for checkpoint, early stopping, etc.)
callbacks = [checkpoint, early_stopping]
# Load the best model saved during previous training
dogs_cats_cnn_model = keras.models.load_model('best_model.keras')

steps_per_epoch = trainGen.samples // trainGen.batch_size
validation_steps = valGen.samples // valGen.batch_size
# Continue training the model for more epochs
history = dogs_cats_cnn_model.fit(
    trainGen,
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # total num of epochs
    validation_data=valGen,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stopping],  # Continue using the same callbacks
    initial_epoch=history.epoch[-1] if 'history' in locals() else 0
)
# Continue from the last epoch
```

 
    
