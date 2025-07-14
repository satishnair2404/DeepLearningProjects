## Standard Imports
import os
import warnings
warnings.filterwarnings("ignore")

## ML/DL Sepcific Imports
import pandas as pd
## [OPT] Optional to improve readability of Datafram
#pd.set_option('display.max_colwidth', None)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

## Setting up path to Data files 
base_dir = r'/Users/s.nair/Downloads/DLProjects/P4/P4files'
# Create full path to the CSV
csv_path = os.path.join(base_dir, 'driver_imgs_list.csv')
img_path = os.path.join(base_dir, 'imgs')
train_path = os.path.join(base_dir, 'imgs', 'train')
# Read the CSV
df = pd.read_csv(csv_path)
# [VRFY] Run the following command to verify if the dataset has been correctly loaded
#print("LOADING DF..........")
#df.head()

## Adding a New Column in the Dataframe with the FUll pathname
df['path'] = df.apply(lambda row: os.path.join(img_path, 'train', row['classname'], row['img']), axis=1)
# [VRFY] Run the following command to verify if the dataset has been correctly upatd with Full Pathname
#print("LOADING DF with NEW Path column.......")
#df.head()

## Splitting TRAIN and TEST data
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['classname'], random_state=42)

## Saving DF files TO_CSV
train_df.to_csv(os.path.join(base_dir,"train_split.csv"), index=False)
val_df.to_csv(os.path.join(base_dir,"val_split.csv"), index=False)

# [VRFY] Run the following command to verify if the dataset has been correctly loaded
#print("LOADING TRAIN_DF and VAL_DF..........")
#train_df.head()
#val_df.head()

# Create ImageDataGenerator instances for training and validation
#train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Define Mage Size and Batch Size
image_size = (224, 224)
batch_size = 32

# Instantiate Image generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='classname',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    validate_filenames=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='classname',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# [VRFY] After creating train_df, val_df:
#for d in [train_df, val_df]:
#    print("Samples:", len(d),"\nClass counts:\n", d['classname'].value_counts(), "\n")

# After initializing generator:
print("Train img count:", train_generator.n)
print("Val img count:", val_generator.n)


# MODEL BUILDING
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# MODEL COMPLIATION
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# MODEL FITTING
model.fit(train_generator, epochs=2, validation_data=val_generator)

# MODEL EVALUATION
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')