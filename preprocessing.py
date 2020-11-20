import pandas as pd
from keras.utils import np_utils
import numpy as np
# TODO - pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

IMG_WIDTH = 48
IMG_HEIGHT = 48
PERCENTAGE = 100
print("Reading dataset from file...")

dataset = pd.read_csv("../archive (1)/fer2013.csv")

print("Dataset read successfully.")
################################################################
####                    SAMPLING DATA                    ######
###############################################################
print("Sampling data...")
# sample_max = dataset.value_counts("emotion").min()
# data = pd.DataFrame()
# for key, value in dataset["emotion"].value_counts().items():
#     data = pd.concat([data, dataset[dataset["emotion"] == int(key)].sample(int(sample_max), replace=True)])

training_data = dataset.loc[dataset["Usage"] != "PrivateTest"]
sample = training_data.value_counts("emotion").min()
training_data_sampled = pd.DataFrame()
for key, value in dataset["emotion"].value_counts().items():
    training_data_sampled = pd.concat([training_data_sampled, training_data[training_data["emotion"] == int(key)].sample(int(sample), replace=True)])

validation_data = dataset.loc[dataset["Usage"] == "PrivateTest"]
sample = validation_data.value_counts("emotion").min()
validation_data_sampled = pd.DataFrame()
for key, value in dataset["emotion"].value_counts().items():
    validation_data_sampled = pd.concat([validation_data_sampled, training_data[training_data["emotion"] == int(key)].sample(int(sample), replace=True)])

# smote = SMOTE(sampling_strategy="minority")
# smote.fit()

print("Data sampled successfully.")
################################################################
####                   SHRINKING DATASET                 ######
###############################################################
# print(f"Shrinking dataset to {PERCENTAGE}%...")
# training_data = pd.DataFrame()
# validation_data = pd.DataFrame()
# training_data_all = dataset.loc[dataset['Usage'] != 'PrivateTest']
# for key, value in training_data_all["emotion"].value_counts().items():
#     end = ceil(value * PERCENTAGE / 100)
#     training_data = pd.concat([training_data, training_data_all[training_data_all['emotion'] == int(key)][:end]])
#
# validation_data_all = dataset.loc[dataset['Usage'] == 'PrivateTest']
# for key, value in validation_data_all["emotion"].value_counts().items():
#     end = ceil(value * PERCENTAGE / 100)
#     validation_data = pd.concat([validation_data, validation_data_all[validation_data_all['emotion'] == int(key)][:end]])

VALIDATION_DATA_COUNT = len(validation_data_sampled)
TRAINING_DATA_COUNT = len(training_data_sampled)
#
# print("Data shrank successfully.")
print(training_data_sampled['emotion'].value_counts())
print(validation_data_sampled['emotion'].value_counts())

################################################################
####                  GENERATE LABELS                    ######
###############################################################
print("Generating data...")

train_labels = np_utils.to_categorical(training_data_sampled['emotion'])
validation_labels = np_utils.to_categorical(validation_data_sampled['emotion'])

#################################################################
#####              GENERATE TRAINING DATA                 ######
################################################################
# d = int(len(training_data) / 10)
# train_pixels = []
# for i in range(0, 10):
#     if i == 9:
#         temp = training_data["pixels"].str.split(" ").tolist()[i*d:]
#     else:
#         temp = training_data["pixels"].str.split(" ").tolist()[i*d: (i+1) * d]
#     train_pixels.extend(temp)

train_pixels = training_data_sampled["pixels"].str.split(" ").tolist()
train_pixels = np.uint8(train_pixels)
train_pixels = train_pixels.reshape((TRAINING_DATA_COUNT, 48, 48, 1))
train_pixels = train_pixels.astype("float32") / 255

#################################################################
#####              GENERATE VALIDATION DATA               ######
################################################################
validation_pixels = validation_data_sampled["pixels"].str.split(" ").tolist()
validation_pixels = np.uint8(validation_pixels)
validation_pixels = validation_pixels.reshape((VALIDATION_DATA_COUNT, 48, 48, 1))
validation_pixels = validation_pixels.astype("float32") / 255

print("Data generated successfully.")

##################################################################
#####                    SAVE DATA                         ######
################################################################
print("Saving data...")

np.save("./train_data.npy", train_pixels)
np.save("./validation_data.npy", validation_pixels)
np.save("./train_labels.npy", train_labels)
np.save("./validation_labels.npy", validation_labels)

print("Data saved successfully.")
