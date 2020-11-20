import tensorflow as tf
import tensorflow.keras.regularizers
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
import kerastuner as kt
import time

# GPU Config
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

batch_size = 32
img_height = 48
img_width = 48

NAME = f"test-phase2-over-tuner-no-aug-reg-tune=reg-conv=2L-{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
#################################################################
#####                    LOAD DATA                        ######
################################################################
train_data = np.load("train_data.npy")
print(train_data.shape)
validation_data = np.load("validation_data.npy")
print(validation_data.shape)
train_labels = np.load("train_labels.npy")
validation_labels = np.load("validation_labels.npy")


#################################################################
#####                DATA AUGMENTATION                    ######
################################################################
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     # width_shift_range=0.2,
#     # height_shift_range=0.2,
#     # shear_range=0.2,
#     # zoom_range=0.2,
#     horizontal_flip=True
# )

#################################################################
#####             MODEL TUNING IMPLEMENTATION             ######
################################################################
def model_builder(hp):
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(img_width, img_height, 1),
                            padding='same',
                            bias_regularizer=tf.keras.regularizers.l1_l2(
                                l1=hp.Float("bias-l1-1-1", 0.0, 0.015, 0.003),
                                l2=hp.Float("bias-l2-1-1", 0.0, 0.015, 0.003)),
                            kernel_regularizer=tf.keras.regularizers.l1_l2(
                                l1=hp.Float("kernel-l1-1-1", 0.0, 0.015, 0.003),
                                l2=hp.Float("kernel-l2-1-1", 0.0, 0.015, 0.003))))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(img_width, img_height, 1),
                            padding='same',
                            bias_regularizer=tf.keras.regularizers.l1_l2(
                                l1=hp.Float("bias-l1-1-2", 0.0, 0.015, 0.003),
                                l2=hp.Float("bias-l2-1-2", 0.0, 0.015, 0.003)),
                            kernel_regularizer=tf.keras.regularizers.l1_l2(
                                l1=hp.Float("kernel-l1-1-2", 0.0, 0.015, 0.003),
                                l2=hp.Float("kernel-l2-1-2", 0.0, 0.015, 0.003))))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(img_width, img_height, 1),
                            padding='same',
                            bias_regularizer=tf.keras.regularizers.l1_l2(
                                l1=hp.Float("bias-l1-2-1", 0.0, 0.015, 0.003),
                                l2=hp.Float("bias-l2-2-1", 0.0, 0.015, 0.003)),
                            kernel_regularizer=tf.keras.regularizers.l1_l2(
                                l1=hp.Float("kernel-l1-2-1", 0.0, 0.015, 0.003),
                                l2=hp.Float("kernel-l2-2-1", 0.0, 0.015, 0.003))))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(img_width, img_height, 1),
                            padding='same',
                            bias_regularizer=tf.keras.regularizers.l1_l2(
                                l1=hp.Float("bias-l1-2-2", 0.0, 0.015, 0.003),
                                l2=hp.Float("bias-l2-2-2", 0.0, 0.015, 0.003)),
                            kernel_regularizer=tf.keras.regularizers.l1_l2(
                                l1=hp.Float("kernel-l1-2-2", 0.0, 0.015, 0.003),
                                l2=hp.Float("kernel-l2-2-2", 0.0, 0.015, 0.003))))
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())

    # for i in range(hp.Int("n_layers", 1, 2, step=1)):
    model.add(layers.Dense(160, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=7, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

# def model_builder(hp):
#     inputs = tf.keras.Input(shape=(48, 48, 1))
#     x = inputs
#     filters = 64
#     x = tf.keras.layers.Convolution2D(
#         filters, kernel_size=(3, 3), padding='same', activation='relu',
#         bias_regularizer=tf.keras.regularizers.l1_l2(
#             l1=hp.Float(f'bias-reg-l1-{i}-{j}', min_value=0.00, max_value=0.0, step=0.001),
#             l2=hp.Choice(f'bias-reg-l2-{i}-{j}', min_value=0.00, max_value=0.01, step=0.001)),
#         kernel_regularizer=tf.keras.regularizers.l1_l2(
#             l1=hp.Choice(f'kernel-reg-l1-{i}-{j}', values=[0.0, 0.005, 0.01, 0.015, 0.02]),
#             l2=hp.Choice(f'kernel-reg-l2-{i}-{j}', values=[0.0, 0.005, 0.01, 0.015, 0.02])))(x)
#
#     x = tf.keras.layers.MaxPool2D()(x)
#
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(
#         160,
#         activation='relu')(x)
#     # x = tf.keras.layers.Dropout(
#     #     hp.Float('dropout', 0.0, 0.5, step=0.1))(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
#
#     model = tf.keras.models.Model(inputs, outputs)
#     # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(
#             0.0001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy'])
#     return model

#################################################################
#####                 MODEL IMPLEMENTATION                ######
################################################################
model = tf.keras.models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu',
                  input_shape=(img_width, img_height, 1),
                  padding='same'),

    layers.Conv2D(64, (3, 3), activation='relu',
                  padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu',
                  input_shape=(img_width, img_height, 1),
                  padding='same'),

    layers.Conv2D(128, (3, 3), activation='relu',
                  padding='same'),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(160, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(units=7, activation='softmax')
])
#################################################################
#####                  MODEL COMPILATION                  ######
################################################################
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

hist = model.fit(train_data, train_labels, batch_size=32, epochs=30,
                 validation_data=(validation_data, validation_labels), verbose=2)
# hist = model.fit(datagen.flow(train_data, train_labels), epochs=30, validation_data=(validation_data, validation_labels), verbose=2, callbacks=[tensorboard])

#################################################################
#####                     TUNER SETUP                     ######
################################################################

# tuner = kt.Hyperband(model_builder,
#                      objective='val_accuracy',
#                      max_epochs=15,
#                      hyperband_iterations=2,
#                      directory='tuner_res',
#                      project_name=NAME)
#

# tuner = kt.RandomSearch(
#     model_builder,
#     objective='val_accuracy',
#     max_trials=40,
#     executions_per_trial=1,
#     directory='tuner_res',
#     project_name=NAME
# )
# # #
# tuner.search(train_data, train_labels,
#              validation_data=(validation_data, validation_labels),
#              batch_size=32,
#              epochs=10)
# #
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#
# optimal_model = tuner.hypermodel.build(best_hps)
# optimal_model.summary()
# # optimal_model = tf.keras.model6s.load_model("optimal_model-all-data")
# # optimal_model.save(NAME)
# optimal_model.fit(train_data, train_labels, epochs=30, batch_size=32,
#                   validation_data=(validation_data, validation_labels),
#                   verbose=2, callbacks=[tensorboard])
#
# for layer in optimal_model.layers:
#     print(layer.get_config())
