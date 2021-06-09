import tensorflow as tf
import math
import seaborn as sns
import tensorflow.keras.regularizers
import numpy as np
import os
import cv2
import pandas as pd
import seaborn as sn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
import kerastuner as kt
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
import time
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# GPU Config
# config_cpu = tf.compat.v1.ConfigProto(
#     device_count={'GPU': 0}
# )

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

nn = "all"
NAME = f"{nn}-old-no_reg-aug-2conv-{int(time.time())}"

# callbacks_logger = tf.keras.callbacks.CSVLogger(f"training_logs/{NAME}", append=True)
callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-7,
                                                          verbose=1)
callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                       patience=7,
                                                       verbose=1,
                                                       restore_best_weights=True,
                                                       )


def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.9
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


callback_schedule_lr = tf.keras.callbacks.LearningRateScheduler(step_decay)

batch_size = 32
img_height = 48
img_width = 48

selected_class = 2
selection_1 = [0, 4, 6]
selection_2 = [1, 3, 5]

tensorboard = TensorBoard(log_dir=f"../logs/{NAME}")
#################################################################
#####                    LOAD DATA                        ######
################################################################
# train_data = np.load(f"../image_sets/data/train_data.npy")
# train_data = np.load(f"data_sampled_training_{nn}.npy")
# print(train_data.shape)
# validation_data = np.load(f"../image_sets/data/validation_data.npy")
# validation_data = np.load(f"data_sampled_validation_{nn}.npy")
# print(validation_data.shape)
test_data = np.load(f"../image_sets/data/data_test.npy")
test_labels = np.load(f"../image_sets/labels/labels_test.npy")
# print(test_data.shape)
# train_labels = np.load(f"../image_sets/labels/train_labels.npy")
# validation_labels = np.load(f"../image_sets/labels/validation_labels.npy")
# train_labels = np.load(f"labels_sampled_training_{nn}.npy")
# validation_labels = np.load(f"labels_sampled_validation_{nn}.npy")

ck_data = np.load("../image_sets/data/data_ck.npy")
ck_labels = np.load("../image_sets/labels/labels_ck.npy")

print(ck_data.shape)

#################################################################
#####                DATA AUGMENTATION                    ######
################################################################
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     # shear_range=0.1,
#     # zoom_range=0.1,
#     horizontal_flip=True
# )
# datagen.fit(train_data)

#
# x = datagen.flow(train_data, train_labels)
# x_aug, y_aug = x.next()
#
# print(len(y_aug))

#################################################################
#####             MODEL TUNING IMPLEMENTATION             ######
################################################################
# def model_builder(hp):
#     model = tf.keras.models.Sequential()
#     model.add(layers.Input(shape=(48, 48, 1)))
#     # for i in range(3):
#     # filters = 64 if i == 0 else 128
#     # for j in range(2):
#     #     model.add(layers.Conv2D(filters, (3, 3),
#     #                             padding='same',
#     #                             kernel_regularizer=tf.keras.regularizers.l1_l2(
#     #                                 l1=hp.Float(f"kernel-l1-{i}-{j}", 0.0, 0.01, 0.002),
#     #                                 l2=hp.Float(f"kernel-l2-{i}-{j}", 0.0, 0.01, 0.002))))
#     #     model.add(tf.keras.layers.Activation('relu'))
#     #     model.add(layers.BatchNormalization())
#     #
#     # model.add(layers.MaxPooling2D())
#     # model.add(layers.Dropout(hp.Float(f"dropout-{i}", 0.0, 0.5, 0.1)))
#
#     model.add(layers.Conv2D(64, (3, 3), padding='same'))
#     model.add(tf.keras.layers.Activation('relu'))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(layers.MaxPooling2D())
#
#     model.add(layers.Conv2D(128, (3, 3), padding='same'))
#     model.add(tf.keras.layers.Activation('relu'))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(layers.MaxPooling2D())
#
#     model.add(layers.Flatten())
#
#     # for i in range(hp.Int("n_layers", 1, 2, step=1)):
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.BatchNormalization())
#     # model.add(layers.Dropout(0.2))
#
#     model.add(layers.Dense(units=2, activation='softmax'))
#
#     model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-7, 1e-4, sampling='log')),
#                   loss=tf.keras.losses.CategoricalCrossentropy(),
#                   metrics=['accuracy'])
#
#     return model

#################################################################
#####                 MODEL IMPLEMENTATION                ######
################################################################
# model = tf.keras.models.Sequential()
# model.add(layers.Input(shape=(48, 48, 1)))
#
# model.add(layers.Conv2D(64, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.2))
#
# model.add(layers.Conv2D(128, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.3))
#
# model.add(layers.Conv2D(256, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Conv2D(256, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Conv2D(256, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.4))
#
# model.add(layers.Conv2D(256, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Conv2D(256, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Conv2D(256, (3, 3), padding='same'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.4))
#
# model.add(layers.Flatten())
#
# model.add(layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Dropout(0.5))
#
# model.add(layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(layers.Dropout(0.5))
#
# model.add(layers.Dense(units=7, activation='softmax'))

#################################################################
#####                  MODEL COMPILATION                  ######
################################################################
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#               loss=tf.keras.losses.CategoricalCrossentropy(),
#               metrics=['accuracy'])
# model.summary()
#
# hist = model.fit(datagen.flow(train_data, train_labels), batch_size=batch_size, epochs=30, steps_per_epoch=len(train_data) / batch_size,
#                  validation_data=(validation_data, validation_labels),
#                  callbacks=[])
#
# model.save("model_" + NAME)
#
# #
# sns.set()
# fig = pyplot.figure()
# sns.lineplot(hist.epoch, hist.history['accuracy'], label='training')
# sns.lineplot(hist.epoch, hist.history['val_accuracy'], label='validation')
# pyplot.xlabel("Epoch")
# pyplot.ylabel("Accuracy")
#
# pyplot.savefig(NAME)
# pyplot.show()


# print(test_labels[2])
#
alpha = tf.keras.models.load_model("../saved_models/model_alpha--1614719512")
beta = tf.keras.models.load_model("../saved_models/model_beta--1614582905")
gama1 = tf.keras.models.load_model("../saved_models/model_gama_1--1614837580")
gama2 = tf.keras.models.load_model("../saved_models/model_gama_2--1614875102")

model = tf.keras.models.load_model("../saved_models/model_all--1613902058")

########### DT testing ################
# labels = ck_labels.copy()
# data = ck_data.copy()
# Y_pred = []
# for index in range(len(data)):
#     # plt.imshow(test_data[index], cmap='gray')
#     # plt.show()
#     # print(f"Label: {np.argmax(test_labels[index])}")
#
#     label = np.argmax(labels[index])
#
#     image = data[index: index + 1]
#     pred = alpha.predict(image)
#     y_pred = np.argmax(pred, axis=1)
#
#     if y_pred[0] == 0:
#         prediction = 2
#     else:
#         pred = beta.predict(image)
#         y_pred = np.argmax(pred, axis=1)
#         if y_pred[0] == 0:
#             pred = gama1.predict(image)
#             y_pred = np.argmax(pred, axis=1)
#             prediction = selection_1[y_pred[0]]
#         else:
#             pred = gama2.predict(image)
#             y_pred = np.argmax(pred, axis=1)
#             prediction = selection_2[y_pred[0]]
#
#     Y_pred.append(prediction)


def plot_cm(y_true, y_pred, figsize=(10, 10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    print(cm)
    # cm = np.zeros((6, 7), dtype=float)
    # for true, pred in zip(y_true, y_pred):
    #     cm[true][pred] = cm[true][pred] + 1

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
            cm[i, j] = p

    sn.set(font_scale=1.4)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_pred))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = pyplot.subplots(figsize=figsize)
    sn.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax, vmax=100, vmin=0)


# Y_test = np.argmax(test_labels, axis=1)
# y_pred = model.predict(test_data)
# Y_pred = np.argmax(y_pred, axis=1)
# #
# plot_cm(Y_test, Y_pred)
# pyplot.show()
# print(classification_report(Y_test, Y_pred))

# print("Original labels")
# print(ck_data.shape)
# print(ck_labels.shape)
Y_test = np.argmax(test_labels, axis=1)
y_pred = model.predict(test_data)
Y_pred = np.argmax(y_pred, axis=1)
# np.save("Y_pred", Y_pred)
#
plot_cm(Y_test, Y_pred)
pyplot.show()
# print(classification_report(Y_test, Y_pred))


# path = "../clusters/labels/"
# labels_dir = os.listdir(path)
#
# for labels in labels_dir:
#     print(labels)
#
#     labels = np.load(path + labels)
#     print(labels)
#     Y_test = np.argmax(labels, axis=1)
#     y_pred = model.predict(ck_data)
#     Y_pred = np.argmax(y_pred, axis=1)
#
#     plot_cm(Y_test, Y_pred)
#     pyplot.show()
#     print(classification_report(Y_test, Y_pred))

#################################################################
#####                TRANSFER LEARNING                    ######
################################################################
# model = tf.keras.models.load_model("model_all--1613902058")
#
# model.summary()
# model.pop()
# model.pop()
# model.pop()
# model.pop()
# model.summary()

# model.trainable = False
# finetune_model = tf.keras.models.Sequential([
#     model,
    # tf.keras.layers.Dense(units=128, activation='relu', name="new_dense"),
    # tf.keras.layers.BatchNormalization(name="new_batch"),
    # layers.Dropout(0.3, name="new_dropout"),
    # tf.keras.layers.Dense(units=6, activation='softmax', name="new_output")
# ])
# finetune_model.build(input_shape=(None, 48, 48, 1))
#
# model.pop()
# model.pop()
# model.pop()
# model.pop()
# model.pop()
# model.trainable = False
# new_layer = tf.keras.layers.Dense(units=7, activation='softmax', name="new_output")

# x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name="conv_last_0_0")(model.layers[-1].output)
# x = tf.keras.layers.BatchNormalization(name=f"bn_conv_0_0")(x)
# for i in range(2):
#     for j in range(1):
#         if i != 0 and j != 0:
#             x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name=f"conv_last_{i}_{j}")(x)
#             x = tf.keras.layers.BatchNormalization(name=f"bn_conv_{i}_{j}")(x)
#
#     x = layers.MaxPooling2D(name=f"pooling_conv_{i}")(x)
#     x = layers.Dropout(0.3, name=f"dropout_conv{i}")(x)


# x = layers.Flatten()(x)
# x = tf.keras.layers.Dense(units=256, activation='relu', name="last_dense")(x)
# x = tf.keras.layers.BatchNormalization(name="last_bn")(x)
# x = layers.Dropout(0.3, name="last_drop")(x)

# inp = model.input
# out = new_layer(x)
# finetune_model = tf.keras.models.Model(inp, out)

# old_weights = finetune_model.get_weights()
# X_train, X_test, y_train, y_test = train_test_split(ck_data, ck_labels, test_size=0.2)
# finetune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                        loss=tf.keras.losses.CategoricalCrossentropy(),
#                        metrics=['accuracy'])
# finetune_model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[callback_early_stop, callback_reduce_lr])
# print(finetune_model.summary())

#
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# for x in (5, 8, 10):
#     cv = KFold(n_splits=x, shuffle=True)
#     accuracy_arr = []
#     for train, test in cv.split(ck_data, ck_labels):
#         finetune_model.set_weights(old_weights)
#         finetune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                                loss=tf.keras.losses.CategoricalCrossentropy(),
#                                metrics=['accuracy'])
#         print(finetune_model.summary())
#         finetune_model.fit(ck_data[train], ck_labels[train], batch_size=32, epochs=50, callbacks=[callback_early_stop, callback_reduce_lr])
#
#         Y_test = np.argmax(ck_labels[test], axis=1)
#         y_pred = finetune_model.predict(ck_data[test])
#         Y_pred = np.argmax(y_pred, axis=1)
#
#         accuracy_arr.append(accuracy_score(Y_test, Y_pred))
#
#     print(f"Mean accuracy {x}-fold : ", np.mean(accuracy_arr))

# hist = finetune_model.fit(X_train, y_train, batch_size=batch_size, epochs=30,
#                           # validation_data=(X_val, y_val),
#                           callbacks=[callback_early_stop, callback_reduce_lr])

# finetune_model.trainable = True
# print(finetune_model.summary())
#
# hist2 = finetune_model.fit(X_train, y_train, batch_size=batch_size, epochs=100,
#                            validation_data=(X_val, y_val),
#                            callbacks=[callback_early_stop, callback_reduce_lr])

# finetune_model.save("../saved_models/model_transfer_ck")
#

# Y_test = np.argmax(y_test, axis=1)
# y_pred = finetune_model.predict(X_test)
# Y_pred = np.argmax(y_pred, axis=1)
#
# plot_cm(Y_test, Y_pred)
# pyplot.show()
# print(classification_report(Y_test, Y_pred))


#################################################################
#####                     TUNER SETUP                     ######
################################################################

# tuner = kt.Hyperband(model_builder,
#                      objective='val_accuracy',
#                      hyperband_iterations=2,
#                      directory='tuner_res',
#                      project_name=NAME)

# tuner = kt.RandomSearch(
#     model_builder,
#     objective='val_accuracy',
#     max_trials=10,
#     executions_per_trial=2,
#     directory='tuner_res',
#     project_name=NAME
# )
# # # # # # # #
# tuner.search(datagen.flow(train_data, train_labels),
#              validation_data=(validation_data, validation_labels),
#              batch_size=32,
#              steps_per_epoch=len(train_data) / 32,
#              epochs=100, callbacks=[callback_early_stop, callback_reduce_lr])
# #
# #
# # best_hps = tuner.get_best_hyperparameters(num_trials=10)
# # # for hps in best_hps:
# # #     print(f"""
# # #     The hyperparameter search is complete. The optimal number of units in the first convolutional
# # #     layer is {hps.get('filters-0')} and {hps.get('filters-1')} in second. The optimal number of dense units
# # #     is {hps.get('n_dense_units')} and optimal dropout is {hps.get('dropout')}
# # #     """)
# #
# optimal_model = tuner.get_best_models(num_models=1)[0]
# # optimal_model = tf.keras.models.load_model("alpha-tuning-without_aug-batch_norm-1612388183")
#
# optimal_model.save(NAME)
# optimal_model.fit(datagen.flow(train_data, train_labels), epochs=30, batch_size=32,
#                   validation_data=(validation_data, validation_labels),
#                   verbose=2)
#
# for layer in optimal_model.layers:
#     print(layer.get_config())
