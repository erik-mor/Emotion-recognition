import pandas as pd
from keras.utils import np_utils
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from scipy.linalg import norm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

# TODO - pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

IMG_WIDTH = 48
IMG_HEIGHT = 48
PERCENTAGE = 100
selected_class = 2
selection_1 = [0, 4, 6]
selection_2 = [1, 3, 5]

print("Reading dataset from file...")
dataset = pd.read_csv("../../archive (1)/fer2013.csv")
print("Dataset read successfully.")

################################################################
####                    SAMPLING DATA                    ######
###############################################################
# print("Sampling data...")

# training_data = dataset.loc[dataset["Usage"] == "Training"]
# validation_data = dataset.loc[dataset["Usage"] == "PrivateTest"]
# print(training_data["emotion"].value_counts())
# print(validation_data["emotion"].value_counts())

################################################################
####                SAMPLE ALL DATA                      ######
###############################################################
sample_min = dataset.value_counts("emotion").min()
data = pd.DataFrame()
for key in range(7):
    data = pd.concat([data, dataset[dataset['emotion'] == key].sample(int(sample_min), replace=False)])

print(data["emotion"].value_counts())
# sample_max = validation_data.value_counts("emotion").median()
# validation_data_sampled = pd.DataFrame()
# for key, value in validation_data["emotion"].value_counts().items():
#     validation_data_sampled = pd.concat([validation_data_sampled, validation_data[validation_data["emotion"] == int(key)].sample(int(sample_max), replace=True)])
#
# print(training_data_sampled["emotion"].value_counts())
# print(validation_data_sampled["emotion"].value_counts())
#
################################################################
####                     ALPHA NN                        ######
###############################################################
# sample_selected = len(training_data[training_data["emotion"] == selected_class].index) * 2
# sample = sample_selected // 6
# data_sampled_alpha = pd.DataFrame()
# for key, value in training_data["emotion"].value_counts().items():
#     if int(key) == selected_class:
#         data_sampled_alpha = pd.concat([data_sampled_alpha,
#                                         training_data[training_data["emotion"] == selected_class].sample(
#                                             sample_selected, replace=True)])
#     else:
#         data_sampled_alpha = pd.concat(
#             [data_sampled_alpha,
#              training_data[training_data["emotion"] == int(key)].sample(sample, replace=True)])
#
# data_sampled_alpha.loc[data_sampled_alpha["emotion"] != selected_class, "emotion"] = 1
# data_sampled_alpha.loc[data_sampled_alpha["emotion"] == selected_class, "emotion"] = 0
#
# validation_data_sampled_alpha = validation_data.copy()
# validation_data_sampled_alpha.loc[validation_data_sampled_alpha["emotion"] != selected_class, "emotion"] = 1
# validation_data_sampled_alpha.loc[validation_data_sampled_alpha["emotion"] == selected_class, "emotion"] = 0
#
# print(data_sampled_alpha["emotion"].value_counts())
# print(validation_data_sampled_alpha["emotion"].value_counts())

################################################################
####                     BETA NN                        ######
###############################################################
# sample = int(training_data.value_counts("emotion").median())
# data_sampled_beta = pd.DataFrame()
# for key, value in training_data["emotion"].value_counts().items():
#     if int(key) != selected_class:
#         data_sampled_beta = pd.concat([data_sampled_beta,
#                                        training_data[training_data["emotion"] == int(key)].sample(
#                                            sample, replace=True)])
#
# data_sampled_beta.loc[data_sampled_beta["emotion"].isin(selection_1), "emotion"] = 0
# data_sampled_beta.loc[data_sampled_beta["emotion"].isin(selection_2), "emotion"] = 1
#
# validation_data_sampled_beta = validation_data.copy()
# validation_data_sampled_beta = validation_data_sampled_beta.loc[validation_data_sampled_beta["emotion"] != selected_class]
# validation_data_sampled_beta.loc[validation_data_sampled_beta["emotion"].isin(selection_1), "emotion"] = 0
# validation_data_sampled_beta.loc[validation_data_sampled_beta["emotion"].isin(selection_2), "emotion"] = 1
#
# print(data_sampled_beta["emotion"].value_counts())
# print(validation_data_sampled_beta["emotion"].value_counts())

################################################################
####                     GAMA NN                        ######
###############################################################
# def get_data_for_selection(data, selection, validation=False):
#     data = data[data["emotion"].isin(selection)]
#     for i in selection:
#         data.loc[data['emotion'] == i, 'emotion'] = selection.index(i)
#
#     if not validation:
#         sample_size = int(data.value_counts("emotion").max() * 2)
#         print(sample_size, selection)
#         data_sampled = pd.DataFrame()
#         for key, value in data["emotion"].value_counts().items():
#             data_sampled = pd.concat(
#                 [data_sampled, data[data["emotion"] == int(key)].sample(sample_size, replace=True)])
#
#         return data_sampled
#
#     return data
#
#
# def sample_per_selection(data, validation=False):
#     X_1 = get_data_for_selection(data, selection_1, validation)
#     X_2 = get_data_for_selection(data, selection_2, validation)
#
#     return X_1, X_2
#
# training_data_sampled_gama_1, training_data_sampled_gama_2 = sample_per_selection(training_data)
# validation_data_sampled_gama_1, validation_data_sampled_gama_2 = sample_per_selection(validation_data, validation=True)

# print(training_data_sampled["emotion"].value_counts())
# print(validation_data["emotion"].value_counts())

################################################################
####                     TEST DATA                       ######
###############################################################
# testing_data = dataset.loc[dataset["Usage"] == "PublicTest"]
# print(testing_data["emotion"].value_counts())
#
# testing_data_alpha = testing_data.copy()
# testing_data_alpha.loc[testing_data_alpha["emotion"] != selected_class, "emotion"] = 1
# testing_data_alpha.loc[testing_data_alpha["emotion"] == selected_class, "emotion"] = 0
#
# testing_data_beta = testing_data.copy()
# testing_data_beta = testing_data_beta.loc[testing_data_beta["emotion"] != selected_class]
# testing_data_beta.loc[testing_data_beta["emotion"].isin(selection_1), "emotion"] = 0
# testing_data_beta.loc[testing_data_beta["emotion"].isin(selection_2), "emotion"] = 1
#
# testing_data_gama_1 = get_data_for_selection(testing_data.copy(), selection_1, True)
# testing_data_gama_2 = get_data_for_selection(testing_data.copy(), selection_2, True)
#
# print(testing_data_alpha["emotion"].value_counts())
# print(testing_data_beta["emotion"].value_counts())
# print(testing_data_gama_1["emotion"].value_counts())
# print(testing_data_gama_2["emotion"].value_counts())

# sample = int(testing_data.value_counts("emotion").median() / 2)
# testing_data_sampled = pd.DataFrame()
# for key, value in testing_data["emotion"].value_counts().items():
#     testing_data_sampled = pd.concat([testing_data_sampled, testing_data[testing_data["emotion"] == int(key)].sample(int(sample), replace=True)])
#
# print(testing_data_sampled["emotion"].value_counts())
######     SMOTE      ##############
# x = training_data.drop(["Usage", "emotion"], 1)
# y = training_data.drop(["Usage", "pixels"], 1)
# print(y.value_counts())
# print(x.head())
# smote = SMOTE(sampling_strategy="minority")
# x_sm, y_sm = smote.fit_sample(x, y)
# print(y_sm.value_counts())

print("Data sampled successfully.")

# ################################################################
# ####                   SHRINKING DATASET                 ######
# ###############################################################
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
#
#
# print("Data shrank successfully.")
# print(training_data_sampled['emotion'].value_counts())
# print(validation_data_sampled['emotion'].value_counts())
# print(testing_data_sampled['emotion'].value_counts())

# #################################################################
# #####                     CLUSTERING                      ######
# ################################################################
classes_fer = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
classes_ck = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Contempt"]

# arr = []
# for i in range(7):
#     gen_data = data[data["emotion"] == i]
#     gen_data = gen_data["pixels"].str.split(" ").tolist()
#     gen_data = np.uint8(gen_data)
#     gen_data = gen_data.reshape((len(gen_data), 48, 48, 1))
#     arr.append(gen_data)
# #
# np.savez("clusters.npz", anger=arr[0], disgust=arr[1], fear=arr[2], happy=arr[3], sad=arr[4], surprise=arr[5], neutral=arr[6])

face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
# data = np.load("clusters.npz")
#
# for image in data['disgust']:
#     faces = face_cascade.detectMultiScale(image, 1.02, 5)
#     print(faces)
#     # masked = np.copy(image)
#     # for (x, y, w, h) in faces:
#     #     masked[:y, :] = 255
#     #     masked[:, :x] = 255
#     #     masked[y + h + 1:, :] = 255
#     #     masked[:, x + w + 1:] = 255
#
#     mask = np.zeros(image.shape, np.uint8)
#     for (x, y, w, h) in faces:
#         mask[y:y + h, x:x + w] = 255
#
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(image, cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(122)
#     plt.imshow(mask, 'gray')
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()

compare_method = 2


def crop(image, face):
    masked = np.copy(image)
    for (x, y, w, h) in face:
        masked[:y, :] = 255
        masked[:, :x] = 255
        masked[y + h + 1:, :] = 255
        masked[:, x + w + 1:] = 255

    return masked


def get_mask(shape, face):
    mask = np.zeros(shape, np.uint8)
    for (x, y, w, h) in face:
        mask[y:y + h, x:x + w] = 255
    return mask


def hist_norm(image, face):
    if len(face) != 0:
        mask = get_mask(image.shape, face)
    else:
        mask = None

    h = cv2.calcHist([image], [0], mask, [256], [0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h


def diff(img1, img2):
    d = img1 - img2
    n_0 = norm(d.ravel(), 0)
    return n_0


def find_centers(find_face=True):
    data = np.load("../clusters/clusters.npz")
    centers_arr = []
    centers_arr_px = []

    for label in data:
        gen_data = data[label]
        cropped_arr = []

        if find_face:
            faces_arr = [face_cascade.detectMultiScale(i, 1.02, 5) for i in gen_data]

            hist_array = []
            for image, face in zip(gen_data, faces_arr):
                cropped_arr.append(crop(image, face))
                hist_array.append(hist_norm(image, face))
        else:
            hist_array = [hist_norm(img, ()) for img in gen_data]

        matrix_of_hist = np.zeros((len(gen_data), len(gen_data)))
        matrix_of_diff = np.zeros((len(gen_data), len(gen_data)))
        for i in range(0, len(gen_data)):
            for j in range(i + 1, len(gen_data)):
                h = cv2.compareHist(hist_array[i], hist_array[j], compare_method)
                if find_face:
                    d = diff(cropped_arr[i], cropped_arr[j])
                else:
                    d = diff(gen_data[i], gen_data[j])

                matrix_of_hist[i][j] = h
                matrix_of_hist[j][i] = h
                matrix_of_diff[i][j] = d
                matrix_of_diff[j][i] = d

        diff_center = int(np.argmin(np.sum(matrix_of_diff, axis=1)))
        if compare_method == 1 or compare_method == 3:
            hist_center = int(np.argmin(np.sum(matrix_of_hist, axis=1)))
        else:
            hist_center = int(np.argmax(np.sum(matrix_of_hist, axis=1)))

        centers_arr.append(gen_data[hist_center])
        centers_arr_px.append(gen_data[diff_center])

        plt.figure(figsize=(4.5, 2.5))
        plt.subplot(121)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(gen_data[hist_center], cmap='gray')
        plt.subplot(122)
        plt.xlim([0, 256])
        plt.plot(hist_array[hist_center])
        # plt.subplot(223)
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(gen_data[diff_center], cmap='gray')
        # plt.subplot(224)
        # plt.xlim([0, 256])
        # plt.plot(hist_array[diff_center])
        plt.tight_layout()
        plt.show()

    np.savez("../clusters/cluster_center.npz", anger=centers_arr[0], disgust=centers_arr[1], fear=centers_arr[2],
             happy=centers_arr[3], sad=centers_arr[4], surprise=centers_arr[5], neutral=centers_arr[6])

    np.savez("../clusters/cluster_center_px-dff.npz", anger=centers_arr_px[0], disgust=centers_arr_px[1], fear=centers_arr_px[2],
             happy=centers_arr_px[3], sad=centers_arr_px[4], surprise=centers_arr_px[5], neutral=centers_arr_px[6])


def get_centers(find_faces=True):
    centers_px = np.load("../clusters/cluster_center_px-diff.npz")
    centers_hist = np.load("../clusters/cluster_center.npz")

    centers_arr = []
    centers_arr_hist = []
    for x, y in zip(centers_px, centers_hist):
        centers_arr.append(centers_px[x])
        centers_arr_hist.append(centers_hist[y])

    if find_faces:
        faces_arr_px = [face_cascade.detectMultiScale(i, 1.02, 5) for i in centers_arr]
        faces_arr_hist = [face_cascade.detectMultiScale(i, 1.02, 5) for i in centers_arr_hist]

        cropped_arr = []
        hist_array = []
        for i, (image_px, image_hist) in enumerate(zip(centers_arr, centers_arr_hist)):
            cropped_arr.append(crop(image_px, faces_arr_px[i]))
            hist_array.append(hist_norm(image_hist, faces_arr_hist[i]))

        cropped_arr = np.array(cropped_arr)
        hist_array = np.array(hist_array)
        return cropped_arr, hist_array

    else:
        hist_array = [hist_norm(image, ()) for image in centers_arr_hist]
        return np.array(centers_arr), np.array(hist_array)


# def show_centers(centers_img, centers_hist):
#     for i, (hist, image) in enumerate(zip(centers_hist, centers_img)):
#         plt.figure()
#         plt.suptitle(classes_fer[i])
#         plt.subplot(121)
#         plt.xticks([])
#         plt.yticks([])
#         plt.imshow(image, cmap='gray')
#         plt.subplot(122)
#         plt.xlim([0, 256])
#         plt.plot(hist)
#         plt.show()


# #################################################################
# #####                  CKPLUS DATASET                      ######
# ################################################################
def plot_cm(y_true, y_pred, figsize=(10, 10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    print(cm)
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

    cm = pd.DataFrame(cm, index=classes_ck, columns=classes_fer)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sn.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax, vmax=100, vmin=0)


def calc_label_diff(image, centers, faces, find_faces=True):
    i = np.copy(image)
    if find_faces:
        i = crop(image, faces)
    diff_hist = [diff(i, x) for x in centers]
    return np.argmin(diff_hist)


def calc_label_hist(image, centers_hist, faces):
    hist = hist_norm(image, faces)
    diff_hist = [cv2.compareHist(hist, x, compare_method) for x in centers_hist]
    if compare_method == 1 or compare_method == 3:
        return np.argmin(diff_hist)
    else:
        return np.argmax(diff_hist)


def process_ck(find_faces=True):
    data_path = '../../CK/CK+48/'
    data_dir_list = os.listdir(data_path)
    img_data_list = []
    labels = []
    labels_diff = []
    labels_hist = []

    # centers_array, centers_array_histogram = get_centers(find_faces)

    for index, dataset in enumerate(sorted(data_dir_list)):
        img_list = os.listdir(data_path + '/' + dataset)
        # img_list_new = [img_list[i] for i in range(0, len(img_list), 3)]
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        for img in img_list:
            input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize = cv2.resize(input_img, (48, 48))
            input_img_resize = np.reshape(input_img_resize, (48, 48, 1))
            img_data_list.append(input_img_resize)
            if index == 6:
                labels.append(3)
            else:
                labels.append(index)

            # if index == 6:
            #     faces = face_cascade.detectMultiScale(input_img_resize, 1.02, 5) if find_faces else ()
            #     label_diff = calc_label_diff(input_img_resize, centers_array, faces, find_faces)
            #     labels_diff.append(label_diff)
            #
            #     label_hist = calc_label_hist(input_img_resize, centers_array_histogram, faces)
            #     labels_hist.append(label_hist)
            # else:
            #     labels_diff.append(index)
            #     labels_hist.append(index)

            # plt.figure()
            # plt.subplot(121)
            # plt.xticks([])
            # plt.yticks([])
            # plt.imshow(input_img_resize, cmap='gray')
            # plt.subplot(122)
            # plt.xlim([0, 256])
            # plt.plot(hist_norm(input_img_resize, ()))
            # plt.show()

    labels = np.array(labels)
    # labels_diff = np.array(labels_diff)
    # labels_hist = np.array(labels_hist)

    # plot_cm(labels, labels_diff)
    # print(classification_report(labels, labels_diff))
    # plt.show()
    # plot_cm(labels, labels_hist)
    # print(classification_report(labels, labels_hist))
    # plt.show()

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32') / 255
    print(img_data.shape)
    # np.save(f"labels_ck-no_face-static-pixel_diff", np_utils.to_categorical(labels_diff))
    # np.save(f"labels_ck-no_face-static-hist_diff", np_utils.to_categorical(labels_hist))
    np.save(f"../image_sets/data/data_ck_new.npy", img_data)
    np.save(f"../image_sets/labels/labels_ck_new.npy", np_utils.to_categorical(labels))


# find_centers(False)
process_ck(False)
# #################################################################
# #####                  GENERATE DATA                      ######
# ################################################################
# def generate_data(data, name):
#     gen_data = data["pixels"].str.split(" ").tolist()
#     gen_data = np.uint8(gen_data)
#     gen_data = gen_data.reshape((len(data), 48, 48, 1))
#     gen_data = gen_data.astype("float32") / 255
#     print(gen_data.shape)
#     np.save(f"labels_{name}", np_utils.to_categorical(data["emotion"]))
#     np.save(f"data_{name}", gen_data)


# generate_data(data_sampled_alpha, "training_alpha")
# generate_data(validation_data_sampled_alpha, "validation_alpha")
# generate_data(data_sampled_beta, "training_beta")
# generate_data(validation_data_sampled_beta, "validation_beta")
# generate_data(training_data_sampled_gama_2, "training_gama_2")
# generate_data(validation_data_sampled_gama_2, "validation_gama_2")

# generate_data(training_data_sampled, "sampled_training_all")
# generate_data(validation_data_sampled, "sampled_validation_all")

# generate_data(testing_data_sampled, "sampled_test")

# generate_data(testing_data_alpha, "test_alpha")
# generate_data(testing_data_beta, "test_beta")
# generate_data(testing_data_gama_1, "test_gama_1")
# generate_data(testing_data_gama_2, "test_gama_2")

print("Data generated successfully.")
#
# ##################################################################
# #####                    SAVE DATA                         ######
# ################################################################
# print("Saving data...")
# #
# np.save("./train_data.npy", train_pixels)
# np.save("./validation_data.npy", validation_pixels)
# # np.save("./test_data.npy", test_pixels)
# np.save("./train_labels.npy", train_labels)
# np.save("./validation_labels.npy", validation_labels)
# # np.save("./test_labels.npy", testing_labels)
# #
# print("Data saved successfully.")
