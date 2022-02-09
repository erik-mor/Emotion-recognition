import os
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
from sklearn.metrics import classification_report
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score

# string = """
#            0       0.55      0.64      0.59       467
#            1       0.72      0.70      0.71        56
#            2       0.52      0.45      0.48       496
#            3       0.90      0.83      0.86       895
#            4       0.58      0.51      0.54       653
#            5       0.76      0.81      0.79       415
#            6       0.57      0.67      0.62       607
#
# Average       0.66      0.66      0.66      3589
# WeightedAverage      0.67      0.66      0.66      3589
# """
#
# splitted = string.split()
# sep = "\t&\t"
# for x in range(0, len(splitted), 5):
#
#     print(sep.join(splitted[x:x+5]) + "\t \\\\")

# matrix = [
#     [106, 3, 28, 39, 31, 12, 29],
#     [47, 147, 21, 13, 0, 11, 9],
#     [53, 4, 67, 25, 39, 29 , 31],
#     [17, 2, 18, 166, 17, 17, 11],
#     [66, 8, 24, 22, 64, 16, 48],
#     [33, 1, 30, 12, 11, 158, 3],
#     [44, 4, 29, 22, 40, 7, 102]
# ]
#
# matrix2 = [[130, 1,  12,  18,  15,  15,  57],
#  [ 96,  53,  25,  31,  18,   0,  25],
#  [ 29,   0,  55,  13,  56,  33,  62],
#  [  6,   1,   2, 215,   2,   9,  13],
#  [ 33,   0,  19,  29,  67,   5,  95],
#  [ 15,   2,  20,  16,   0, 178,  17],
#  [ 20,   0,  16,  24,  14,   4, 170]]
#
# matrix2 = np.array(matrix)
#
# precision = np.diagonal(matrix2) / np.sum(matrix2, axis=0)
# recall = np.diagonal(matrix2) / np.sum(matrix2, axis=1)
# f1 = 2 * (precision * recall) / (precision + recall)
# accuracy = np.average(np.diagonal(matrix2) / 248)
# macro_avg_precision = np.average(precision)
# macro_avg_recall = np.average(recall)
# avg_f1 = np.average(f1)
#
# for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
#     print(f"{i}\t&\t{p:.2f}\t&\t{r:.2f}\t&\t{f:.2f}\t&\t248  \\\\")
#
# print(f"average\t&\t{macro_avg_precision:.2f}\t&\t{macro_avg_recall:.2f}\t&\t{avg_f1:.2f}\t&\t{248*7}  \\\\")

classes_ck = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Contempt"]
classes_fer = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral\\\nContempt"]
counts_fer = [4593, 547, 5121, 8989, 6077, 4002, 6198]
counts_ck = [45, 59, 25, 69, 28, 83, 18]

x = np.arange(len(classes_fer))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, counts_fer, width, label='FER2013')
rects2 = ax.bar(x + width/2, counts_ck, width, label='CK+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Samples')
ax.set_title('Distribution of samples per class')
ax.set_xticks(x)
ax.set_xticklabels(classes_fer)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.show()

# data_path = "../../CK/CK+48/"
# data_dir_list = os.listdir(data_path)

# dataset = pd.read_csv("../../archive (1)/fer2013.csv")
# example_data = pd.DataFrame()

# for x in range(50):
#     example_data = pd.DataFrame()
#     for key in range(7):
#         example_data = pd.concat([example_data, dataset[dataset['emotion'] == key][10+x:11+x]])
#
#     gen_data = example_data["pixels"].str.split(" ").tolist()
#     gen_data = np.uint8(gen_data)
#     gen_data = gen_data.reshape((len(example_data), 48, 48, 1))
#     gen_data = gen_data.astype("float32") / 255
#     plt.figure(figsize=(7, 1.5))
#     for i, image in enumerate(gen_data):
#         plt.subplot(1, 7, i + 1)
#         plt.imshow(image, cmap='gray')
#         plt.xticks([])
#         plt.yticks([])
#         plt.xlabel(classes_fer[i])
#     plt.tight_layout()
#     plt.show()

# images = [cv2.imread(data_path + '/' + data + '/' + os.listdir(data_path + '/' + data)[0]) for index, data in enumerate(sorted(data_dir_list))]

# centers_px = np.load("../clusters/cluster_center_px-diff.npz")
# centers_hist = np.load("../clusters/cluster_center.npz")
# centers_arr = []
# centers_arr_hist = []
# for x, y in zip(centers_px, centers_hist):
#     centers_arr.append(centers_px[x])
#     centers_arr_hist.append(centers_hist[y])
#
# plt.figure(figsize=(7, 1.5))
# for i, image in enumerate(images):
#     plt.subplot(1, 7, i + 1)
#     plt.imshow(image, cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(classes_ck[i])
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(7, 1.5))
# for i, image in enumerate(gen_data):
#     plt.subplot(1, 7, i + 1)
#     plt.imshow(image, cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(classes_fer[i])
# plt.tight_layout()
# plt.show()


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
    fig, ax = plt.subplots(figsize=figsize)
    sn.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax, vmax=100, vmin=0)


