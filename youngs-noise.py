import numpy as np
import pandas as pd
import re
import cv2

from random import sample
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
from glob import glob
from json import load, dump
from os.path import basename
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from matplotlib import pyplot as plt
import seaborn
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("ggplot")
from collections import Counter
from random import sample
from tqdm import tqdm

from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
import scipy.io

RANDOM_SEED = 5012021

input_df = pd.read_csv("./data/data-10-12/Disp_Output.csv")
input_df.head()

feature_df = input_df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 14, 18, 19]]
feature_df.loc[:, "phi"] = np.sin(feature_df["phi"])
feature_df.loc[:, "psi"] = np.sin(feature_df["psi"])

# Need to handle the nan cases in the last feature
max_stress = list(np.sort(list(set(feature_df["yield_stress"]))))[-2] * 1.2
feature_df.loc[:, "yield_stress"] = [
    x if x != np.inf else max_stress for x in feature_df["yield_stress"]
]

feature_df_normalized = feature_df.copy()
feature_df_normalized

# feature_js = [1, 2, 3, 4, 7, 8]
feature_js = [1, 2, 4, 7, 8, 10, 11]
scaler = StandardScaler()
scaler.fit(feature_df.iloc[:, feature_js])
feature_df_normalized.iloc[:, feature_js] = scaler.transform(
    feature_df.iloc[:, feature_js]
)

# Normalize E using the min and max values
e_range = np.max(feature_df["E"]) - np.min(feature_df["E"])
e_min = np.min(feature_df["E"])
feature_df_normalized["norm_E"] = (
    feature_df_normalized["E"] - np.min(feature_df["E"])
) / e_range

heatmap_image_data = np.load("./data/displayment-heatmaps-normalized-resized.npz")
heatmaps_images = heatmap_image_data["rs_normalized_heatmaps"]
heatmaps_ids = heatmap_image_data["heatmaps_ids"]

indexes = [int(x[5:]) for x in feature_df["Folder"]]
indexes.index(10)

features = []
scores = []

for i in tqdm(range(len(heatmaps_ids))):
    df_i = indexes.index(heatmaps_ids[i])

    # Collect the features
    row = feature_df_normalized.iloc[df_i, :]
    cur_features = row[["Depth", "d", "Mw", "Mc"]].to_numpy().astype(float)
    features.append((heatmaps_images[i], cur_features))

    # Collect the y's
    scores.append(row["norm_E"])

features = np.array(features)
scores = np.array(scores)

train_features, temp_features, train_scores, temp_scores = train_test_split(
    features, scores, train_size=0.6, random_state=RANDOM_SEED
)
vali_features, test_features, vali_scores, test_scores = train_test_split(
    temp_features, temp_scores, train_size=0.5, random_state=RANDOM_SEED
)

print(
    train_features.shape,
    train_scores.shape,
    vali_features.shape,
    vali_scores.shape,
    test_features.shape,
    test_scores.shape,
)


class SimpleCNN(nn.Module):
    def __init__(self, additional_feature_size):
        super(SimpleCNN, self).__init__()
        self.additional_feature_size = additional_feature_size
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(6, 6, 3)
        self.conv4 = nn.Conv2d(6, 6, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(6, 6, 3)
        self.conv6 = nn.Conv2d(6, 6, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(6 * 28 * 28 + self.additional_feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, image, feature):
        image = image.view(-1, 1, 256, 256)

        # Passing the CNN layers
        x = F.relu(self.conv1(image))
        x = self.pool1(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))

        x = F.relu(self.conv5(x))
        x = self.pool3(F.relu(self.conv6(x)))

        x = x.view(-1, 6 * 28 * 28)

        # Concatenate with extra features
        x = torch.cat((x, feature), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # x = F.softmax(x, dim=1)
        return x


class HeatmapDataset(Dataset):
    def __init__(self, features, labels, std=0):
        self.features = features
        self.labels = labels
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cur_feature = self.features[idx]
        # label = nn.functional.one_hot(torch.tensor(labels[idx]), num_classes=3)
        label = torch.tensor([self.labels[idx]]).float()

        # Load image and extra feature
        image = torch.tensor(cur_feature[0]).float()
        extra_feature = torch.tensor(cur_feature[1]).float()

        # Transform image
        image = image.view((1, 256, 256))

        # Add random noise
        noise = torch.normal(0, self.std, (1, 256, 256))
        image = torch.clamp(image + noise, 0, 1)

        sample = {"image": image, "feature": extra_feature, "label": label}

        return sample


def train_one_epoch(
    train_dataloader, epoch, print_every_ter=None, verbose=True, selected_indexes=None
):
    losses = []
    all_losses = []
    y_predict = []
    y_true = []
    for i, data in enumerate(train_dataloader, 0):
        cur_images, cur_features, cur_labels = (
            data["image"].to(device),
            data["feature"].to(device),
            data["label"].to(device),
        )

        # Only use depth and d here
        if selected_indexes != None:
            cur_features = cur_features[:, selected_indexes]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(cur_images, cur_features)
        loss = criterion(outputs, cur_labels)
        loss.backward()
        optimizer.step()

        y_predict.extend(outputs.tolist())
        y_true.extend(cur_labels.tolist())

        # print statistics
        losses.append(loss.item())

        if print_every_ter != None:
            if i % print_every_ter == print_every_ter - 1:
                print(
                    "(epoch {}, iter {}) avg loss: {:3f}".format(
                        epoch + 1, i + 1, np.mean(losses)
                    )
                )
                all_losses.extend(losses)
                losses = []
        else:
            all_losses.append(loss.item())

    # Evaluate the accuracy
    mse = metrics.mean_squared_error(y_true, y_predict)
    avg_loss = np.mean(all_losses)

    if verbose:
        print(
            "Epoch {}: training acc: {:.4f} avg loss: {:.4f}".format(
                epoch, acc, avg_loss
            )
        )

    return mse, avg_loss


def eval_model(test_dataloader, model, selected_indexes=None, l1_error=False):
    losses = []
    y_predict = []
    y_true = []

    for i, data in enumerate(test_dataloader, 0):
        cur_images, cur_features, cur_labels = (
            data["image"].to(device),
            data["feature"].to(device),
            data["label"].to(device),
        )

        # Only use depth and d here
        if selected_indexes != None:
            cur_features = cur_features[:, selected_indexes]

        # forward + backward + optimize
        outputs = model(cur_images, cur_features)
        loss = criterion(outputs, cur_labels)
        losses.append(loss.item())

        y_predict.extend(outputs.tolist())
        y_true.extend(cur_labels.tolist())

    mse = metrics.mean_squared_error(y_true, y_predict)
    mae = metrics.mean_absolute_error(y_true, y_predict)
    avg_loss = np.mean(losses)

    # print('Test accuracy {:.4f} test avg loss {:.4f}'.format(acc, avg_loss))

    if not l1_error:
        return mse, avg_loss
    else:
        return mae, avg_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for std in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    train_dataset = HeatmapDataset(train_features, train_scores, std=std)
    vali_dataset = HeatmapDataset(vali_features, vali_scores, std=std)
    test_dataset = HeatmapDataset(test_features, test_scores, std=std)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    vali_dataloader = DataLoader(vali_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    # PATIENCE = 200
    PATIENCE = 50
    epochs = 2000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print("Using device:", device)

    selected_indexes = [0, 1, 2, 3]

    model = SimpleCNN(len(selected_indexes)).to(device)

    criterion = nn.MSELoss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_mses = []
    train_avg_losses = []

    val_mses = []
    val_avg_losses = []

    best_loss = np.inf
    best_mse = np.inf
    waited = 0
    best_model = None

    for e in tqdm(range(epochs)):
        mse, avg_loss = train_one_epoch(
            train_dataloader, e, verbose=False, selected_indexes=selected_indexes
        )
        train_mses.append(mse)
        train_avg_losses.append(avg_loss)

        # Test on the validation set
        val_mse, val_avg_loss = eval_model(
            vali_dataloader, model, selected_indexes=selected_indexes
        )
        val_mses.append(val_mse)
        val_avg_losses.append(val_avg_loss)

        if val_mse < best_mse:
            # best_loss = val_avg_loss
            best_mse = val_mse
            waited = 0
            best_model = copy.deepcopy(model)
        else:
            waited += 1

        if waited == PATIENCE + 1:
            break

    print("Done!", std)
    torch.save(best_model.state_dict(), f"./model/cnn-e-extra-noise-{std}.model")
