'''
    This file contains the primary model architecture, which will consist of a CNN,
    along with an RNN after the image is processed, and a few fully-connected layers
    to finish the network. It willa also contain the training loop and stats code for
    model training and validation in terms of the outputs. stats_calc.py will contain
    distance-related calculations.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The neural net itself in all its glory
class FetchNN(torch.nn.Module):
    def __init__(self, rnn_hidden_size=64):
        super(FetchNN, self).__init__()

        # Size of the hidden state in the rnn
        self.rnn_hidden_size = rnn_hidden_size

        # Convolutional portion: Go from 3 to 64 channels, from a 96x96 image down to 
        # 8x8 to capture as many features as possible
        self.conv1 == nn.Conv2d(3, 8, kernel_size=3) # 96x96 --> 94x94
        self.pool = nn.MaxPool2d(2, 2) # 94x94 --> 47x47
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3) # 47x47 --> 45x45 --> 22x22
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3) # 22x22 --> 20x20 --> 10x10
        self.conv4 = nn.Conv2d(32, 64, kernel_size = 3) # 10x10 --> 8x8

        # Pass these through a fully-connected layer before giving it to the RNN
        self.conv_fc = nn.Linear(64*8*8, 64)

        # Create the RNN layer which will need input of size 82:
        # 64 for the previous fc layer, 12 for joint angles, 6 for imu acc and gyro
        self.rnn1 = nn.RNN(82, rnn_hidden_size, batch_first=True)

        # Final FC layers for output
        self.final_fc1 = nn.Linear(64, 16)
        self.final_fc2 = nn.Linear(16, 2)

    def forward(self, camera_img, sensor_data, prev_output):
        # Convolutional forward pass
        x = self.pool(F.relu(self.conv1(camera_img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten but keep batch size and pass through first fc
        x = x.view(x.size(0), -1)
        x = F.relu(self.conv_fc(x))

        # Concatenate inputs and give to RNN
        x = torch.cat((camera_img, sensor_data, prev_output), dim=0)

        # Set starting hidden state to be zeroed out and pass to rnn
        h0 = torch.zeros(1, x.size(0), self.rnn_hidden_size)
        out, _ = self.rnn1(x, h0)

        # Final ANN gets last RNN output
        x = F.relu(self.final_fc1(out[:, -1, :]))
        return self.final_fc2(x)

# Get DataLoader objects for test, train, and val data from the saved .npz files
def process_dataset():
    # Load in all the data and put it in an array
    runs = []
    for f in sorted(glob.glob("data/run_*.npz")):
        data = np.load(f)
        runs.append((torch.from_numpy(data["img"]), 
                torch.from_numpy(np.concatenate([data["acc"], data["gyro"], data["q"]], 1)).float(), 
                torch.from_numpy(data["label"]).float())) 

    # Do train test val split
    np.random.seed(50)

    indices = list(range(len(runs)))
    np.random.shuffle(indices)

    val_begin = round(len(indices)*0.7)
    test_begin = round(len(indices)*0.85)

    train_data = runs[indices[:val_begin]]
    val_data = runs[indices[val_begin:test_begin]]
    test_data = runs[indices[test_begin:]]

    train_data_cat = torch.cat([p for _, p, _ in train_data])
    mean, std_dev = train_data_cat.mean(0), train_data_cat.std(0) + 1e-6

    return train_data, val_data, test_data, mean, std_dev

# Create dataloader objects from the data
def create_dl(ds, mean, std_dev, window=64, stride=16, bs=16, shuffle=False):
    img_data = []
    joint_data = []
    label_data = []

    for img, joint, label, in ds:
        for i in range(0, len(label) - window  + 1, stride):
            img_data.append(img[i:i+window])
            # This has to be specially normalized here as joint data, accelerometer, and gyro are on different scales
            joint_data.append((joint[i:i+window] - mean) / std_dev)
            label_data.append(label[i:i+window])

    return DataLoader(TensorDataset(torch.stack(img_data), torch.stack(joint_data), torch.stack(label_data)), batch_size=bs, shuffle=shuffle)

# Preprocess images so they can be fed without biases, like normalziing colour channels and reordering sections
def preprocess_image(img):
    B, T = img.shape[:2]
    x = img.flatten(0, 1).permute(0, 3, 1, 2).float() / 255

    # Squash width to make square
    x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False, antialias=True)
    return x.unflatten(0, (B, T))

# Get the absolute error between the predicted values and absolute values of the direction
def pred_mean_abs_err(pred, gt):
    # 2-dimensional, first is the error of the turn, second is of the forward strength
    return (pred - gt).abs().mean(0)

# Also get accuracy in terms of whether or not we are facing the right way
def right_way_accuracy(pred, gt):
    # Ignore when we didn't really need to turn or change directions
    mask = gt[:, 0].abs() > 0.1

    # Basically gives us 1 if we guess both right, 0.5 if we got one right, and 0 if we missed both
    return (pred[mask, 0].sign() == gt[mask, 0].sign()).float().mean()

def train(model, train_data, val_data, num_epochs=20, learning_rate=0.001):
    torch.manual_seed(67)
    model.to(device)

    # Regression problem so use MSELoss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation stats to track
    stats = {k: [] for k in ("train_loss", "train_abs_err", "train_sign_acc",
                            "val_loss", "val_abs_err", "val_sign_acc")}

    def run_epoch(loader, training):
        model.train(training)
        # Individual stats per epoch to track
        loss_sum = ae_sum = hit = n_sign = n = 0.0

        with torch.set_grad_enabled(training):
            for img, sensor, label in loader:
                img, sensor, label = img.to(device), sensor.to(device), label.to(device)
                img = preprocess_image(img)

                out = model(img, sensor)                  # (B, T, 2)
                loss = criterion(out, label)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                k = label.numel()
                loss_sum += loss.item() * k
                ae_sum += (out - label).abs().sum().item()
                n += k

                # right turn direction? skip frames where the target is ~straight
                valid = label[..., 0].abs() > 0.1
                hit += ((out[..., 0].sign() == label[..., 0].sign()) & valid).sum().item()
                n_sign += valid.sum().item()

        return loss_sum / n, ae_sum / n, hit / max(n_sign, 1)

    for epoch in range(num_epochs):
        tl, tae, tsa = run_epoch(train_data, True)
        vl, vae, vsa = run_epoch(val_data, False)
        for key, v in zip(stats, (tl, tae, tsa, vl, vae, vsa)):
            stats[key].append(v)
        print(f"epoch {epoch+1:2d}/{num_epochs} | loss {tl:.4f}/{vl:.4f} | "
              f"mae {tae:.3f}/{vae:.3f} | sign acc {tsa:.2f}/{vsa:.2f}")

    return model, stats

def plot_hist(stats):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name, title in zip(axes,
                               ("loss", "abs_err", "sign_acc"),
                               ("MSE loss", "Mean abs error", "Turn-sign accuracy")):
        ax.plot(stats[f"train_{name}"], label="train")
        ax.plot(stats[f"val_{name}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Instantiate model
    fetch_model = FetchNN()

    # Get train, test, and validation dataloaders
    train_data, val_data, test_data, mean, std_dev = process_dataset()

    train_dl = create_dl(train_data, mean, std_dev, stride=16, shuffle=True)
    val_dl = create_dl(val_data, mean, std_dev, stride=32, shuffle=False)
    test_dl = create_dl(test_data, mean, std_dev, stride=32, shuffle=False)

    # Actually train the model, then plot the data after all epochs
    fetch_model, stats = train(fetch_model, train_dl, val_dl, num_epochs=10, learning_rate=0.001)
    plot_hist(stats)