import os
import pandas as pd
import torch
from torch.optim import Adam
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from models.Transformer import TransformerHandcrafted
from models.DataLoader import VideoDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_vids_seq(sequence_length, input_path):
    """Load Video Sequences"""
    file_names = []
    sequences = []
    file_list = os.listdir(input_path)
    for file_name in file_list:
        video_file = pd.read_csv(input_path + file_name, index_col=None)
        sequence = video_file.iloc[:, 2].to_list()
        file_names.append(file_name)
        sequences.append(sequence)

    sequences = [
        sequence + [0] * (sequence_length - len(sequence)) for sequence in sequences
    ]
    return file_names, sequences


def load_vids_csv(data_name, label_name):
    labels = pd.read_csv(label_name).class_label.tolist()
    sequences = torch.Tensor(pd.read_csv(data_name).to_numpy())
    return sequences, labels


def progress(i, total_epoches, loss):
    """Show Current Progress"""
    print(f"Epoch {i}/{total_epoches}, Loss: {loss:.4f}")


def train(model, model_name, criterion, optimizer, dataloader, num_epochs=100):
    """Train Model"""
    for i in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num = 0
        for X_train, Y_train in dataloader:
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            num += 1
            optimizer.zero_grad()
            y = model(X_train)
            loss = criterion(y, Y_train).sum()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        progress(i, num_epochs, epoch_loss / num)
        # 指定保存路径
        folder_name = "4-weights"
        save_path = os.path.join(folder_name, model_name)
        torch.save(model.state_dict(), save_path)
    print("Train Finish")


def score(model, X_test, Y_test, labels_enc):
    """Calculate Precision,Recall and Accuracy"""
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    Y_pred = model(X_test).argmax(axis=1)
    cm = confusion_matrix(Y_test, Y_pred)
    accs = cm.diagonal() / cm.sum(axis=1)
    precs = precision_score(Y_test, Y_pred, average=None)
    recalls = recall_score(Y_test, Y_pred, average=None)

    rates, nums = [], []
    labels = torch.unique(Y_test)
    labels.sort()
    for label in labels:
        indices = torch.where(Y_test == label)[0]
        rates.append(len(indices) / len(Y_pred))
        nums.append(len(indices))
    labels = labels_enc.inverse_transform(labels)
    print("*" * 80)
    print(
        "{:20s}{:10}{:10}{:10}{:10}{:10}".format(
            "name", "acc", "prec", "recall", "rate", "num"
        )
    )
    for label, acc, prec, recall, rate, num in zip(
        labels, accs, precs, recalls, rates, nums
    ):
        print(
            "{:20s}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}{:<10}".format(
                label, acc, prec, recall, rate, num
            )
        )
    print("*" * 80)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average="macro")
    recall = recall_score(Y_test, Y_pred, average="macro")
    print("Accuracy Score:", accuracy)
    print("Precision Score:", precision)
    print("Recall Score:", recall)
    print("*" * 80)


if __name__ == "__main__":
    """1.Config Device"""
    # device = torch.device('cuda')
    device = torch.device('cpu')
    #device = torch.device("mps")

    """2.Load Data"""
    # Loads Videos' Sequences
    data_name = "9-train_data_new/train-data.csv"
    label_name = "9-train_data_new/train-label.csv"
    sequences, labels = load_vids_csv(data_name, label_name)
    sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    print(sequences.shape)
    labels_enc = LabelEncoder()
    labels = labels_enc.fit_transform(labels)
    labels = torch.Tensor(labels).long()
    X_train, X_test, Y_train, Y_test = train_test_split(
        sequences, labels, test_size=0.6
    )
    input_dim = X_train.shape[1]
    output_dim = torch.unique(Y_train).shape[0]
    print("input_dim:", input_dim)
    print("output_dim:", output_dim)
    print("sequences shape:", *sequences.shape)

    # Get Dataset
    dataset = VideoDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=16, drop_last=True, shuffle=True, num_workers=4) 

    """3.Train Model"""
    model_params = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "in_channels": 1,
        "out_channels": 10,
        "kernel_size": 3,
        "stride": 3,
        "num_layers": 1,
    }
    model_name = "transformer_weights_0327.pt"
    model = TransformerHandcrafted(**model_params)

    if model_name in os.listdir("4-weights"):
        model_path = os.path.join(os.getcwd(), "4-weights", model_name)
        model.load_state_dict(
            torch.load(
                model_path, 
                map_location=torch.device(device),
            )
        )

    epoch = 100
    lr = 0.0001
    criterion = nn.CrossEntropyLoss()
    #optimizer = SGD(model.parameters(), lr=lr)
    optimizer = Adam(model.parameters(), lr=lr) # 替换了优化器
    if device.type != "cpu":
        model = model.to(device)
        
    # train(model, model_name, criterion, optimizer, dataloader, epoch)

    """4.Evaluate Model"""
    model.to(device=torch.device(device))
    score(model, X_test, Y_test, labels_enc)