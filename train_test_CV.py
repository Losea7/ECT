import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.Transformer import TransformerHandcrafted
from models.DataLoader import VideoDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


def load_vids_csv(data_name, label_name):
    labels = pd.read_csv(label_name).class_label.tolist()
    sequences = torch.Tensor(pd.read_csv(data_name).to_numpy())
    return sequences, labels


def progress(i, total_epoches, loss):
    """展示进度"""
    if i % 100 == 0:
        print(f"Epoch {i}/{total_epoches}, Loss: {loss:.4f}")


def train(model, criterion, optimizer, dataloader, num_epochs=100):
    """训练模型"""
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


def score(model, X_test, Y_test, labels_enc):
    """计算模型准确度"""
    device = torch.device('cpu')
    model.eval()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    model.to(device)
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
    return accuracy, precision, recall



def k_fold_cross_validation(n_splits, sequences, labels, model_params, num_epochs=100, lr=0.0001, batch_size=16):
    '''k折交叉验证'''
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, precisions, recalls = [],[],[]
    n = 0
    for train_index, test_index in kf.split(sequences):
        n += 1
        print('Traning epoch',num_epochs,'fold',n)
        X_train, X_test = sequences[train_index], sequences[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]

        train_dataset = VideoDataset(X_train, Y_train)
        test_dataset = VideoDataset(X_test, Y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = TransformerHandcrafted(**model_params)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)
        model.to(device=device)  # 将模型移到GPU上

        folder_name = "10-weights_CV"
        model_name = "weights_" + str(num_epochs) + "_" + str(n) + ".pt"

        if model_name in os.listdir("10-weights_CV"):
            model_path = os.path.join(os.getcwd(), "10-weights_CV", model_name)
            model.load_state_dict(
                torch.load(
                    model_path, 
                    map_location=torch.device(device),
                )
            )
        else:
            train(model, criterion, optimizer, train_loader, num_epochs)
            save_path = os.path.join(folder_name, model_name)
            torch.save(model.state_dict(), save_path)


        accuracy, precision, recall = score(model, X_test, Y_test, labels_enc)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    print(f"Average Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"Average Precision: {sum(precisions) / len(precisions)}")
    print(f"Average Recall: {sum(recalls) / len(recalls)}")
    return sum(accuracies) / len(accuracies), sum(precisions) / len(precisions), sum(recalls) / len(recalls)


if __name__ == "__main__":

    """1.定义设备"""
    device = torch.device('cuda')

    """2.加载数据"""
    data_name = "9-train_data_new/train-data.csv"
    label_name = "9-train_data_new/train-label.csv"
    sequences, labels = load_vids_csv(data_name, label_name)
    sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    labels_enc = LabelEncoder()
    labels = labels_enc.fit_transform(labels)
    print(type(labels_enc.classes_),labels_enc.classes_)
    labels = torch.Tensor(labels).long()
    input_dim = sequences.shape[1]
    output_dim = torch.unique(labels).shape[0]
    print("input_dim:", input_dim)
    print("output_dim:", output_dim)
    print("sequences shape:", *sequences.shape)

    """3.定义模型"""
    model_params = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "in_channels": 1,
        "out_channels": 10,
        "kernel_size": 3,
        "stride": 3,
        "num_layers": 1,
    }

    """4.应用交叉验证"""
    n_splits = 5
    # num_epochs = 100
    # lr = 0.0001
    # k_fold_cross_validation(n_splits, sequences, labels, model_params, num_epochs=num_epochs, lr=lr)

    accuracies, precisions, recalls = [],[],[]
    epochs_list = list(range(1500, 4501, 500))
    for num_epochs in epochs_list:
        lr = 0.0001
        accuracy, precision, recall = k_fold_cross_validation(n_splits, sequences, labels, model_params, num_epochs=num_epochs, lr=lr)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    print(accuracies,precisions,recalls)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, accuracies, label='Accuracy', marker='o', linestyle='-')
    plt.plot(epochs_list, precisions, label='Precision', marker='s', linestyle='--')
    plt.plot(epochs_list, recalls, label='Recall', marker='^', linestyle=':')
    plt.legend()

    plt.title('Performance Metrics vs. Number of Epochs')
    plt.xlabel('Number of Epochs',fontsize=14)
    plt.ylabel('Metrics Values',fontsize=14)
    plt.grid(True)
    plt.savefig("train_test_result.png", dpi=300, bbox_inches='tight')
    plt.show()
    

