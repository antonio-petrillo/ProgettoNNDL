import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

import numpy as np

from enum import Enum
from typing import Tuple, Dict, List, Union
from datetime import datetime

OUTPUT = 10
INPUT = 784
CPU = torch.device("cpu")
PATIENCE = 5
EPOCHS = 50
BATCH_SIZE = 128
MODELS_TO_KEEP = 3


class ParameterSearchKind(Enum):
    GRID = "GRID"
    RANDOM = "RANDOM"

    def __str__(self):
        return self.value.upper()


class Network(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT, hidden),
            nn.ReLU(),
            nn.Linear(hidden, OUTPUT),
        )

    def forward(self, x):
        return self.model(x)


def train_and_validate_with_params(
    model: Network,  # the network model to train
    train_loader: DataLoader,  # training dataset
    validation_loader: DataLoader,  # validation dataset
    etas: Tuple[float, float],  # the (η-,η+) to use in Rprop
    epochs: int = EPOCHS,  # for how many epoch the network should be trained
    patience: int = PATIENCE,  # how many 'bad epochs' should we see before early stopping
):
    # cross entropy since we have a classification problem
    cross_loss = nn.CrossEntropyLoss()

    # use the Resilient Backpropagation as weight update algorithm
    optimizer = torch.optim.Rprop(
        model.parameters(),
        etas=etas,
    )

    best_loss = float("inf")

    patience_counter = 0

    loss, accuracy = 0, 0
    all_predictions, all_targets = [], []

    for epoch in range(1, epochs + 1):
        model.train()  # set the model in training mode
        for x, y in train_loader:
            optimizer.zero_grad()  # reset the gradient

            x, y = x.to(CPU), y.to(CPU)
            output = model(x)
            loss = cross_loss(output, y)

            loss.backward()
            optimizer.step()

        model.eval()  # change to evaluation mode

        losses = []
        predictions, targets = [], []

        # disable gradient calculation, so the evaluation goes faster
        with torch.no_grad():
            for x, y in validation_loader:
                x, y = x.to(CPU), y.to(CPU)

                output = model(x)
                loss = cross_loss(output, y)

                losses.append(loss.item())
                predictions.extend(output.argmax(dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())

        all_predictions.extend(predictions)
        all_targets.extend(targets)
        loss = np.mean(losses)
        accuracy = metrics.accuracy_score(targets, predictions)

        print(
            f"Epoch:{epoch:3d}/{epochs}: - Loss: {loss:.7f} - Accuracy: {accuracy:.7f}%"
        )

        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                # The 'model' doesn't improve for '$patience' epochs, kill the training to prevent overfitting
                break

    return float(loss), accuracy, all_predictions, all_targets


def k_fold_cross_validate(
    dataset: VisionDataset,
    k: int,
    parameters_space: Dict[str, List[Union[int, float]]],
    search_kind: ParameterSearchKind,
    random_iteration: int | None = None,
    # the number of random search to perform in case of 'random' search
):

    if search_kind == ParameterSearchKind.RANDOM and random_iteration is None:
        raise ValueError(
            "With 'random' expected 'random_iteration' to be an integer but got None"
        )

    params_list = list(
        ParameterGrid(parameters_space)
        if search_kind == ParameterSearchKind.GRID
        else ParameterSampler(parameters_space, n_iter=random_iteration)
    )

    # K-Fold cross validation
    indices = [i for i in range(len(dataset))]
    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)

    results = []

    for params in params_list:
        accuracies, losses = [], []
        print("\n------------------ START ---------------------")
        print(f"Search Type:{search_kind} - Parameters: {params}")

        all_predictions, all_targets = [], []
        for fold, (train_idx, validate_idx) in enumerate(k_fold.split(indices)):
            print(f"Current Fold: {fold+1:2d}/{k}")

            train = Subset(dataset, train_idx)
            validate = Subset(dataset, validate_idx)

            train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
            validation_loader = DataLoader(
                validate, batch_size=BATCH_SIZE, shuffle=False
            )

            model = Network(hidden=params["hidden_size"]).to(CPU)
            loss, accuracy, predictions, targets = train_and_validate_with_params(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                etas=(params["eta_minus"], params["eta_plus"]),
            )

            accuracies.append(accuracy)
            losses.append(loss)
            all_predictions.extend(predictions)
            all_targets.extend(targets)

        avg_accuracy = np.mean(accuracies)
        avg_loss = np.mean(losses)

        results.append((params, avg_accuracy, avg_loss, all_predictions, all_targets))

        print("------------------- RECAP ----------------------\n")
        print(f"Avg loss: {avg_loss:.7f}, Avg accuracy: {avg_accuracy:.7f}")
        print("------------------- END ----------------------\n")

    results = sorted(results, key=lambda x: x[1], reverse=True)
    # return only the first 'MODELS_TO_KEEP'
    return results[: min(MODELS_TO_KEEP, len(results))]


def plots_results(result_grid, result_random):
    results = [
        (result_grid, ParameterSearchKind.GRID),
        (result_random, ParameterSearchKind.RANDOM),
    ]
    _, axes = plt.subplots(2, len(result_grid), figsize=(15, 10))

    for i, (data, kind) in enumerate(results):
        for j, result in enumerate(data):
            params, avg_accuracy, avg_loss, predictions, targets = result
            confusion_matrix = metrics.confusion_matrix(targets, predictions)
            precision = metrics.precision_score(
                targets, predictions, average="macro", zero_division=0
            )
            recall = metrics.recall_score(
                targets, predictions, average="macro", zero_division=0
            )

            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            disp.plot(ax=axes[i, j], cmap=plt.cm.Oranges)
            title = f"Search: {kind}\nParams: hidden_size={params['hidden_size']}, eta_minus={params['eta_minus']}, eta_plus={params['eta_plus']}"
            axes[i, j].set_title(title, fontsize=10)
            score = f" Avg Loss: {avg_loss:4.4f}        Avg Accuracy: {avg_accuracy:4.4f}\nRecall: {recall:4.4f}       Precision: {precision:4.4f}"
            axes[i, j].text(
                0.5,
                -0.15,
                score,
                ha="center",
                va="top",
                transform=axes[i, j].transAxes,
                fontsize=9,
            )

    # denote like file name
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"plot--{timestamp}"
    plt.tight_layout()
    plt.savefig(f"./plots/{filename}.png")
    plt.savefig(f"./plots/{filename}.svg")
    plt.show()


def main():
    K = 10  # how many folds
    N_ITER = 10  # how many iteration in random search

    mnist = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        download=True,
    )

    param_space = {
        "hidden_size": [128, 256, 512, 768, 1024],
        "eta_plus": [1.1, 1.2, 1.5],
        "eta_minus": [0.4, 0.5, 0.6],
    }

    print("GRID")
    best_grid_results = k_fold_cross_validate(
        mnist, K, param_space, ParameterSearchKind.GRID
    )
    print("RANDOM")
    best_random_results = k_fold_cross_validate(
        mnist, K, param_space, ParameterSearchKind.RANDOM, N_ITER
    )

    plots_results(best_grid_results, best_random_results)


if __name__ == "__main__":
    try:
        main()
    except:
        pass
