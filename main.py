# MNIST_FCNN
#
# Created by Stefan Jakovljevic
#
# Simple FCNN that runs on the MNIST dataset

import math
from torch import Tensor
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Tuple, Callable
import matplotlib.pyplot as plt
import random

TRANSFORM: transforms.Compose = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

TRAIN_DATASET: datasets.MNIST = datasets.MNIST(
    root="./data", train=True, download=True, transform=TRANSFORM
)

TEST_DATASET: datasets.MNIST = datasets.MNIST(
    root="./data", train=False, download=True, transform=TRANSFORM
)

def reLU(a: Tensor) -> Tensor:
    return a.relu()


def softmax(a: Tensor) -> Tensor:
    return torch.softmax(a, dim=0)


class MNIST_FCNN:
    def __init__(self, train_data: Tensor, train_targets: Tensor, learning_rate=0.1) -> None:
        # -- Hyperparameters -- #
        self.learning_rate: float = learning_rate
        self.g_1 = reLU
        self.g_2 = softmax

        # -- Sizes -- #
        self.n_0: int = train_data.size(0)
        self.n_1: int = 128 # change and see how model does
        self.m: int = train_data.size(1)

        # -- Matrices -- #
        self.Y: Tensor = train_targets
        self.A_0: Tensor = train_data

        self.W_1: Tensor = torch.empty((self.n_1, self.n_0))
        torch.nn.init.kaiming_normal_(self.W_1, nonlinearity="relu")
        self.B_1: Tensor = torch.zeros((self.n_1, 1))

        self.W_2: Tensor = torch.empty((10, self.n_1))
        torch.nn.init.xavier_normal_(self.W_2)
        self.B_2: Tensor = torch.zeros((10, 1))

    def forward_prop(self, A_0: Tensor) -> Tensor:
        self.A_0 = A_0
        self.Z_1: Tensor = self.W_1 @ self.A_0 + self.B_1
        self.A_1: Tensor = self.g_1(self.Z_1)

        self.Z_2: Tensor = self.W_2 @ self.A_1 + self.B_2
        self.A_2: Tensor = self.g_2(self.Z_2)
        return self.A_2

    def backward_prop(self, Y: Tensor) -> None:
        m = Y.shape[1]

        dZ_2: Tensor = self.A_2 - Y
        dW_2: Tensor = (dZ_2 @ self.A_1.mT) / m
        db_2 = dZ_2.sum(dim=1, keepdim=True) / m

        dA_1 = self.W_2.mT @ dZ_2
        dZ_1 = dA_1 * (self.Z_1 > 0)
        dW_1 = (dZ_1 @ self.A_0.mT) / m
        db_1 = dZ_1.sum(dim=1, keepdim=True) / m

        self.W_2 -= self.learning_rate * dW_2
        self.B_2 -= self.learning_rate * db_2
        self.W_1 -= self.learning_rate * dW_1
        self.B_1 -= self.learning_rate * db_1


def compute_loss(Y_hat: Tensor, Y: Tensor) -> Tensor:
    eps = 1e-9
    loss = -torch.sum(Y * torch.log(Y_hat + eps)) / Y.shape[1]
    return loss


def k_fold_cross_validation() -> Dict:
    m: int = len(TRAIN_DATASET.data)
    k: int = 5  # works great because 60 % 5 = 0
    BATCH_SIZE: int = 64
    EPOCHS: int = 20

    for i in range(k):

        # -- K - fold cross computations -- #
        start: int = (m // k) * i
        end: int = (m // k) * (i + 1)
        val_data: Tensor = TRAIN_DATASET.data[start:end]
        val_targets: Tensor = TRAIN_DATASET.targets[start:end]

        train_data: Tensor = torch.cat(
            (TRAIN_DATASET.data[:start], TRAIN_DATASET.data[end:])
        )
        train_targets: Tensor = torch.cat(
            (TRAIN_DATASET.targets[:start], TRAIN_DATASET.targets[end:])
        )

        # -- One hot encoding -- #
        train_targets_onehot: Tensor = torch.nn.functional.one_hot(train_targets, num_classes=10).T.float()
        val_targets_onehot: Tensor = torch.nn.functional.one_hot(val_targets, num_classes=10).T.float()

        model = MNIST_FCNN(train_data.mT, train_targets_onehot)

        best_val_loss = float('inf')
        best_state = None

        # -- Mini batch -- #
        for epoch in range(EPOCHS):
            # -- Training -- #
            for batch_start in range(0, train_data.shape[0], BATCH_SIZE):
                batch_end = batch_start + BATCH_SIZE
                X_batch = train_data[batch_start:batch_end].mT  # (784, batch)
                Y_batch = train_targets_onehot[:, batch_start:batch_end]  # (10, batch)

                Y_hat = model.forward_prop(X_batch)

                loss = compute_loss(Y_hat, Y_batch)

                model.backward_prop(Y_batch)

            # --- Validation --- #
            Y_val_hat = model.forward_prop(val_data.mT)
            val_loss = compute_loss(Y_val_hat, val_targets_onehot)

            preds = torch.argmax(Y_val_hat, dim=0)
            true_labels = torch.argmax(val_targets_onehot, dim=0)
            val_acc = (preds == true_labels).float().mean()

            print(
                f"Fold {i+1}, Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"Val Acc: {val_acc.item()*100:.2f}%"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'W1': model.W_1.clone(),
                    'B1': model.B_1.clone(),
                    'W2': model.W_2.clone(),
                    'B2': model.B_2.clone(),
                }

    if best_state is None:
        raise RuntimeError("Training failed: no validation loss computed, best_state is undefined.")

    return best_state


def show_ascii(img: Tensor):
    for row in img:
        print("".join("██" if val < 0.5 else "  " for val in row))

def testing_data(model):
    while True:
        user_input = input(
            f"\nEnter a test image index (0-{len(TEST_DATASET.data)-1}), 'r' for random, 'all' to test all, or 'q' to quit: "
        ).strip().lower()

        if user_input == "q":
            print("Exiting.")
            print("******************")
            break

        elif user_input == "all":
            print("Running full test evaluation...")
            X_test = TEST_DATASET.data.T  # [784, m]
            Y_test = TEST_DATASET.targets

            Y_hat_all = model.forward_prop(X_test)
            preds = torch.argmax(Y_hat_all, dim=0)
            correct = (preds == Y_test).sum().item()
            total = Y_test.size(0)
            acc = correct / total * 100

            print(f"Model accuracy on entire test set: {acc:.2f}% ({correct}/{total})")
            print("******************")
            continue

        elif user_input == "r":
            idx = random.randint(0, len(TEST_DATASET.data) - 1)

        else:
            try:
                idx = int(user_input)
                if idx < 0 or idx >= len(TEST_DATASET.data):
                    print("Index out of range.")
                    continue
            except ValueError:
                print("Invalid input. Enter an integer, 'r', 'all', or 'q'.")
                continue

        img = TEST_DATASET.data[idx].view(28, 28)
        label = TEST_DATASET.targets[idx].item()

        x = img.view(-1, 1)  # [784, 1]
        y_hat = model.forward_prop(x)
        pred = torch.argmax(y_hat).item()

        plt.imshow(img, cmap="gray")
        plt.title(f"True: {label} | Predicted: {pred}", fontsize=14)
        plt.axis("off")

        plt.savefig(f"tests/test_image_{idx}.png")
        print(f"Saved image to tests/test_image_{idx}.png")
        plt.close()

        print("\nASCII rendering of the digit:\n")
        show_ascii(img)
        print(f"\nModel prediction: {pred}")
        print(f"True label: {label}")
        print("******************")

# Build a shallow, 2-layer NN
def main() -> None:
    # Flatten 28x28 → 784x1
    TRAIN_DATASET.data = TRAIN_DATASET.data.view(-1, 28 * 28).float() / 255.0
    TEST_DATASET.data = TEST_DATASET.data.view(-1, 28 * 28).float() / 255.0

    best_state = k_fold_cross_validation()
    model = MNIST_FCNN(
            TRAIN_DATASET.data.T,
            torch.nn.functional.one_hot(TRAIN_DATASET.targets, num_classes=10).T.float(),
        )
    model.W_1 = best_state["W1"].clone()
    model.B_1 = best_state["B1"].clone()
    model.W_2 = best_state["W2"].clone()
    model.B_2 = best_state["B2"].clone()

    testing_data(model)


if __name__ == "__main__":
    main()
