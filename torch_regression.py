import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim


def plot_regression(ax, model, xlow=170, xhigh=235):
    model_x = np.linspace(xlow, xhigh, 1000)
    model_y = model(model_x)
    ax.plot(model_x, model_y, color="tab:orange")


def plot_points(ax, features, targets, xlow=170, xhigh=235, ylow=2400, yhigh=6500):
    ax.scatter(features, targets, marker=".")
    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlabel("Flipper length (mm)")
    ax.set_ylabel("Body mass (g)")


def main():

    penguins_df = pd.read_csv("penguins.csv")
    features, targets = penguins_df.dropna()[
        ["flipper_length_mm", "body_mass_g"]
    ].values.T

    model = nn.Linear(1, 1)
    print(model)

    loss_function = nn.MSELoss()
    optimiser = optim.Rprop(model.parameters())

    tensor_features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    tensor_targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    for _ in range(1000):
        # Tell optimiser to begin optimisation step.
        optimiser.zero_grad()

        # Use model as prediction function.
        preds = model(tensor_features)

        # Calculate the loss.
        loss = loss_function(preds, tensor_targets)

        # Tell loss function and optimiser to end step.
        loss.backward()
        optimiser.step()

    _fig, ax = plt.subplots()

    def numpy_model(x):
        tensor_x = torch.tensor(x[:, np.newaxis], dtype=torch.float32)
        return model(tensor_x).detach().numpy()

    plot_regression(ax, numpy_model)
    plot_points(ax, tensor_features, tensor_targets)

    plt.show()


if __name__ == "__main__":
    main()
