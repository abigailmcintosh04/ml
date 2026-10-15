import pandas as pd
import torch
from torch import nn, optim


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


if __name__ == "__main__":
    main()
