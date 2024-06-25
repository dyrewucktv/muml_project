import torch
from sklearn.model_selection import train_test_split
import numpy as np


class CMIEstimator(torch.nn.Module):
    def __init__(self, z_size=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = torch.nn.Sequential(
            torch.nn.Linear(z_size + 2, 8, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(8, 1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


def divergence_based_cmi(v2, vp, batch_size=16, z_size=1):
    """
    calculate cmi using mine with f-divergence representation of KL
    f-divergence sgd gradients should be unbiased in contrast to DV representation
    """
    # create model
    model = CMIEstimator(z_size)
    optimizer = torch.optim.RAdam(model.parameters(), lr=.01)
    # split data for training and testing
    v2_train, v2_test, vp_train, vp_test = train_test_split(v2, vp, test_size=.33)
    # train the model
    # iterate over batches of the dataset
    v2_train = torch.tensor(v2_train, dtype=torch.float32)
    vp_train = torch.tensor(vp_train, dtype=torch.float32)
    max_res = -torch.inf
    no_imp_ctx = 0
    for epoch in range(100):
        v2_train = v2_train[np.random.permutation(v2_train.shape[0])]
        vp_train = vp_train[np.random.permutation(vp_train.shape[0])]
        model.train()
        for i in range(0, v2_train.shape[0], batch_size):
            optimizer.zero_grad()
            loss = - torch.mean(model(v2_train[i:i+batch_size, :])) + torch.mean(torch.exp(model(vp_train[i:i+batch_size, :]) - 1))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            cmi = torch.mean(model(torch.tensor(v2_test, dtype=torch.float32))) - torch.mean(torch.exp(model(torch.tensor(vp_test, dtype=torch.float32)) - 1)).item()
            if cmi > max_res:
                max_res = cmi
                no_imp_ctx = 0
            else:
                no_imp_ctx += 1
                if no_imp_ctx > 5:
                    return max_res
    return max_res
