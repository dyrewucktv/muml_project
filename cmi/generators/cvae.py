import torch
import numpy as np


class CVAE_ENCODER(torch.nn.Module):
    def __init__(self, z_dim=1, latent_dim=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = torch.nn.Sequential(
            torch.nn.Linear(z_dim + 1, latent_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim * 4, latent_dim),
            torch.nn.Tanh(),
        )
        self.mean_out = torch.nn.Sequential(torch.nn.Linear(latent_dim, latent_dim))

        self.log_var_out = torch.nn.Sequential(torch.nn.Linear(latent_dim, latent_dim))

    def forward(self, x):
        latent = self.block(x)
        return self.mean_out(latent), self.log_var_out(latent)


class CVAE(torch.nn.Module):
    """
    y - a single column
    z - possibly a few columns
    """

    def __init__(self, z_dim=1, latent_dim=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.z_dim = z_dim
        self.encoder = CVAE_ENCODER(z_dim, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + z_dim, 8 * latent_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(latent_dim * 8, latent_dim * 8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(latent_dim * 8, latent_dim * 2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2 * latent_dim, 1),
        )

    def _sample(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        return z_mean + eps * torch.exp(z_log_var / 2)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self._sample(z_mean, z_log_var)
        return (
            self.decoder(torch.column_stack([z, x[:, -self.z_dim:]])),
            z_mean,
            z_log_var,
        )


def elbo_vae(y, y_reconst, z_mean, z_log_var):
    reconstruction_loss = torch.mean(torch.abs(y - y_reconst))
    kl_loss = 0.5 * torch.mean(-z_log_var + z_mean**2 + torch.exp(z_log_var) - 1)
    return reconstruction_loss + kl_loss


def sample_cvae(x, y, z, batch_size=1024):
    model = CVAE(z_dim=z.shape[1])
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    perm = np.random.permutation(x.shape[0])
    x_train = torch.tensor(x[perm], dtype=torch.float32)
    y_train = torch.tensor(y[perm], dtype=torch.float32)
    z_train = torch.tensor(z[perm], dtype=torch.float32)

    model_input = torch.column_stack([y_train, z_train]).clone()

    for epoch in range(10):
        model.train()

        perm = np.random.permutation(x_train.shape[0])
        model_input = model_input[perm]

        for i in range(0, x_train.shape[0], batch_size):
            optimizer.zero_grad()
            y_recon, z_mean, z_log_var = model(model_input[i: i + batch_size, :])
            loss = elbo_vae(
                model_input[i: i + batch_size, 0], y_recon, z_mean, z_log_var
            )
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        model.eval()
        v2 = np.column_stack([x_train, y_train, z_train])
        pred_input = torch.column_stack(
            [
                torch.randn_like(y_train) * torch.std(y_train) + torch.mean(y_train),
                z_train,
            ]
        )
        y_pred = model(pred_input)[0].numpy()
        vp = np.column_stack([x_train, y_pred, z_train])
    return v2, vp
