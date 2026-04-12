import torch
import torch.nn as nn

class NeuroTwinLatent(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, latent_dim=64):
        super().__init__()

        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        self.fc_latent = nn.Linear(latent_dim, hidden_size)

        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, target_len=1):
        _, (hidden, _) = self.encoder(x)

        h = hidden[-1]

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        hidden_dec = self.fc_latent(z).unsqueeze(0)
        cell_dec = torch.zeros_like(hidden_dec)

        decoder_input = x[:, -1:, :]
        outputs = []

        for _ in range(target_len):
            out, (hidden_dec, cell_dec) = self.decoder(decoder_input, (hidden_dec, cell_dec))
            pred = self.fc_out(out)
            outputs.append(pred)
            decoder_input = pred

        return torch.cat(outputs, dim=1), mu, logvar