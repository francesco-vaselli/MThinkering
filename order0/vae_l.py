import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import KBinsDiscretizer, label_binarize, LabelBinarizer
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from scipy.stats import norm, normaltest, wasserstein_distance, beta


bin_num = 100

bs = 350


class VAE(LightningModule):
    def __init__(self, enc_out_dim=200, latent_dim=30, lr = 5e-5):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = nn.Sequential(nn.Linear(bin_num, 20), nn.ReLU(),
                        nn.Linear(20, 64), nn.ReLU(),
                        nn.Linear(64, 200), nn.ReLU(),
                        nn.Linear(200, 100), nn.ReLU(),
                        nn.Linear(100, enc_out_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 200), nn.ReLU(),
                        nn.Linear(200, 200), nn.ReLU(),
                        nn.Linear(200, 64), nn.ReLU(),
                        nn.Linear(64, 20), nn.ReLU(),
                        nn.Linear(20, bin_num))


        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        self.learning_rate = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def loss_function(self, recon_x, x, mu, logvar):
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        recon_loss = loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss  + KLD

    def training_step(self, batch, batch_idx):
        x = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # loss
        loss = self.loss_function(x_hat, x, mu, log_var)

        self.log_dict({
            'loss': loss
        })

        return loss

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return self.decoder(z), mu, log_var



def train(model, data):
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(data, batch_size=bs,
                                         shuffle=True, num_workers=4, pin_memory=True)
    trainer = Trainer(log_every_n_steps=10, gpus=1, max_epochs=200, auto_lr_find = True)
    trainer.fit(model, dataloader)


if __name__ == '__main__':

    vae = VAE()

    # data 
    rng = default_rng(43)
    # load and batch training data
    # dataset = np.load('order0/dataset.npy')[999:1998, 1]
    b1 = (1 - beta.rvs(a=17, b=1, size=10000))
    b2 = (beta.rvs(a=17, b=1, size=10000))
    dataset_b = np.concatenate((b1, b2))
    rng.shuffle(dataset_b)
    # dataset_g = rng.normal(10, 1, 10000)
    dataset_b = np.array(dataset_b, dtype=np.float32)
    est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
    X = est.fit_transform(dataset_b.reshape(-1, 1))
    lb = LabelBinarizer()
    X =  np.array(lb.fit_transform(X), dtype=np.float32)
    train(vae, X)

    # Z COMES FROM NORMAL(0, 1)
    num_preds = 1000
    # p = torch.distributions.Normal(torch.zeros(1), torch.ones(1))
    #z = p.rsample((num_preds, 20)).to(device)
    enc_dim = 30
    z = torch.randn(num_preds, enc_dim).to(vae.device)

    with torch.no_grad():
        pred = vae.decoder(z).cpu().numpy()
        pred = np.reshape(pred, (1000, bin_num))
        pred = lb.inverse_transform(pred)
        # encm, encs = model.encode(torch.tensor(dataset_b[0:1000]).view(-1, 1000).to(device))
        # encz = model.reparameterize(encm, encs).cpu().numpy()
        # encz = np.reshape(encz, (1000,))
        # pred = torch.special.logit(pred, eps=1e-8).cpu()
        recon, _, _ = vae(torch.tensor(X[0:1000]).to(vae.device))
        # recon = torch.special.logit(recon, eps=1e-8)
        recon = recon.cpu().numpy()
        recon = np.reshape(recon, (1000, bin_num))
        print(recon)
        recon = lb.inverse_transform(recon)
        recon = est.inverse_transform(recon.reshape(-1, 1))
        # recon = torch.special.logit(recon, eps=1e-8).cpu()

   
    plt.hist(recon, bins=50)
    plt.show()