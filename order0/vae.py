# TODO p(z) is not following N(0, I), thus reconstruction is nontrivial
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from scipy.stats import norm, normaltest, wasserstein_distance, beta
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.optim as optim
from tqdm import tqdm
import ot
from pathlib import Path
import os


ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

l_dim = 500
enc_dim = 1

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(1, l_dim)
        self.fc21 = nn.Linear(l_dim, enc_dim)
        self.fc22 = nn.Linear(l_dim, enc_dim)
        self.fc3 = nn.Linear(enc_dim, l_dim)
        self.fc4 = nn.Linear(l_dim, 1)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return (self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1))
        z = self.reparameterize(mu, logvar)
        print(z)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, logscale):
    loss = torch.nn.MSELoss(reduction='sum')
    recon_loss = loss(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # scale = torch.exp(logscale)
    #scale = torch.ones((1, 784), scale.values(), device=device)
    
    # mean = recon_x
    # dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    # log_pxz = dist.log_prob(x)
    # GLL = log_pxz.sum()
    
    #reco_loss = torch.nn.GaussianNLLLoss(reduction = 'sum')
    #GLL = reco_loss(recon_x, x.view(-1, 784), scale)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss  + KLD


def train(epoch, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, model.log_scale)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


if __name__ == '__main__':

    print(device, '\n')
    rng = default_rng(43)
    # load and batch training data
    # dataset = np.load('order0/dataset.npy')[999:1998, 1]
    b1 = (1 - beta.rvs(a=17, b=1, size=30000))
    b2 = (beta.rvs(a=17, b=1, size=30000))
    dataset_b = np.concatenate((b1, b2))
    rng.shuffle(dataset_b)
    dataset_g = rng.normal(10, 1, 30000)
    dataset_b = np.array(b1, dtype=np.float32)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset_b, batch_size=15000,
                                         shuffle=True, num_workers=4, pin_memory=True)

    
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 150
    for epoch in range(1, epochs + 1):
        train(epoch, dataloader)

    # Z COMES FROM NORMAL(0, 1)
    num_preds = 1000
    p = torch.distributions.Normal(torch.zeros(1), torch.ones(1))
    #z = p.rsample((num_preds, 20)).to(device)
    z = torch.randn(num_preds, enc_dim).to(device)

    # SAMPLE data
    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()
        pred = np.reshape(pred, (1000,))
        # encm, encs = model.encode(torch.tensor(dataset_b[0:1000]).view(-1, 1000).to(device))
        # encz = model.reparameterize(encm, encs).cpu().numpy()
        # encz = np.reshape(encz, (1000,))
        # pred = torch.special.logit(pred, eps=1e-8).cpu()
        recon, _, _ = model(torch.tensor(dataset_b[0:1000]).to(device))
        recon = recon.cpu().numpy()
        recon = np.reshape(recon, (1000,))
        # recon = torch.special.logit(recon, eps=1e-8).cpu()

    # print(recon)
    plt.hist(dataset_b[:1000], bins=50)
    plt.hist(recon, bins=50)
    plt.show()
