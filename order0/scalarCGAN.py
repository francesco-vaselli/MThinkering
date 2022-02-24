import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from scipy.stats import norm, normaltest, wasserstein_distance, beta
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from tqdm import tqdm
import ot
from pathlib import Path
import os

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

nz = 128

nc = 2

n_classes = 2

embedding_dim = 50

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.label_conditioned_generator = nn.Sequential(nn.Embedding(2, embedding_dim),
                      nn.Linear(embedding_dim, 16))
        
    
        self.latent = nn.Sequential(nn.Linear(nz, 128),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.main = nn.Sequential(
            nn.Linear(nz+16, 128),
            nn.LeakyReLU(0.2,True),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),

            nn.Linear(128, 2),
            # nn.ReLU(True)  no activation :O
            
        )

    def forward(self, x):
        noise_vector, label = x
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 16)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 128)
        concat = torch.cat((latent_output, label_output), dim=1)
        return self.main(concat)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.label_condition_disc = nn.Sequential(nn.Embedding(2, embedding_dim),
	                      nn.Linear(embedding_dim, nc))

        self.main = nn.Sequential(
            nn.Linear(nc+nc, 32),
            nn.LeakyReLU(0.2,True),

            nn.Linear(32, 32),
            nn.LeakyReLU(0.2,True),

            nn.Linear(32, 32),
            nn.LeakyReLU(0.2,True),

            nn.Linear(32, 32),
            nn.LeakyReLU(0.2,True),

            nn.Linear(32, 1)

        )

    def forward(self, x):
        img, label = x
        # print(img.size())
        label_output = self.label_condition_disc(label)
        label_output = label_output
        # print(label_output.size())
        concat = torch.cat((img, label_output), dim=1)
        # print(concat.size())
        return self.main(concat)


def weights_init(m):
    # optionally, initialize models weights
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data)


def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images[:, :2] + ((1 - alpha) * fake_images)).requires_grad_(True)
    
    model_interpolates = model((interpolates, real_images[:, 2].int()))
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


if __name__ == '__main__':

    print(device, '\n')
    rng = default_rng(43)
    # load and batch training data
    # dataset = np.load('order0/dataset.npy')[999:1998, 1]
    dataset_g = rng.normal(10, 1, 3000)
    b1 = np.column_stack((1 - beta.rvs(a=17, b=1, size=1500), np.zeros(1500)))
    b2 = np.column_stack((beta.rvs(a=17, b=1, size=1500), np.ones(1500)))
    dataset_b = np.concatenate((b1, b2))
    rng.shuffle(dataset_b)
    
    full = np.array(np.column_stack((dataset_g, dataset_b)), dtype=np.float32)
    
    fig = plt.figure()
    plt.plot(full[:, 0], full[:, 1], 'o', markersize=1)
    savepath = Path.home()
    file = 'test.png'
    plt.savefig(file)
    
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(full, batch_size=512,
                                         shuffle=True, num_workers=4, pin_memory=True)

    netG = Generator(ngpu).to(device)
    # netG.apply(weights_init)
    print(netG, '\n')

    netD = Discriminator(ngpu).to(device)
    # netD.apply(weights_init)
    print(netD, '\n')

    # Setup Adam optimizers for both G and D, optionally set lower lr, eg lr = 0.0005
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=[0.5, 0.9])
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=[0.5, 0.9])

    num_epochs = 1000
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in progress_bar:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            data = data.clone().detach()
            # .requires_grad_(True)
            # torch.tensor(data, dtype=torch.float)
            real_cpu = data.to(device)
            real_cpu = real_cpu.view(real_cpu.size(0), -1)
            # print(real_cpu)
            b_size = real_cpu.size(0)
            # Forward pass real batch through D
            # print(real_cpu[:, :2])
            output = netD((real_cpu[:, :2],real_cpu[:, 2].int())).view(-1)
            # Calculate loss on all-real batch
            errD_real = torch.mean(output)
            # Calculate gradients for D in backward pass
            # errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors (from uniform)
            noise = torch.randn((b_size, nz), dtype=torch.float32).to(device)
            # noise = torch.column_stack((data[:, 0], noise)).to(device)
            # Generate fake batch with G
            fake = netG((noise, real_cpu[:, 2].int()))
            # Classify all fake batch with Dc
            # input_fake = torch.column_stack((real_cpu[:, 0], fake))
            output = netD((fake.detach(), real_cpu[:, 2].int())).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = torch.mean(output)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            # Add the gradients from the all-real and all-fake batches
            # Calculate W-div gradient penalty
            gradient_penalty = calculate_gradient_penalty(netD,
                                                              real_cpu, fake,
                                                              device)

            errD = -errD_real + errD_fake + gradient_penalty * 10
            errD.backward()
            # Update D
            optimizerD.step()

            # Train the generator every n iterations
            if (i + 1) % 5 == 0:

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                ## Train with all-fake batch
                # Generate fake batch with G
                fake = netG((noise, real_cpu[:, 2].int()))
                # Since we just updated D, perform another forward pass of all-fake batch through D
                # input_fake = torch.column_stack((real_cpu[:, 0], fake))
                output = netD((fake, real_cpu[:, 2].int())).view(-1)
                # Calculate G's loss based on this output
                errG = -torch.mean(output)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                progress_bar.set_description(f"[{epoch + 1}/{num_epochs}][{i + 1}/{len(dataloader)}] "
                                                 f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                                 f"D(x): {D_x:.6f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.6f}")

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            iters += 1

    # testing the perofrmance of the generator on new inputs
    # test_number = torch.full((10000, ), 10., dtype=torch.float, device=device)
    test_noise = torch.randn((10000, nz), dtype=torch.float32, device=device)
    label_0 = torch.zeros(5000, dtype=torch.int, device=device)
    label_1 = torch.ones(5000, dtype=torch.int, device=device)
    test_labels = torch.cat((label_0, label_1), 0)
    # input_test = torch.column_stack((test_number, test_noise))

    # test statistics
    test_output = netG((test_noise, test_labels)).cpu().detach().numpy()

    rng = default_rng(44)
    g = rng.normal(10, 1, 10000)
    b1 = (1 - beta.rvs(a=17, b=1, size=5000))
    b2 = (beta.rvs(a=17, b=1, size=5000))
    b = np.concatenate((b1, b2))
    rng.shuffle(b)
    test_truth = np.array(np.column_stack((g, b)), dtype=np.float32)

    M1 = ot.dist(test_truth, test_output, metric='euclidean')
    M1 /= M1.max()
    n = 10000
    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
    EMD = ot.bregman.sinkhorn2(a, b, M1, 1, numItermax=1000)
    print('\n', 'EMD = ', EMD)
    
    fig1 = plt.figure()
    plt.plot(test_truth[:, 0], test_truth[:, 1], 'o', markersize=1, color='r', alpha=0.5)
    plt.plot(test_output[:, 0], test_output[:, 1], 'o', markersize=1, color='b', label=f'EMD={EMD}')
    plt.legend()
    plt.savefig('5e5')
    # plt.show()

    fig2 = plt.figure()
    plt.plot(G_losses, label='G')
    plt.plot(D_losses, label= 'D')
    plt.legend()
    plt.savefig('5e5loss')
    # plt.show()