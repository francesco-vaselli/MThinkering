import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

nz = 2

nc = 2

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, 16),
            nn.ReLU(True),

            nn.Linear(16, 16),
            nn.ReLU(True),

            nn.Linear(16, 1),
            nn.ReLU(True)

        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nc, 16),
            nn.ReLU(True),

            nn.Linear(16, 32),
            nn.ReLU(True),

            nn.Linear(32, 16),
            nn.ReLU(True),

            nn.Linear(16, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':

    print(device, '\n')
    rng = default_rng()
    # load and batch training data: 1000 batches of size 1000
    dataset = np.split(np.load('order0/dataset.npy'), 1000)

    netG = Generator(ngpu).to(device)
    print(netG, '\n')

    netD = Discriminator(ngpu).to(device)
    print(netD, '\n')

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters())
    optimizerG = optim.Adam(netG.parameters())

    num_epochs = 3
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        for i, data in enumerate(dataset):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            data = torch.tensor(data, dtype=torch.float)
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors(from uniform)
            noise = rng.random(b_size,)
            noise = np.column_stack((data[:, 0], noise))
            noise = torch.tensor(noise, dtype=torch.float, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with Dc
            input_fake = torch.column_stack((real_cpu[:, 0], fake))
            output = netD(input_fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(input_fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataset),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1


    test_number = torch.full((10000, ), 10., dtype=torch.float, device=device)
    test_noise = torch.rand(10000, dtype=torch.float32, device=device)
    input_test = torch.column_stack((test_number, test_noise))

    test_output = netG(input_test).to('cpu')
    print(test_output)

