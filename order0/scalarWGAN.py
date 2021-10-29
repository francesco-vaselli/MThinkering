# TODO: understand models outputs
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from scipy.stats import norm, normaltest
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from tqdm import tqdm

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

nz = 6

nc = 2

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, 32),
            nn.ReLU(True),

            nn.Linear(32, 64),
            nn.Dropout(p=0.2),
            nn.ReLU(True),

            nn.Linear(64, 32),
            nn.Dropout(p=0.2),
            nn.ReLU(True),

            nn.Linear(32, 16),
            nn.Dropout(p=0.2),
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
            nn.Linear(nc, 64),
            nn.ReLU(True),

            nn.Linear(64, 64),
            nn.ReLU(True),

            nn.Linear(64, 64),
            nn.ReLU(True),

            nn.Linear(64, 16),
            nn.ReLU(True),

            nn.Linear(16, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    # optionally, initialize models weights
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data)


def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
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
    rng = default_rng()
    # load and batch training data
    dataset = np.load('order0/dataset.npy')
    # [0:1000000]
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512,
                                         shuffle=True, num_workers=2)
    # test = dataset[0][:, 1]
    # plt.hist(test, bins=25, density=True, alpha=0.6)
    # plt.show()

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    print(netG, '\n')

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD, '\n')

    # Setup Adam optimizers for both G and D, optionally set lower lr, eg lr = 0.0005
    optimizerD = optim.Adam(netD.parameters())
    optimizerG = optim.Adam(netG.parameters())

    num_epochs = 3
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
            data = data.clone().detach().requires_grad_(True)
            # torch.tensor(data, dtype=torch.float)
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = torch.mean(output)
            # Calculate gradients for D in backward pass
            # errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors (from uniform)
            noise = torch.rand((b_size, 5), dtype=torch.float32)
            noise = torch.column_stack((data[:, 0], noise)).to(device)
            # Generate fake batch with G
            fake = netG(noise)
            # Classify all fake batch with Dc
            input_fake = torch.column_stack((real_cpu[:, 0], fake))
            output = netD(input_fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = torch.mean(output)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            # Add the gradients from the all-real and all-fake batches
            # Calculate W-div gradient penalty
            gradient_penalty = calculate_gradient_penalty(netD,
                                                              real_cpu, input_fake,
                                                              device)

            errD = -errD_real + errD_fake + gradient_penalty * 1
            errD.backward()
            # Update D
            optimizerD.step()

            # Train the generator every n iterations
            if (i + 1) % 10 == 0:

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                ## Train with all-fake batch
                # Generate fake batch with G
                fake = netG(noise)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                input_fake = torch.column_stack((real_cpu[:, 0], fake))
                output = netD(input_fake).view(-1)
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
    test_number = torch.full((100000, ), 10., dtype=torch.float, device=device)
    test_noise = torch.rand((100000, 5), dtype=torch.float32, device=device)
    input_test = torch.column_stack((test_number, test_noise))

    test_output = netG(input_test).cpu().detach().numpy()
    k2, p = normaltest(test_output)
    print('\n', 'p-value for normality =', p)
    
    plt.hist(test_output, bins=100, density=True, alpha=0.6, color='b')
    # Plot the expected PDF against the generator outputs
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, 10, 1)
  
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()
