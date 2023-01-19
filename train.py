import torch

from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision.utils import save_image

from tqdm import tqdm

from config import Config
from utils import VAE
from custom_dataset import EvolutionDataset


# mean squared error loss object declaration
reconstruction_loss = nn.MSELoss()

def vae_loss(predicted, evolved, mean, log_var):
    reproduction_loss = reconstruction_loss(predicted, evolved)
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train():
    device = torch.device("cuda")

    config = Config()

    vae = VAE(config, device)
    vae.to(device)


    dataset = EvolutionDataset(config.data_dir, config.img_dims)
    print("Dataset size: " + str(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


    # optimizer object declaration to update model parameters
    optimizer = optim.Adam(vae.parameters(), lr=config.lr, betas=(config.beta, 0.9))

    #seeded sample
    seeded_batch = next(iter(dataloader))
    pre_evolution_seeded_sample = seeded_batch["pre-evolution"]
    evolved_seeded_sample = seeded_batch["evolved"]

    # saves original image
    save_image(pre_evolution_seeded_sample*0.5+0.5, "output/pre_evolution.jpg")
    save_image(evolved_seeded_sample*0.5+0.5, "output/evolved.jpg")

    # saving untrained model
    print("Saving untrained model...")
    torch.save(vae, "saved_models/untrained")


    for epoch in range(config.epochs):
        print("EPOCH: " + str(epoch))
        if epoch % 5 == 0:
            # seeded test output
            with torch.no_grad():
                output = vae(pre_evolution_seeded_sample.to(device))[0]
                save_image(output*0.5+0.5, "output/epoch_" + str(epoch) + ".jpg")

        for i, batch in tqdm(enumerate(dataloader)):
            # normalized between -1->1 in custom dataset class
            evolved = batch["evolved"]
            pre_evolution = batch["pre-evolution"]

            # ensures gradients are reset for every training step
            optimizer.zero_grad()

            predicted, mean, log_var = vae(pre_evolution.to(device))

            # 1 minus the similarity output
            loss_output = vae_loss(predicted, evolved.to(device), mean, log_var)

            loss_output.backward()

            optimizer.step()
        
        print("Epoch loss: " + str(loss_output))
    
    # saving model after training
    print("Saving final model...")
    torch.save(vae, "saved_models/model1")


if __name__ == '__main__':
    train()