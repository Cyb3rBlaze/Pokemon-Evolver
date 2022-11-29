import torch

from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision.utils import save_image

from tqdm import tqdm

from config import Config
from utils import VAE
from custom_dataset import EvolutionDataset


def train():
    device = torch.device("cuda")

    config = Config()

    vae = VAE(config, device)
    vae.to(device)


    dataset = EvolutionDataset(config.data_dir, config.img_dims)
    print("Dataset size: " + str(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


    # mean squared error loss object declaration
    loss = nn.MSELoss()
    # optimizer object declaration to update model parameters
    optimizer = optim.Adam(vae.parameters(), lr=config.lr, betas=(config.beta, 0.9))

    #seeded sample
    seeded_batch = next(iter(dataloader))
    pre_evolution_seeded_sample = seeded_batch["pre-evolution"]*0.5+0.5
    evolved_seeded_sample = seeded_batch["evolved"]*0.5+0.5

    # saves original image
    save_image(pre_evolution_seeded_sample, "output/pre_evolution.jpg")
    save_image(evolved_seeded_sample, "output/evolved.jpg")

    # saving untrained model
    print("Saving untrained model...")
    torch.save(vae, "saved_models/untrained")


    for epoch in range(config.epochs):
        print("EPOCH: " + str(epoch))
        if epoch % 5 == 0:
            # seeded test output
            with torch.no_grad():
                output = vae(pre_evolution_seeded_sample.to(device))
                save_image((output*0.5+0.5), "output/epoch_" + str(epoch) + ".jpg")

        for i, batch in tqdm(enumerate(dataloader)):
            # normalized between -1->1 in custom dataset class
            evolved = batch["evolved"]
            pre_evolution = batch["pre-evolution"]

            # ensures gradients are reset for every training step
            optimizer.zero_grad()

            vae_output = vae(pre_evolution.to(device))

            loss_output = loss(vae_output, evolved.to(device))

            loss_output.backward()

            optimizer.step()
        
        print("Epoch loss: " + str(loss_output))
    
    # saving model after training
    print("Saving final model...")
    torch.save(vae, "saved_models/model1")


if __name__ == '__main__':
    train()