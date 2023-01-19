import torch
from torch import nn, optim

# for creating dataloader
from torch.utils.data import DataLoader

from torchvision.utils import save_image

from gan.gan_config import Config
from gan.gan_utils import Generator, Discriminator
from pix2pix.custom_dataset import EvolutionDataset


normalization_stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# for saving image purposes
def denorm(image):
    return image * normalization_stats[1][0] + normalization_stats[0][0]


# training script - referenced DCGAN tutorial when neccesary
def train():
    device = torch.device("cuda")

    config = Config()

    # creating dataloader object
    dataset = EvolutionDataset(config.data_dir, config.img_dims)
    print("Dataset size: " + str(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    generator = Generator(config, device)
    generator.to(device)
    discriminator = Discriminator(config, device)
    discriminator.to(device)

    # binary cross-entropy loss object declaration
    loss = nn.BCELoss()

    # declaring optimizer objects for each submodel (beta + lr values pulled from DCGAN tutorial)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta, 0.9))
    gen_optimizer = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta, 0.9))

    # seeded input
    seeded_batch = next(iter(dataloader))
    pre_evolution_seeded_sample = seeded_batch["pre-evolution"]
    evolved_seeded_sample = seeded_batch["evolved"]

    # saves original image
    save_image(denorm(pre_evolution_seeded_sample), "output/pre_evolution.jpg")
    save_image(denorm(evolved_seeded_sample), "output/evolved.jpg")


    for epoch in range(config.epochs):
        print("EPOCH: " + str(epoch))
        if epoch % 5 == 0:
            # seeded test output
            with torch.no_grad():
                output = generator(pre_evolution_seeded_sample.to(device))
                save_image(denorm(output), "output/epoch_" + str(epoch) + ".jpg")


        # iterate through samples produced by dataloader  
        for i, batch in enumerate(dataloader):
            # normalized between -1->1 in custom dataset class
            evolved = batch["evolved"]
            pre_evolution = batch["pre-evolution"]


            total_dis_loss = 0


            # DISCRIMINATOR TRAIN STEP


            # initialize gradients of discriminator to zero to begin training step
            dis_optimizer.zero_grad()

            # train discriminator on real batch first
            true_labels = (torch.rand(evolved.shape[0], 1, device=device) * (0.1 - 0) + 0).to(device)

            true_output = discriminator((evolved).to(device))

            true_loss = loss(true_output, true_labels)

            # train discriminator on fake batch after - contains some noisy data to throw of discriminator
            false_labels = (torch.rand(pre_evolution.shape[0], 1, device=device) * (1 - 0.9) + 0.9).to(device)

            # training on generator output
            false_gen_output = generator(pre_evolution.to(device))
            false_dis_output = discriminator(false_gen_output)

            false_loss = loss(false_dis_output, false_labels)

            # total discriminator loss
            total_dis_loss = true_loss + false_loss
            total_dis_loss.backward()

            # only apply gradients to update discriminator weights
            dis_optimizer.step()


            # GENERATOR TRAIN STEP


            # initialize gradients of generator to zero to begin training step
            gen_optimizer.zero_grad()

            # train generator on mispredicted discriminator labels
            gen_labels = torch.zeros((pre_evolution.shape[0],1), dtype=torch.float).to(device)

            # train generator to trick discriminator
            false_gen_output = generator(pre_evolution.to(device))
            train_gen_dis_output = discriminator(false_gen_output)

            gen_loss = loss(train_gen_dis_output, gen_labels)
            gen_loss.backward()

            # only apply gradients to update generator weights
            gen_optimizer.step()


            if i % 200 == 0:
                print("Discriminator loss: " + str(torch.mean(total_dis_loss)))
                print("Generator loss: " + str(torch.mean(gen_loss)))
        

if __name__ == '__main__':
    train()