import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from tqdm import tqdm


# custom dataset used to load pairs for VAE training
class EvolutionDataset(Dataset):
    def __init__(self, root_dir, img_dims):
        self.root_dir = root_dir

        self.convert_tensor = transforms.ToTensor()

        self.img_dims = img_dims

        self.pairs = []

        print("Loading dataset sample names...")

        # only one image pair per pokemon pair
        for pair in tqdm(os.listdir(root_dir)):
            for pre_evolution in os.listdir(root_dir + "/" + pair + "/pre_evolution"):
                for evolved in os.listdir(root_dir + "/" + pair + "/evolved"):
                    self.pairs += [[root_dir + "/" + pair + "/evolved/" + evolved, root_dir + "/" + pair + "/pre_evolution/" + pre_evolution]]
                    break


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        evolved_image = self.convert_tensor(Image.open(self.pairs[idx][0]).resize(self.img_dims))*2-1
        pre_evolution_image = self.convert_tensor(Image.open(self.pairs[idx][1]).resize(self.img_dims))*2-1

        return {"evolved": evolved_image, "pre-evolution": pre_evolution_image}