import os

import tensorflow as tf

from PIL import Image

import numpy as np

from tqdm import tqdm


IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image):
  image = random_jitter(image)
  image = normalize(image)
  return image

# custom dataset used to load pairs for VAE training
class EvolutionDataset():
    def __init__(self, root_dir, img_dims):
        self.root_dir = root_dir

        self.convert_tensor = tf.convert_to_tensor

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
        evolved_image = self.convert_tensor(np.array(Image.open(self.pairs[idx][0]).resize(self.img_dims)))
        pre_evolution_image = self.convert_tensor(np.array(Image.open(self.pairs[idx][1]).resize(self.img_dims)))

        return {"evolved": tf.reshape(preprocess_image_train(evolved_image), (1, 256, 256, 3)), "pre-evolution": tf.reshape(preprocess_image_train(pre_evolution_image), (1, 256, 256, 3))}