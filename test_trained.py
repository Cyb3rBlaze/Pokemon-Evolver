import torch
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm

from PIL import Image


model = torch.load("./saved_models/model1", map_location=torch.device('cpu'))


# sample input
convert_tensor = transforms.ToTensor()
sample_input =  convert_tensor(Image.open("./inference_tests/test_input2.jpg").resize((128, 128)))

output = sample_input

for i in tqdm(range(1)):
    output = model(torch.reshape(output, (1, 3, 128, 128)))

save_image(output, "inference_tests/test_output2.jpg")