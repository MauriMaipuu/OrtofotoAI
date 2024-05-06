"""
Skript is created to test how .pth model weights can be read in clean pytorch. The initial model was trained
in ArcGIS pro environment.
The reading of model weights works in this script. To implement the model on our images we have to prepare
images and process them (this part is not done).
"""

import torch
from torchvision import models, transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = 625000100

test_image = r"D:\ArcPro_katsetused\Building\res_orto\64404.tif"
output = r"D:\ArcPro_katsetused\solarPanels\script\paneltest.gpkg"

print("preparing initial model resnet50")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

print("reading trained kevad_tune1_30_bs24_vld10_unfr")
loaded_state_dict = torch.load(r"D:\ArcPro_katsetused\Test\kevad_katsetused\mudelid\kevad_tune1_30_bs24_vld10_unfr\kevad_tune1_30_bs24_vld10_unfr.pth",
                               map_location=torch.device('cuda'))
print(loaded_state_dict)

print("loading trained model into initial model")
model.load_state_dict(loaded_state_dict, strict=False)

print("creating image transformer")
data_transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()])

print(f"reading {test_image}")
image = Image.open(test_image)

print("applying transformer")
image = data_transform(image).unsqueeze(0).cuda()
# image = data_transform(image).unsqueeze(0).cpu()

print("...")
model.cuda()
# model.cpu()
model.eval()
out = model(image)

print("listing result")
# print(out)

# out.save(output)
