
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from models.modeling import VisionTransformer, CONFIGS
import os
from os.path import join

test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                            transforms.CenterCrop((448, 448)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--name", required=True, 
                    help="Name of this run. Used for monitoring.")
parser.add_argument('--data_root', type=str, default='./dataset')
parser.add_argument("--pretrained_dir", type=str, default="./pretrained_models/imagenet21k_ViT-B_16.npz",
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--pretrained_model", type=str, default='./checkpoints/test_checkpoint.bin', 
                    help="load pretrained model")
parser.add_argument("--img_size", default=448, type=int, help="Resolution size")
parser.add_argument('--split', type=str, default='non-overlap', help="Split method")
parser.add_argument('--slide_step', type=int, default=12, help="Slide step for overlap split")
parser.add_argument('--smoothing_value', type=float, default=0.0, help="Label smoothing value\n")
args = parser.parse_args()

args.pretrained_model = './checkpoints/{}/{}_checkpoint.bin'.format(args.name, args.name)

# Find GPU Sources
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Use: {}'.format(device))

# Create Classes List
class_names = []
class_file = open(join(args.data_root,'classes.txt'))
for line in class_file:
    class_names.append(line)
print('{} classes.'.format(len(class_names)))

# Prepare model
config = CONFIGS['ViT-B_16']
config.split = args.split
config.slide_step = args.slide_step

num_classes = 200

model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
model.load_from(np.load(args.pretrained_dir))
if args.pretrained_model is not None:
    pretrained_model = torch.load(args.pretrained_model)['model']
    model.load_state_dict(pretrained_model)
model.to(device)
model.eval()

print('Use pretrained model: {}'.format(args.pretrained_model))
# Create Test Dataset
annotations_file = join(args.data_root, 'testing_img_order.txt')
img_dir = join(args.data_root, 'testing_images')

# Set answer path
result_dir = './result/{}'.format(args.name)
os.makedirs(result_dir, exist_ok=True)
result_path = join(result_dir, 'answer.txt')
ans = open(result_path,'w')

with open(annotations_file) as f:
    test_images = [x.strip() for x in f.readlines()]  # all the testing images

for img in test_images:
    img_PIL = Image.open(join(img_dir, img)).convert('RGB')
    inputs = test_transform(img_PIL)
    outputs = model((inputs.unsqueeze(0)).to(device))
    _, preds = torch.max(outputs, 1)

    ans.write('{} {}'.format(img, class_names[preds]))
    # print('{} {}'.format(img, class_names[preds]))

ans.close()