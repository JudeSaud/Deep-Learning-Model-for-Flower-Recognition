import argparse
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision import transforms
import time
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser(description='Predicting the name of a flower from an image with the probability of that name')
parser.add_argument('image_path', help='Path to the image')
# parser.add_argument('checkpoint', help='Path to the saved network checkpoint')
parser.add_argument('save_path', type=str, help='Provide path to the file of the trained model')
parser.add_argument('--top_k', type=int, default=1, help='Number of most likely classes to return')
parser.add_argument('--category_names', default='cat_to_name.json', help='Path to the file mapping categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for performing inference')

args = parser.parse_args()

device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError('Model arch error.')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

checkpoint_path = args.save_path
model = load_checkpoint(checkpoint_path)
model.to(device)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    img = Image.open(image)

    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img)

    return img_tensor


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()
    with torch.no_grad():
        img_tensor = process_image(image_path)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)  # Move tensor to GPU if available
        output = model(img_tensor)
        ps = torch.exp(output)
        probs, indices = ps.topk(topk, dim=1)

        class_to_idx = model.class_to_idx
        idx_to_class = {idx: class_ for class_, idx in class_to_idx.items()}

        classes = [idx_to_class[idx.item()] for idx in indices[0]]

        return probs[0].cpu().numpy(), classes  # Move tensor back to CPU


image_path = args.image_path
top_k = args.top_k

probs, classes = predict(image_path, model, topk=top_k)

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[key] for key in classes]
    print("Class name:")
    print(names)

print("Class number:")
print(classes)
print("Probability:")
for idx, item in enumerate(probs):
    probs[idx] = round(item*100, 2)
print(probs)