import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load ImageNet classes
with open("models/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Load pretrained ResNet
model = models.resnet18(pretrained=True)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image, topk):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_idxs = probs.topk(topk)
    predictions = [
        {"label": labels[idx], "prob": float(prob)}
        for prob, idx in zip(top_probs, top_idxs)
    ]
    return predictions

