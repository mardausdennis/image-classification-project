import torch
import os
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import ImagesDataset
from architecture import model

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, '..', 'data', 'training_data')
    val_indices_path = os.path.join(base_dir, '..', 'data', 'validation_indices.npy')

    print(f"Loading dataset from {image_dir}...")

    transform = transforms.Compose([
        transforms.Resize((100, 100)),  # Ensure all images are resized to the same size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = ImagesDataset(image_dir=image_dir, transform=transform)
    val_indices = np.load(val_indices_path)
    val_dataset = Subset(dataset, val_indices)
    dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model_path = os.path.join(base_dir, '..', 'models', 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # eval
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy of the model on the validation images: {accuracy:.2f}')

if __name__ == '__main__':
    evaluate_model()
