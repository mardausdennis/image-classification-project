import torch
import os
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from dataset import ImagesDataset
from architecture import model


def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, '..', 'data', 'training_data')
    val_indices_path = os.path.join(base_dir, '..', 'data', 'validation_indices.npy')

    val_indices = np.load(val_indices_path)

    # Load the model
    model_path = os.path.join(base_dir, '..', 'models', 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dataset = ImagesDataset(image_dir=image_dir, width=100, height=100, dtype=int)
    val_dataset = Subset(test_dataset, val_indices)

    test_dl = DataLoader(dataset=val_dataset, shuffle=False, batch_size=len(val_dataset))

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y, _, _ in test_dl:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    print(f'Accuracy of the model on the validation images: {accuracy:.2f}')


if __name__ == '__main__':
    evaluate_model()
