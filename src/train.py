import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ImagesDataset
from architecture import model


def train_model():
    print("Initializing training process...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model.to(device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, '..', 'data', 'training_data')

    print(f"Loading dataset from {image_dir}...")
    # Dataset and DataLoader
    dataset = ImagesDataset(image_dir=image_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Initializing loss and optimizer...")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("Starting training loop...")
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels, _, _) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:  # Log every 10 batches
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        scheduler.step()  # Step the scheduler after each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

    print("Saving the trained model...")
    model_path = os.path.join(base_dir, '..', 'models', 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    train_model()
