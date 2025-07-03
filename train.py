"""
Train fucntion
TODO

"""

import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    correct = 0
    for X_batch, y_batch in tqdm(dataloader, total=len(dataloader.dataset)/8):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device) # gpu or cpu
        optimizer.zero_grad() #remove preivous gradient
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward() #backpropagation
        optimizer.step() # update weights
        total_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation or test data.
    """
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


def train_epochs(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Train the model over multiple epochs.
    """
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs} "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        

if __name__ == "__main__":

    """
    Test training and evluate unit
    """
    # import here are in main because not usefull in our file. Just needed
    # for tests.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import SGD
    class DummyModel(nn.Module):
        def __init__(self, input_shape, num_classes):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(input_shape[0] * input_shape[1], num_classes)

        def forward(self, x):
            x = self.flatten(x)
            return self.fc(x)

    # Create dummy dataset
    X = torch.randn(20, 62, 100)         # 20 samples, 62 channels, 100 time steps
    y = torch.randint(0, 3, (20,))       # 3 classes
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4)

    # Model, loss, optimizer
    model = DummyModel((62, 100), 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    # Test 1: Train one epoch
    train_loss, train_acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
    print(f"[Test] train_one_epoch => Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # Test 2: Evaluate
    eval_loss, eval_acc = evaluate(model, dataloader, criterion, device)
    print(f"[Test] evaluate => Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}")

    # Test 3: Full training loop (just 1 epoch to test logic)
    train_epochs(model, dataloader, dataloader, criterion, optimizer, device, num_epochs=1)
    print("[Test] train_epochs => Completed 1 epoch successfully")