import os
import torch
from torch import nn
from tqdm import tqdm

class BrainCNN(nn.Module):
  def __init__(self, dropout_probability):
    super().__init__()

    self.dropout_probability = dropout_probability

    self.conv1 = nn.Sequential(
          nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self.fc = nn.Sequential(
        nn.Dropout(self.dropout_probability),
        nn.Linear(16 * 16 * 128, 1024),
        nn.ReLU(),
        nn.Dropout(self.dropout_probability),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(self.dropout_probability),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
  
def train_validation(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, checkpoint_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # keeps track of the losses
    losses = []
    losses_val = []

    # Training the model
    for epoch in range(num_epochs):
        ######### TRAINING ##########
        model.train()
        running_loss = 0  # To track loss for this epoch

        # Using tqdm for the progress bar
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

        for batch_idx, (data, targets) in loop:
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent step
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Update progress bar with loss and epoch information
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

        #scheduler
        scheduler.step(avg_loss)

        # Print loss for this epoch
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        ####### VALIDATION ########
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device=device)
                targets = targets.to(device=device)

                scores = model(data)
                loss = criterion(scores, targets)
                val_loss += loss.item()
            # Calculate average loss for the epoch
            avg_val_loss = val_loss / len(val_loader)
            losses_val.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            # if avg val_loss is better than the one before, save the model
            if epoch == 0:
                # create directory if not exist
                os.makedirs("checkpoint", exist_ok=True)
                best_loss = avg_val_loss
                torch.save(model.state_dict(), f"checkpoint/{checkpoint_name}.pth")
            elif avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), f"checkpoint/{checkpoint_name}.pth")

