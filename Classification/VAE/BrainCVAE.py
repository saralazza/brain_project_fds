import torch
from torch import nn
import numpy as np
import os
from tqdm import tqdm
import utils
from sklearn.cluster import KMeans

class BrainCVAE(nn.Module):
  def __init__(self, latent_dim):
    super().__init__()
    self.latent_dim = latent_dim
    self.encoder = self.BrainEncoder(self.latent_dim)
    self.decoder = self.BrainDecoder(self.latent_dim)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def forward(self, x):
    mean, log_var = self.encoder(x)
    epsilon = torch.randn(size=mean.shape).to(self.device)
    z = epsilon * torch.exp(log_var * .5) + mean # denormalize the sampled data
    z = self.decoder(z)
    return z, mean, log_var

  @staticmethod
  def vae_loss(recon_loss, mu, logvar):
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0)
    return recon_loss + KLD

  class BrainEncoder(nn.Module):
    def __init__(self, latent_dim):
      super().__init__()
      self.latent_dim = latent_dim

      # Dout = (Din - kernel_size + 2*padding)/stride + 1
      # => (128 - 3 + 2)/2 +1 => 64
      self.vae_conv1 = nn.Sequential(
          nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
          nn.ReLU(),
      )

      # Dout = (Din - kernel_size + 2*padding)/stride + 1
      # => (64 - 3 + 2)/2 +1 => 32
      self.vae_conv2 = nn.Sequential(
          nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
          nn.ReLU(),
          nn.Flatten()
      )

      self.fc_mean = nn.Linear(32*32*64, latent_dim)
      self.fc_log_var = nn.Linear(32*32*64, latent_dim)

    def forward(self, x):
      x = self.vae_conv1(x)
      x = self.vae_conv2(x)
      mean, log_var = self.fc_mean(x), self.fc_log_var(x)
      return mean, log_var

  class BrainDecoder(nn.Module):
    def __init__(self, latent_dim):
      super().__init__()
      self.latent_dim = latent_dim

      self.linear = nn.Sequential(
          nn.Linear(self.latent_dim, 32*32*64),
          nn.ReLU(),
          nn.Unflatten(1, (64, 32, 32))
      )

      # Dout = (Din-1)*stride - 2*padding + (kernel_size-1) + 1
      # => (32-1)*1 - 0 + 33 = 64
      self.convT_1 = nn.Sequential(
          nn.ConvTranspose2d(64, 32, kernel_size=33),
          nn.ReLU()
      )

      # => (64-1)*1 - 0 + 65 = 128
      self.convT_2 = nn.ConvTranspose2d(32, 1, kernel_size=65)

    def forward(self, z):
      z = self.linear(z)
      z = self.convT_1(z)
      z = self.convT_2(z)
      return z
    
def train_vae(model, train_loader, num_epochs, criterion, optimizer, scheduler, checkpoint_name):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # keeps track of the losses
  losses = []

  # Training the model
  for epoch in range(num_epochs):
      ######### TRAINING ##########
      model.train()
      running_loss = 0  # To track loss for this epoch

      # Using tqdm for the progress bar
      loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

      for batch_idx, (data, _) in loop:
          # Get data to cuda if possible
          data = data.to(device=device)
          #targets = targets.to(device=device)

          # Forward pass
          reconstruction, mean, log_var = model(data)

          loss = criterion(reconstruction, data)
          loss = BrainCVAE.vae_loss(loss, mean, log_var)

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

      if epoch == 0:
          # create directory if not exist
          os.makedirs("checkpoint", exist_ok=True)
          best_loss = avg_loss
          torch.save(model.state_dict(), f"checkpoint/{checkpoint_name}.pth")
      elif avg_loss < best_loss:
          best_loss = avg_loss
          torch.save(model.state_dict(), f"checkpoint/{checkpoint_name}.pth")

def validate_brain_cvae(model, val_loader, anomaly_threshold_range=(0.05, 0.95), tumor_threshold_range=(0.05, 0.95), alpha=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    start_anomaly_threshold, end_anomaly_threshold = anomaly_threshold_range
    anomaly_thresholds = np.linspace(start_anomaly_threshold, end_anomaly_threshold, num=100)

    start_tumor_threshold, end_tumor_threshold = tumor_threshold_range
    tumor_thresholds = np.linspace(start_tumor_threshold, end_tumor_threshold, num=100)

    # store reconstruction errors and ground truths
    all_reconstruction_errors = []
    ground_truth = []

    # compute reconstruction error
    with torch.no_grad():
      for data, targets in val_loader:
        data = data.to(device=device)
        targets = targets.to(device=device)

        reconstruction, mean, log_var = model(data)

        # get the absolute difference Manhattan Distance
        reconstruction_error = torch.abs(data - reconstruction).view(targets.size()[0], -1)
        # normalize errors from 0 to 1
        batch_min = torch.min(reconstruction_error, dim=1).values
        batch_max = torch.max(reconstruction_error, dim=1).values
        reconstruction_error = ((reconstruction_error.T - batch_min) / (batch_max - batch_min)).T

        # store the current reconstruction
        all_reconstruction_errors.append(reconstruction_error)
        ground_truth.append(targets)

    # concatenate the reconstructions and labels across all batches
    all_reconstruction_errors = torch.cat(all_reconstruction_errors, dim=0).to(device)
    ground_truth = torch.cat(ground_truth, dim=0).to(device)

    # getting ready to store the best hyperparameters for each metric
    best_accuracy_with_params = ()
    best_precision_with_params = ()
    best_recall_with_params = ()
    best_f1_with_params = ()

    # iterate over the hyperparameters and save the best hyperparameters
    for anomaly_threshold in anomaly_thresholds:
      for tumor_threshold in tumor_thresholds:
        # observe pixels that are above a certain threshold
        anomaly_reconstruction_error = all_reconstruction_errors >= anomaly_threshold
        # compute the tumor index
        tumor_index = anomaly_reconstruction_error.sum(dim=1) / anomaly_reconstruction_error.size(1)
        # the brain is not healthy if we have a number of anomalies above tumor_threshold
        predictions = (tumor_index >= tumor_threshold).long()

        index_healthy = (ground_truth == 0)
        index_unhealthy = (ground_truth == 1)

        predictions_healthy = predictions == 0
        predictions_unhealthy = predictions == 1

        # compute the metrics
        TP = ((predictions_unhealthy) & (index_unhealthy)).sum().item()
        FP = ((predictions_unhealthy) & (index_healthy)).sum().item()
        TN = ((predictions_healthy) & (index_healthy)).sum().item()
        FN = ((predictions_healthy) & (index_unhealthy)).sum().item()

        accuracy = utils.get_accuracy(TP, TN, FP, FN)
        precision = utils.get_precision(TP, FP)
        recall = utils.get_recall(TP, FN)
        f1 = utils.get_f1_score(precision, recall)

        # compute also the metrics in function to the non-tumors
        TP_healthy = TN
        FP_healthy = FN
        TN_healthy = TP
        FN_healthy = FP

        accuracy_healthy = utils.get_accuracy(TP_healthy, TN_healthy, FP_healthy, FN_healthy)
        precision_healthy = utils.get_precision(TP_healthy, FP_healthy)
        recall_healthy = utils.get_recall(TP_healthy, FN_healthy)
        f1_healthy = utils.get_f1_score(precision_healthy, recall_healthy)

        # compute the weighted metrics
        accuracy_weighted = utils.get_weighted_metric(accuracy, accuracy_healthy, alpha)
        precision_weighted = utils.get_weighted_metric(precision, precision_healthy, alpha)
        recall_weighted = utils.get_weighted_metric(recall, recall_healthy, alpha)
        f1_weighted = utils.get_weighted_metric(f1, f1_healthy, alpha)

        # save the best metrics
        best_accuracy_with_params = utils.get_best_metric(best_accuracy_with_params, (anomaly_threshold, tumor_threshold), accuracy_weighted)
        best_precision_with_params = utils.get_best_metric(best_precision_with_params, (anomaly_threshold, tumor_threshold), precision_weighted)
        best_recall_with_params = utils.get_best_metric(best_recall_with_params, (anomaly_threshold, tumor_threshold), recall_weighted)
        best_f1_with_params = utils.get_best_metric(best_f1_with_params, (anomaly_threshold, tumor_threshold), f1_weighted)

    return best_accuracy_with_params, best_precision_with_params, best_recall_with_params, best_f1_with_params

def test_brain_cvae(model, test_loader, anomaly_threshold, tumor_threshold, alpha, title):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  reconstructions = torch.tensor([]).to(device)
  errors = torch.tensor([]).to(device)
  prediction = torch.tensor([]).to(device)
  ground_truth = torch.tensor([]).to(device)

  model.eval()

  with torch.no_grad():
    loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    for batch_idx, (data, targets) in loop:
      data = data.to(device=device)
      targets = targets.to(device=device)

      reconstruction, mean, log_var = model(data)

      reconstructions = torch.cat((reconstructions, reconstruction))

      # get the absolute difference (manhatthan distance)
      reconstruction_error = torch.abs(data - reconstruction).view(targets.size()[0], -1)
      # normalize error from 0 to 1
      image_min = torch.abs(torch.min(reconstruction_error, dim=1).values)
      image_max = torch.abs(torch.max(reconstruction_error, dim=1).values)
      reconstruction_error = ((reconstruction_error.T + image_min) / (image_max + image_min)).T

      # observe all the pixels that are above a certain threshold
      reconstruction_error = reconstruction_error >= anomaly_threshold
      errors = torch.cat((errors, reconstruction_error.view(data.size())))

      # compute the tumor index for each image
      tumor_index = reconstruction_error.sum(dim=(1)) / reconstruction_error.size()[1]
      # the brain is not healthy if we have a number of anomalies above tumor_threshold
      is_unhealthy = (tumor_index >= tumor_threshold).long()

      prediction = torch.cat((prediction, is_unhealthy))
      ground_truth = torch.cat((ground_truth, targets))

    index_healthy = (ground_truth == 0)
    index_unhealthy = (ground_truth == 1)

    predictions_healthy = prediction == 0
    predictions_unhealthy = prediction ==1

    # compute the metrics
    tp = ((predictions_unhealthy) & (index_unhealthy)).sum().item()
    fp = ((predictions_unhealthy) & (index_healthy)).sum().item()
    tn = ((predictions_healthy) & (index_healthy)).sum().item()
    fn = ((predictions_healthy) & (index_unhealthy)).sum().item()

    accuracy = utils.get_accuracy(tp, tn, fp, fn)
    precision = utils.get_precision(tp, fp)
    recall = utils.get_recall(tp, fn)
    f1 = utils.get_f1_score(precision, recall)

    # compute also the metrics in function to the non-tumors
    tp_healthy = tn
    fp_healthy = fn
    tn_healthy = tp
    fn_healthy = fp

    accuracy_healthy = utils.get_accuracy(tp_healthy, tn_healthy, fp_healthy, fn_healthy)
    precision_healthy = utils.get_precision(tp_healthy, fp_healthy)
    recall_healthy = utils.get_recall(tp_healthy, fn_healthy)
    f1_healthy = utils.get_f1_score(precision_healthy, recall_healthy)

    # compute the weighted metrics
    accuracy_weighted = utils.get_weighted_metric(accuracy, accuracy_healthy, alpha)
    precision_weighted = utils.get_weighted_metric(precision, precision_healthy, alpha)
    recall_weighted = utils.get_weighted_metric(recall, recall_healthy, alpha)
    f1_weighted = utils.get_weighted_metric(f1, f1_healthy, alpha)

    print(f"------{title}------")
    print(f"Train set Accuracy: {accuracy_weighted * 100:.4f}%")
    print(f"Train set Precision: {precision_weighted * 100:.4f}%")
    print(f"Train set Recall: {recall_weighted * 100:.4f}%")
    print(f"Train set F1 Score: {f1_weighted * 100:.4f}%\n")

  return prediction, ground_truth, reconstructions, errors

def test_brain_cvae_clustering(reconstruction_error, ground_truths, anomaly_threshold, alpha, title='', original_dim=128):

    anomaly_reconstruction_error = reconstruction_error >= anomaly_threshold

    kmeans = KMeans(n_clusters=2, random_state=0) # 2 clusters: tumorous and non-tumorous
    kmeans.fit_predict(anomaly_reconstruction_error)

    # Improved cluster-to-label mapping using distances to centroids
    centroids = kmeans.cluster_centers_
    # given the centroids, calculate the distance for each point (reconstruction error image)
    distances_to_c0 = np.linalg.norm(reconstruction_error - centroids[0], axis=1)
    distances_to_c1 = np.linalg.norm(reconstruction_error - centroids[1], axis=1)

    # apply the current alpha to the distance thresholding
    dist_diff = np.abs(distances_to_c0 - distances_to_c1)
    threshold = alpha * np.max(dist_diff)

    # compute the new labels in function to the new distance
    new_labels = (dist_diff < threshold).astype(int)

    # metrics
    index_healthy = (ground_truths == 0)
    index_unhealthy = (ground_truths == 1)

    tp = ((new_labels == 1) & index_unhealthy).sum()
    fp = ((new_labels == 1) & index_healthy).sum()
    tn = ((new_labels == 0) & index_healthy).sum()
    fn = ((new_labels == 0) & index_unhealthy).sum()

    accuracy_weighted, precision_weighted, recall_weighted, f1_weighted = utils.print_metrics_simple(tp, tn, fp, fn, title)

    return accuracy_weighted, precision_weighted, recall_weighted, f1_weighted