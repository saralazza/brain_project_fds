import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

def print_metrics(TP_train, TN_train, FP_train, FN_train, TP_valid, TN_valid, FP_valid, FN_valid, TP_test, TN_test, FP_test, FN_test):

    # Calculate metrics for Train set
    train_accuracy = (TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train)
    train_precision = TP_train / (TP_train + FP_train) if (TP_train + FP_train) > 0 else 0
    train_recall = TP_train / (TP_train + FN_train) if (TP_train + FN_train) > 0 else 0
    train_f1 = (2 * train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0

    # Calculate metrics for Validation set
    valid_accuracy = (TP_valid + TN_valid) / (TP_valid + TN_valid + FP_valid + FN_valid)
    valid_precision = TP_valid / (TP_valid + FP_valid) if (TP_valid + FP_valid) > 0 else 0
    valid_recall = TP_valid / (TP_valid + FN_valid) if (TP_valid + FN_valid) > 0 else 0
    valid_f1 = (2 * valid_precision * valid_recall) / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else 0

    # Calculate metrics for Test set
    test_accuracy = (TP_test + TN_test) / (TP_test + TN_test + FP_test + FN_test)
    test_precision = TP_test / (TP_test + FP_test) if (TP_test + FP_test) > 0 else 0
    test_recall = TP_test / (TP_test + FN_test) if (TP_test + FN_test) > 0 else 0
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

    # Print results
    print(f"Train set Accuracy: {train_accuracy * 100:.4f}%")
    print(f"Train set Precision: {train_precision * 100:.4f}")
    print(f"Train set Recall: {train_recall * 100:.4f}")
    print(f"Train set F1 Score: {train_f1 * 100:.4f}\n")

    print(f"Validation set Accuracy: {valid_accuracy * 100:.4f}%")
    print(f"Validation set Precision: {valid_precision * 100:.4f}")
    print(f"Validation set Recall: {valid_recall * 100:.4f}")
    print(f"Validation set F1 Score: {valid_f1 * 100:.4f}\n")

    print(f"Test set Accuracy: {test_accuracy * 100:.4f}%")
    print(f"Test set Precision: {test_precision * 100:.4f}")
    print(f"Test set Recall: {test_recall * 100:.4f}")
    print(f"Test set F1 Score: {test_f1 * 100:.4f}")

def print_metrics_simple(TP, TN, FP, FN, title):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"------{title}------")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Precision: {precision * 100:.4f}")
    print(f"Recall: {recall * 100:.4f}")
    print(f"F1 Score: {f1 * 100:.4f}\n")

    return accuracy, precision, recall, f1

def get_metrics(model, loader):
    
    """
    Function to calculate the accuracy of the model and confusion matrix metrics.
    """
    model.eval()  # Set the model to evaluation mode
    TP, FP, TN, FN = 0, 0, 0, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():  # Disable gradient computation for efficiency
      for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        scores = model(data)
        predictions = torch.argmax(scores, dim=1)

        # Computes metrics
        TP += ((predictions == 1) & (targets == 1)).sum().item()
        FP += ((predictions == 1) & (targets == 0)).sum().item()
        TN += ((predictions == 0) & (targets == 0)).sum().item()
        FN += ((predictions == 0) & (targets == 1)).sum().item()

    return TP, TN, FP, FN

def get_accuracy(TP, TN, FP, FN):
  return (TP + TN) / (TP + TN + FP + FN)

def get_recall(TP, FN):
  return TP / (TP + FN) if (TP + FN) > 0 else 0

def get_precision(TP, FP):
  return TP / (TP + FP) if (TP + FP) > 0 else 0

def get_f1_score(precision, recall):
  return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def get_weighted_metric(m1, m2, alpha=0.5):
  return ( (alpha * m1) + ((1-alpha) * m2) )

def get_best_metric(current_metric_params, new_params, new_metric, label=''):
  # handles the first iteration where current-metric_params is just an empty tuple
  if not current_metric_params:
        print(f"Found a new best score for {label}: {new_metric}")
        return (new_params, new_metric)

  old_params, old_metric = current_metric_params
  if new_metric > old_metric:
    if len(label)>0:
      print(f"Found a new best score for {label}: {new_metric}")
    return (new_params, new_metric)

  return current_metric_params


def get_cnn_dataset_loaders(dataset_path):
    # Define transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),          # Convert images to tensor
    ])

    # Load the dataset using ImageFolder (implicitly assumes directories for class labels)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Create a DataLoader for batching the dataset
    batch_size = 64

    train_dataset_split = int(len(dataset)*0.7)
    valid_dataset_split = int(len(dataset)*0.2)
    test_dataset_split = len(dataset)-train_dataset_split-valid_dataset_split

    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_dataset_split, valid_dataset_split, test_dataset_split])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_vae_dataset_loaders(dataset_path):
    # Define transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),          # Convert images to tensor
    ])

    # Load the dataset using ImageFolder (implicitly assumes directories for class labels)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    healthy_idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx['no']]
    unhealthy_idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx['yes']]

    healthy_subset = Subset(dataset, healthy_idx)
    unhealthy_subset = Subset(dataset, unhealthy_idx)

    # Create a DataLoader for batching the dataset
    batch_size = 64

    # training ~75% healthy
    # validation ~10% healthy
    # test ~15% healhty, 50% unhealthy
    train_dataset_split = int(len(healthy_subset)*0.75)         # 1125 images in the training set
    valid_dataset_split_healthy = int(len(healthy_subset)*0.1)  # 150 healthy images for the validation set
    test_dataset_split_healthy = len(healthy_subset) - train_dataset_split - valid_dataset_split_healthy # 225 healthy images for the test set

    valid_dataset_split_unhealthy = int(len(unhealthy_subset)*0.4) # 600 unhealthy images for the validation set
    test_dataset_split_unhealthy = len(unhealthy_subset) - valid_dataset_split_unhealthy # 900 unhealthy images for the test set

    # training set: 1125 images
    # valid set:    750 images  (600 unhealthy, 150 healthy)
    # test set:     1125 images (900 unhealthy, 225 healthy)
    train_set, valid_set_healthy, test_set_healthy = torch.utils.data.random_split(healthy_subset, [train_dataset_split, valid_dataset_split_healthy, test_dataset_split_healthy])
    valid_set_unhealthy, test_set_unhealthy = torch.utils.data.random_split(unhealthy_subset, [valid_dataset_split_unhealthy, test_dataset_split_unhealthy])

    valid_set = torch.utils.data.ConcatDataset([valid_set_healthy, valid_set_unhealthy])
    test_set = torch.utils.data.ConcatDataset([test_set_healthy, test_set_unhealthy])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_merged_vae_dataset_loaders(dataset_path1, dataset_path2):
    # Define transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),          # Convert images to tensor
    ])

    # Load the dataset using ImageFolder (
    dataset1 = datasets.ImageFolder(root=dataset_path1, transform=transform)

    healthy_idx1 = [i for i in range(len(dataset1)) if dataset1.imgs[i][1] == dataset1.class_to_idx['no']]
    unhealthy_idx1 = [i for i in range(len(dataset1)) if dataset1.imgs[i][1] == dataset1.class_to_idx['yes']]

    healthy_subset1 = Subset(dataset1, healthy_idx1)
    unhealthy_subset1 = Subset(dataset1, unhealthy_idx1)

    # Load the dataset using ImageFolder 
    dataset2 = datasets.ImageFolder(root=dataset_path2, transform=transform)

    healthy_idx2 = [i for i in range(len(dataset2)) if dataset2.imgs[i][1] == dataset2.class_to_idx['no']]
    unhealthy_idx2 = [i for i in range(len(dataset2)) if dataset2.imgs[i][1] == dataset2.class_to_idx['yes']]

    healthy_subset2 = Subset(dataset2, healthy_idx2)
    unhealthy_subset2 = Subset(dataset2, unhealthy_idx2)

    merged_healthy = ConcatDataset([healthy_subset1, healthy_subset2])
    merged_unhealthy = ConcatDataset([unhealthy_subset1, unhealthy_subset2])

    # Create a DataLoader for batching the dataset
    batch_size = 64


    train_dataset_split = int(len(merged_healthy)*0.75)         # 1125 images in the training set
    valid_dataset_split_healthy = int(len(merged_healthy)*0.1)  # 150 healthy images for the validation set
    test_dataset_split_healthy = len(merged_healthy) - train_dataset_split - valid_dataset_split_healthy # 225 healthy images for the test set

    valid_dataset_split_unhealthy = int(len(merged_unhealthy)*0.4) # 600 unhealthy images for the validation set
    test_dataset_split_unhealthy = len(merged_unhealthy) - valid_dataset_split_unhealthy # 900 unhealthy images for the test set

    merged_train_set, merged_valid_set_healthy, merged_test_set_healthy = torch.utils.data.random_split(merged_healthy, [train_dataset_split, valid_dataset_split_healthy, test_dataset_split_healthy])
    merged_valid_set_unhealthy, merged_test_set_unhealthy = torch.utils.data.random_split(merged_healthy, [valid_dataset_split_unhealthy, test_dataset_split_unhealthy])

    merged_valid_set = torch.utils.data.ConcatDataset([merged_valid_set_healthy, merged_valid_set_unhealthy])
    merged_test_set = torch.utils.data.ConcatDataset([merged_test_set_healthy, merged_test_set_unhealthy])

    merged_train_loader = DataLoader(merged_train_set, batch_size=batch_size, shuffle=True)
    merged_val_loader = DataLoader(merged_valid_set, batch_size=batch_size, shuffle=False)
    merged_test_loader = DataLoader(merged_test_set, batch_size=batch_size, shuffle=False)

    return merged_train_loader, merged_val_loader, merged_test_loader

def get_cnn_synthetic_vae_dataset_loaders(dataset_path):
    # Define transformations to be applied to the images
    transform = transforms.Compose([
    #  transforms.Resize((128, 128)), ### images already are grayscale of size 128x128 
        transforms.Grayscale(),
        transforms.ToTensor(),       
    ])

    # Load the dataset using ImageFolder (implicitly assumes directories for class labels)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Create a DataLoader for batching the dataset
    batch_size = 64

    train_dataset_split = int(len(dataset)*0.7)
    valid_dataset_split = int(len(dataset)*0.2)
    test_dataset_split = len(dataset)-train_dataset_split-valid_dataset_split

    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_dataset_split, valid_dataset_split, test_dataset_split])

    vae_synth_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    vae_synth_val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    vae_synth_test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return vae_synth_train_loader, vae_synth_val_loader, vae_synth_test_loader

def get_cnn_merged_vae_dataset_loaders(dataset_path, synth_dataset_path):
    # Define transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),          # Convert images to tensor
    ])

    # load original data
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # load synth data
    synth_dataset = datasets.ImageFolder(root=synth_dataset_path, transform=transform)

    # merge datasets
    merged_dataset = ConcatDataset([dataset, synth_dataset])

    # define the split
    train_dataset_split = int(len(merged_dataset)*0.7)
    valid_dataset_split = int(len(merged_dataset)*0.2)
    test_dataset_split = len(merged_dataset)-train_dataset_split-valid_dataset_split

    # random split the merged dataset
    merged_train_dataset, merged_valid_dataset, merged_test_dataset =torch.utils.data.random_split(merged_dataset, [train_dataset_split, valid_dataset_split, test_dataset_split])

    # Create a DataLoader for batching the merged dataset
    batch_size = 64
    merged_train_loader = DataLoader(merged_train_dataset, batch_size=batch_size, shuffle=True)
    merged_valid_loader = DataLoader(merged_valid_dataset, batch_size=batch_size, shuffle=False)
    merged_test_loader = DataLoader(merged_test_dataset, batch_size=batch_size, shuffle=False)

    return merged_train_loader, merged_valid_loader, merged_test_loader

def get_reconstruction_error(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
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

    return all_reconstruction_errors, ground_truth

def get_weighted_metrics(tp, tn, fp, fn, alpha=0.5):
    accuracy = get_accuracy(tp, tn, fp, fn)
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    f1 = get_f1_score(precision, recall)

    # Compute metrics for healthy predictions
    TP_healthy = tn
    FP_healthy = fn
    TN_healthy = tp
    FN_healthy = fp

    accuracy_healthy = get_accuracy(TP_healthy, TN_healthy, FP_healthy, FN_healthy)
    precision_healthy = get_precision(TP_healthy, FP_healthy)
    recall_healthy = get_recall(TP_healthy, FN_healthy)
    f1_healthy = get_f1_score(precision_healthy, recall_healthy)

    # Compute weighted metrics
    accuracy_weighted = get_weighted_metric(accuracy, accuracy_healthy, alpha)
    precision_weighted = get_weighted_metric(precision, precision_healthy, alpha)
    recall_weighted = get_weighted_metric(recall, recall_healthy, alpha)
    f1_weighted = get_weighted_metric(f1, f1_healthy, alpha)

    return accuracy_weighted, precision_weighted, recall_weighted, f1_weighted

def encode_positions_and_normalize(X, original_dim=128):
    X_min = np.min(X, axis=(1)) # get min per image
    X_max = np.max(X, axis=(1)) 
    X_norm = (X - X_min[:, np.newaxis]) / (X_max - X_min)[:, np.newaxis]

    X_images_norm = X_norm.reshape(-1, original_dim, original_dim) # nx128x128 where n is the number of samples
    
    N, H, W = X_images_norm.shape

    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))

    # normalize position
    x_coords = np.repeat(x_coords[np.newaxis, :] / (W-1), N, 0) / 2
    y_coords = np.repeat(y_coords[np.newaxis, :] / (H-1), N, 0) / 2

    features = np.concatenate([X_images_norm[:, np.newaxis], x_coords[:, np.newaxis], y_coords[:, np.newaxis]], axis=1)  
    features = features.reshape(N, 3, -1)
    return features 

def extract_data_by_label(loader, prediction, reconstruction, errors, num_points, label, device, batch_size):
    """
    Extracts a specified number of points for a specific label from the loader.

    Args:
        loader (DataLoader): The data loader.
        prediction (Tensor): Tensor of predictions.
        reconstruction (Tensor): Tensor of reconstructions.
        errors (Tensor): Tensor of errors.
        num_points (int): Number of points to extract for the specified label.
        label (int): The label (0 or 1) to extract data for.
        device (str): Device to which tensors should be moved (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing extracted data, labels, predictions, reconstructions, and errors.
    """
    # Initialize lists to hold the extracted data
    extracted_data = []
    extracted_predictions = []
    extracted_reconstructions = []
    extracted_errors = []

    label_count = 0

    current_batch = 0

    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)

        # Find the indices of the samples with the specified label
        label_indices = (labels == label).nonzero(as_tuple=True)[0]
        unbatched_indices = label_indices + (current_batch * batch_size) # rescale the indices so that match with the batch of the loader

        # Extract data for the specified label
        if label_count < num_points:
            remaining_points = num_points - label_count
            selected_indices = label_indices[:remaining_points]

            extracted_data.append(data[selected_indices]) 
            extracted_predictions.append(prediction[unbatched_indices])  
            extracted_reconstructions.append(reconstruction[unbatched_indices]) 
            extracted_errors.append(errors[unbatched_indices])  

            label_count += len(selected_indices)

        if label_count >= num_points: # stop when we extracted enough data
            break

        current_batch += 1

    extracted_data = torch.cat(extracted_data, dim=0)
    extracted_predictions = torch.cat(extracted_predictions, dim=0)
    extracted_reconstructions = torch.cat(extracted_reconstructions, dim=0)
    extracted_errors = torch.cat(extracted_errors, dim=0)

    return extracted_data, extracted_predictions, extracted_reconstructions, extracted_errors

def plot_brains(originals, reconstructions, errors, prediction, num_images=3, images_per_row=1, title='', labels=[]):
    originals = originals.cpu().detach()  # Move images to CPU for matplotlib
    reconstructions = reconstructions.cpu().detach()
    errors = errors.cpu().detach()
    
    # Calculate the number of rows needed
    num_rows = math.ceil(num_images / images_per_row)
    
    # Adjust figure size to make images bigger
    plt.figure(figsize=(images_per_row * 10, num_rows * 4.5))  # Increase the size of each image
    
    # Add a general title for the entire plot
    plt.suptitle(title, fontsize=16)  # Set title above the subplots
    
    for i in range(num_images):
        # Original image
        plt.subplot(num_rows, images_per_row * 3, i * 3 + 1)
        
        original_image = originals[i].squeeze()
        plt.imshow(original_image.numpy())
        plt.axis("off")
        plt.title(f"Original {i + 1}")
        
        # Reconstructed image
        plt.subplot(num_rows, images_per_row * 3, i * 3 + 2)
        plt.text(64, -15, f"Predicted {prediction[i]}", ha="center", va="bottom", fontsize=14, fontweight='bold')
        reconstruction_image = reconstructions[i]
        if reconstruction_image.ndim != 2: # if the image is not plottable, ie it is inside an array
            reconstruction_image = reconstruction_image.squeeze()
        plt.imshow(reconstruction_image.numpy())
        plt.axis("off")
        plt.title(f"Reconstruction {i + 1}")
        
        # Error image (original - reconstruction)
        plt.subplot(num_rows, images_per_row * 3, i * 3 + 3)
        error_image = errors[i].squeeze()
        plt.imshow(error_image.numpy(), cmap='hot')
        plt.axis("off")
        plt.title(f"Error {i + 1}")
    
    plt.tight_layout(pad=3.0)
    plt.show()
