import os
import torch
from torch.utils.data import DataLoader

from utils.load_dataset import load
from utils.preprocess import process_images, load_preprocessed_image
from utils.mri_dataset import MRIDataset
from VAE.vae import train_loop_vae, generate_images

if __name__ == '__main__':

    path_dataset = "../dataset"

    # Load image from original dataset
    image_paths_no, image_paths_yes = load(path_dataset)
    print(f"Number of positive images: {len(image_paths_yes)}")
    print(f"Number of negative images: {len(image_paths_no)}")

    # Preprocess images and save into a local path
    output_folder = './preprocessing/no/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        process_images(image_paths_no, output_folder)

    output_folder = './preprocessing/yes/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        process_images(image_paths_yes, output_folder)
    
    # Load preprocessed images
    image_paths_yes_prepocessed, image_paths_no_prepocessed = load_preprocessed_image("preprocessing")
    labels_yes_preprocessed = torch.ones(len(image_paths_yes_prepocessed), 1)
    labels_no_preprocessed = torch.zeros(len(image_paths_no_prepocessed), 1)
    print(f"Number of positive preprocessed images: {len(image_paths_yes_prepocessed)}")
    print(f"Number of negative preprocessed images: {len(image_paths_no_prepocessed)}")

    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset_no = MRIDataset(image_paths_no_prepocessed, labels_no_preprocessed)
    dataset_yes = MRIDataset(image_paths_yes_prepocessed, labels_yes_preprocessed)
    data_loader_no = DataLoader(dataset_no, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_yes = DataLoader(dataset_yes, batch_size=BATCH_SIZE, shuffle=True)

    # Train VAE model on positive samples
    print("Start training VAE on positive samples")
    checkpoint_dir = '../generation_checkpoints/vae/yes' 
    os.makedirs(checkpoint_dir, exist_ok=True) 
    vae_yes = train_loop_vae(device, data_loader_yes, checkpoint_dir)
    print("Finish training VAE on positive samples")

    # Generate 1500 positive images and save them into a local folder
    save_dir = '../vae_dataset/yes/' 
    os.makedirs(save_dir, exist_ok=True) 
    save_dir = f"{save_dir}yes_"
    generate_images(vae_yes, device, save_dir)
    print(f"Save positive generated images in {save_dir}")

    # Train VAE model on negative samples
    print("Start training VAE on negative samples")
    checkpoint_dir = '../generation_checkpoints/vae/no' 
    os.makedirs(checkpoint_dir, exist_ok=True) 
    vae_no = train_loop_vae(device, data_loader_no, checkpoint_dir)
    print("Finish training VAE on negative samples")

    # Generate 1500 negative images and save them into a local folder
    save_dir = '../vae_dataset/no/' 
    os.makedirs(save_dir, exist_ok=True)
    save_dir = f"{save_dir}no_"
    generate_images(vae_no, device, save_dir)
    print(f"Save negative generated images in {save_dir}")
