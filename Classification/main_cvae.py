import torch
from torch import nn

import utils
from VAE import BrainCVAE, train_vae, validate_brain_cvae, test_brain_cvae_clustering, test_brain_cvae, reconstruction_kmeans

if __name__ == '__main__':
    # # setting up variables that will be used several times in the code
    train_loader, val_loader, test_loader = utils.get_vae_dataset_loaders('../dataset')
    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs=10
    criterion = nn.CrossEntropyLoss()

    # 1 - Train, validation and test on real data
    ### train
    brain_cvae = BrainCVAE(latent_dim)
    brain_cvae.to(device)

    checkpoint_name = 'brain_cvae'
    optimizer = torch.optim.Adam(brain_cvae.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_vae(brain_cvae, train_loader, num_epochs, criterion, optimizer, scheduler, checkpoint_name)
    ### val (based on GIAEM)
    brain_cvae = BrainCVAE(latent_dim)
    brain_cvae.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))
    brain_cvae.to(device)

    alpha = 0.5 # alpha defines the weight of the metrics computed wrt unhealthy subjects and healthy subjects ( (alpha * metric_wrt_unhealthy) + ((1-alpha) * metric_wrt_healthy) )
    best_accuracy_with_params, best_precision_with_params, best_recall_with_params, best_f1_with_params = validate_brain_cvae(brain_cvae, val_loader, alpha=alpha)

    ### latent_dim = 64
    print("Best Metrics and Hyperparameters (anomaly_threshold, tumor_threshold):")
    print("----------------------------------------------------------------------")
    print(f"Best Accuracy: {best_accuracy_with_params[1]:.4f} \t| Hyperparameters: {best_accuracy_with_params[0]}")
    print(f"Best Precision: {best_precision_with_params[1]:.4f} \t| Hyperparameters: {best_precision_with_params[0]}")
    print(f"Best Recall: {best_recall_with_params[1]:.4f} \t| Hyperparameters: {best_recall_with_params[0]}")
    print(f"Best F1 Score: {best_f1_with_params[1]:.4f} \t| Hyperparameters: {best_f1_with_params[0]}")

    ### test (based on GIAEM)
    brain_cvae_test = BrainCVAE(latent_dim)
    brain_cvae_test.to(device)
    brain_cvae_test.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))

    # use the best f1/recall hyperparams
    anomaly_threshold = 0.11363636363636363
    tumor_threshold = 0.48636363636363633

    title = 'CVAE on real data'
    pred, gt, recs, errors = test_brain_cvae(brain_cvae_test, test_loader, anomaly_threshold, tumor_threshold, alpha, title)

    ### val (based on global KMeans)
    checkpoint_name = 'brain_cvae'

    brain_cvae = BrainCVAE(latent_dim)
    brain_cvae.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))
    brain_cvae.to(device)

    reconstructions, ground_truth = utils.get_reconstruction_error(brain_cvae, val_loader)
    ### clustering.py also contains the other implementations for clustering that were scrapped
    best_params_acc, best_params_prec, best_params_recall, best_params_f1 = reconstruction_kmeans(reconstructions.cpu().numpy(), ground_truth.cpu().numpy())

    print("Best Metrics and Hyperparameters (anomaly_threshold):")
    print("----------------------------------------------------------------------")
    print(f"Best Accuracy: {best_params_acc[1]:.4f} \t| Hyperparameters: {best_params_acc[0][0]}")
    print(f"Best Precision: {best_params_prec[1]:.4f} \t| Hyperparameters: {best_params_prec[0][0]}")
    print(f"Best Recall: {best_params_recall[1]:.4f} \t| Hyperparameters: {best_params_recall[0][0]}")
    print(f"Best F1 Score: {best_params_f1[1]:.4f} \t| Hyperparameters: {best_params_f1[0][0]}")

    ### test (based on global KMeans)
    checkpoint_name = 'brain_cvae'
    
    brain_cvae_test = BrainCVAE(latent_dim)
    brain_cvae_test.to(device)
    brain_cvae_test.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))

    # use the best f1/recall hyperparams
    alpha = 0.4646464646464647
    anomaly_threshold = 0.7318181818181818

    reconstructions, ground_truth = utils.get_reconstruction_error(brain_cvae, test_loader)
    accuracy_weighted, precision_weighted, recall_weighted, f1_weighted= test_brain_cvae_clustering(reconstructions.cpu().numpy(), ground_truth.cpu().numpy(), anomaly_threshold, alpha, title='Real Data')

    ### plot reconstruction
    batch_size = 64
    num_points = 4 # number of total plots
    label=0 # first get only negative examples

    neg_data, neg_predictions, neg_reconstructions, neg_errors = utils.extract_data_by_label(
        test_loader, pred, recs, errors, num_points, label, device, batch_size
    )

    label=1 # then get only positive examples
    pos_data, pos_predictions, pos_reconstructions, pos_errors = utils.extract_data_by_label(
        test_loader, pred, recs, errors, num_points, label, device, batch_size
    )

    utils.plot_brains(pos_data, pos_reconstructions, pos_errors, pos_predictions, num_images=4, images_per_row=2, title='Positive Examples')
    utils.plot_brains(neg_data, neg_reconstructions, neg_errors, neg_predictions, num_images=4, images_per_row=2, title='Negative Examples')

    # 2.1 - Synthetic train, validation and test on real data
    vae_synth_brain_cvae = BrainCVAE(latent_dim)
    checkpoint_name = 'vae_synth_brain_cvae'
    vae_synth_train_loader, vae_synth_val_loader, vae_synth_test_loader  = utils.get_vae_dataset_loaders('../vae_dataset')

    vae_synth_brain_cvae.to(device)
    ### train
    train_vae(vae_synth_brain_cvae, vae_synth_train_loader, num_epochs, criterion, optimizer, scheduler, checkpoint_name)

    vae_synth_brain_cvae = BrainCVAE(latent_dim)
    vae_synth_brain_cvae.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))

    vae_synth_brain_cvae.to(device)

    ### validation
    alpha = 0.5 # alpha defines the weight of the metrics computed wrt unhealthy subjects and healthy subjects ( (alpha * metric_wrt_unhealthy) + ((1-alpha) * metric_wrt_healthy) )
    vae_best_accuracy_with_params, vae_best_precision_with_params, vae_best_recall_with_params, vae_best_f1_with_params = validate_brain_cvae(vae_synth_brain_cvae, val_loader, alpha=alpha)

    ### synthetic vae generated datasetlatent_dim = 64
    print("------CVAE trained on real data and validated on synthetic data------\nBest Metrics and Hyperparameters on the validation set (anomaly_threshold, tumor_threshold):")
    print("----------------------------------------------------------------------")
    print(f"Best Accuracy: {vae_best_accuracy_with_params[1]:.4f} \t| Hyperparameters: {vae_best_accuracy_with_params[0], vae_best_accuracy_with_params[1]}")
    print(f"Best Precision: {vae_best_precision_with_params[1]:.4f} \t| Hyperparameters: {vae_best_precision_with_params[0], vae_best_accuracy_with_params[1]}")
    print(f"Best Recall: {vae_best_recall_with_params[1]:.4f} \t| Hyperparameters: {vae_best_recall_with_params[0], vae_best_accuracy_with_params[1]}")
    print(f"Best F1 Score: {vae_best_f1_with_params[1]:.4f} \t| Hyperparameters: {vae_best_f1_with_params[0], vae_best_accuracy_with_params[1]}")

    ### test
    # best hyperparameters
    anomaly_threshold = 0.09545454545454546
    tumor_threshold = 0.46818181818181814

    title = '-----CVAE trained on synthetic data (VAE generated)-----\n-----Validated on real data-----\n-----Tested on real data-----' 
    pred, gt, recs, errors = test_brain_cvae(vae_synth_brain_cvae, test_loader, anomaly_threshold, tumor_threshold, alpha, title)

    # 2.2 - clustering variant -- train is the same
    ## val
    checkpoint_name = 'vae_synth_brain_cvae'
    vae_synth_brain_cvae = BrainCVAE(latent_dim)
    vae_synth_brain_cvae.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))
    vae_synth_brain_cvae.to(device)

    reconstructions, ground_truth = utils.get_reconstruction_error(vae_synth_brain_cvae, val_loader)
    best_params_acc, best_params_prec, best_params_recall, best_params_f1 = reconstruction_kmeans(reconstructions.cpu().numpy(), ground_truth.cpu().numpy())
    
    print("------CVAE (clustering) trained on synthetic data and validated on real data------\nBest Metrics and Hyperparameters on the validation set (alpha, anomaly_threshold):")
    print("----------------------------------------------------------------------")
    print(f"Best Accuracy: {best_params_acc[1]:.4f} \t| Hyperparameters: {best_params_acc[0][0]}, {best_params_acc[0][1]}")
    print(f"Best Precision: {best_params_prec[1]:.4f} \t| Hyperparameters: {best_params_prec[0][0]}, {best_params_acc[0][1]}")
    print(f"Best Recall: {best_params_recall[1]:.4f} \t| Hyperparameters: {best_params_recall[0][0]}, {best_params_acc[0][1]}")
    print(f"Best F1 Score: {best_params_f1[1]:.4f} \t| Hyperparameters: {best_params_f1[0][0]}, {best_params_acc[0][1]}")

    ### test
    # best hyperparameters
    alpha = 0.888888888888889
    anomaly_threshold = 0.14090909090909093
    reconstructions, ground_truth = utils.get_reconstruction_error(vae_synth_brain_cvae, test_loader)
    accuracy_weighted, precision_weighted, recall_weighted, f1_weighted= test_brain_cvae_clustering(reconstructions.cpu().numpy(), ground_truth.cpu().numpy(), anomaly_threshold, alpha, title='Synthetic train, Real Validation, Real test')

    # 3.1 - Mixed train, validation and test on real data
    ## train
    merged_train_loader, merged_val_loader, merged_test_loader = utils.get_merged_vae_dataset_loaders('../dataset', '../vae_dataset')

    merged_brain_cvae = BrainCVAE(latent_dim)
    merged_brain_cvae.to(device)
    checkpoint_name = 'merged_vae_synth_brain_cvae'
    
    train_vae(merged_brain_cvae, merged_train_loader, num_epochs, criterion, optimizer, scheduler, checkpoint_name)
    
    ### val
    merged_brain_cvae = BrainCVAE(latent_dim)
    merged_brain_cvae.to(device)
    checkpoint_name = 'merged_vae_synth_brain_cvae'

    alpha = 0.5 # alpha defines the weight of the metrics computed wrt unhealthy subjects and healthy subjects ( (alpha * metric_wrt_unhealthy) + ((1-alpha) * metric_wrt_healthy) )
    merged_vae_best_accuracy_with_params, \
    merged_vae_best_precision_with_params, \
    merged_vae_best_recall_with_params, \
    merged_vae_best_f1_with_params = \
        validate_brain_cvae(merged_brain_cvae, val_loader, alpha=alpha)

    ### VAE trained on mixed data, validated with real data
    print("------CVAE trained on merged data and validated on real data------\nBest Metrics and Hyperparameters on the validation set (anomaly_threshold):")
    print("----------------------------------------------------------------------")
    print(f"Best Accuracy: {merged_vae_best_accuracy_with_params[1]:.4f} \t| Hyperparameters: {merged_vae_best_accuracy_with_params[0]}")
    print(f"Best Precision: {merged_vae_best_precision_with_params[1]:.4f} \t| Hyperparameters: {merged_vae_best_precision_with_params[0]}")
    print(f"Best Recall: {merged_vae_best_recall_with_params[1]:.4f} \t| Hyperparameters: {merged_vae_best_recall_with_params[0]}")
    print(f"Best F1 Score: {merged_vae_best_f1_with_params[1]:.4f} \t| Hyperparameters: {merged_vae_best_f1_with_params[0]}")

    ### test
    alpha = 0.5
    title = 'CVAE trained on mixed data, validated on real data. Tested on unseen real data'

    merged_brain_cvae_test = BrainCVAE(latent_dim)
    merged_brain_cvae_test.to(device)
    merged_brain_cvae_test.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))
    # get best hyperparameters
    anomaly_threshold = 0.09545454545454546
    tumor_threshold = 0.44090909090909086
    
    pred, gt, recs, errors = test_brain_cvae(merged_brain_cvae_test, test_loader, anomaly_threshold, tumor_threshold, alpha, title)

    # 3.2 clustering variant -- train is the same
    checkpoint_name = 'merged_vae_synth_brain_cvae'
    merged_brain_cvae = BrainCVAE(latent_dim)
    merged_brain_cvae.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))
    merged_brain_cvae.to(device)

    reconstructions, ground_truth = utils.get_reconstruction_error(merged_brain_cvae, val_loader)
    best_params_acc, best_params_prec, best_params_recall, best_params_f1 = reconstruction_kmeans(reconstructions.cpu().numpy(), ground_truth.cpu().numpy())

    ### synthetic vae generated dataset latent_dim = 64
    print("------CVAE (clustering) trained on mixed data and validated on real data------\nBest Metrics and Hyperparameters on the validation set (alpha, anomaly_threshold) (clustering variant):")
    print("----------------------------------------------------------------------")
    print(f"Best Accuracy: {best_params_acc[1]:.4f} \t| Hyperparameters: {best_params_acc[0][0]}, {best_params_acc[0][1]}")
    print(f"Best Precision: {best_params_prec[1]:.4f} \t| Hyperparameters: {best_params_prec[0][0]}, {best_params_acc[0][1]}")
    print(f"Best Recall: {best_params_recall[1]:.4f} \t| Hyperparameters: {best_params_recall[0][0]}, {best_params_acc[0][1]}")
    print(f"Best F1 Score: {best_params_f1[1]:.4f} \t| Hyperparameters: {best_params_f1[0][0]}, {best_params_acc[0][1]}")

    # test
    # get best hyperparameters
    alpha = 0.8989898989898991
    anomaly_threshold = 0.6772727272727272
    reconstructions, ground_truth = utils.get_reconstruction_error(merged_brain_cvae, test_loader)
    accuracy_weighted, precision_weighted, recall_weighted, f1_weighted= test_brain_cvae_clustering(reconstructions.cpu().numpy(), ground_truth.cpu().numpy(), anomaly_threshold, alpha, title='Mixed train, Real Validation, Real test')