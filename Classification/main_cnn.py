import torch
from torch import nn

import utils
from CNN import BrainCNN, train_validation

if __name__ == '__main__':
    # load dataset
    dataset_path = '../dataset'
    train_loader, val_loader, test_loader = utils.get_cnn_dataset_loaders(dataset_path)

    p = 0.5 # dropout probability
    brain_cnn = BrainCNN(p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain_cnn.to(device)

    # 1 - train and validate CNN on real data
    num_epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(brain_cnn.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    checkpoint_name = 'brain_cnn'

    train_validation(brain_cnn, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, checkpoint_name)
   
    # get the best CNN
    p = 0
    brain_cnn = BrainCNN(p) # set dropout probability to zero at inference time
    brain_cnn.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))
    brain_cnn.to(device)

    TP_train, TN_train, FP_train, FN_train = utils.get_metrics(brain_cnn, train_loader)
    TP_valid, TN_valid, FP_valid, FN_valid = utils.get_metrics(brain_cnn, val_loader)
    TP_test, TN_test, FP_test, FN_test = utils.get_metrics(brain_cnn, test_loader)

    # Print all metrics on train, validation and test sets
    utils.print_metrics(TP_train, TN_train, FP_train, FN_train, TP_valid, TN_valid, FP_valid, FN_valid, TP_test, TN_test, FP_test, FN_test)

    # 2 - train CNN on synthetic data, validate on real data
    synth_dataset_path = '../vae_dataset'
    vae_synth_train_loader, vae_synth_val_loader, vae_synth_test_loader = utils.get_cnn_synthetic_vae_dataset_loaders(synth_dataset_path)

    synth_brain_cnn = BrainCNN(p) # p is the dropout probability (0.5)
    synth_brain_cnn.to(device)

    num_epochs = 20
    checkpoint_name = 'vae_synth_brain_cnn'
    train_validation(synth_brain_cnn, vae_synth_train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, checkpoint_name)

    ### Evaluation
    vae_synth_brain_cnn = BrainCNN(0)
    vae_synth_brain_cnn.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))
    vae_synth_brain_cnn.to(device)

    TP_train, TN_train, FP_train, FN_train = utils.get_metrics(vae_synth_brain_cnn, train_loader)
    TP_test, TN_test, FP_test, FN_test = utils.get_metrics(vae_synth_brain_cnn, test_loader)

    # test on all unseen data
    tp = TP_train + TP_test
    tn = TN_train + TN_test
    fp = FP_train + FP_test
    fn = FN_train + FN_test

    utils.print_metrics_simple(tp, tn, fp, fn, 'CNN trained on synthetic data------\n------Validated on real data------\n------Tested on unseen real data')

    # 3 - CNN train and validation on mixed data
    merged_train_loader, merged_valid_loader, merged_test_loader = utils.get_cnn_merged_vae_dataset_loaders(dataset_path, synth_dataset_path)

    merged_brain_cnn = BrainCNN(p) # p is the dropout probability
    merged_brain_cnn.to(device)
    checkpoint_name = 'merged_vae_synth_brain_cnn'
    train_validation(merged_brain_cnn, merged_train_loader, merged_valid_loader, num_epochs, criterion, optimizer, scheduler, checkpoint_name)

    merged_brain_cnn = BrainCNN(0)
    merged_brain_cnn.load_state_dict(torch.load(f"../checkpoint/{checkpoint_name}.pth"))
    merged_brain_cnn.to(device)

    TP_train, TN_train, FP_train, FN_train = utils.get_metrics(merged_brain_cnn, train_loader)
    TP_valid, TN_valid, FP_valid, FN_valid = utils.get_metrics(merged_brain_cnn, val_loader)
    TP_test, TN_test, FP_test, FN_test = utils.get_metrics(merged_brain_cnn, test_loader)

    tp = TP_train + TP_valid + TP_test
    tn = TN_train + TN_valid + TN_test
    fp = FP_train + FP_valid + FP_test
    fn = FN_train + FN_valid + FN_test

    utils.print_metrics_simple(tp, tn, fp, fn, 'CNN trained and validated on merged data (real and synthetic), tested on real data')