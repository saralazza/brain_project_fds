import os
os.environ["OMP_NUM_THREADS"] = '3' # due to kmeans RAM issues

import torch
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import utils

def reconstruction_kmeans(reconstruction_error, ground_truths, anomaly_threshold_range=(0.05, 0.95)):
    #### HYPER PARAMETERS ####
    # alpha remodels the clustering in function to the distances from the centroids. 
    # when alpha=1 we have the default clustering
    alphas = np.linspace(0, 1, num=100) 
    start_anomaly_threshold, end_anomaly_threshold = anomaly_threshold_range
    anomaly_thresholds = np.linspace(start_anomaly_threshold, end_anomaly_threshold, num=100)

    #### store the best metrics and the corresponding parameters ####
    best_params_acc = ()
    best_params_prec = ()
    best_params_recall = ()
    best_params_f1 = ()

    for anomaly_threshold in anomaly_thresholds:
        anomaly_reconstruction_error = reconstruction_error >= anomaly_threshold

        kmeans = KMeans(n_clusters=2, random_state=0) # 2 clusters: tumorous and non-tumorous
        kmeans.fit_predict(anomaly_reconstruction_error)

        # Improved cluster-to-label mapping using distances to centroids
        centroids = kmeans.cluster_centers_
        # given the centroids, calculate the distance for each point (reconstruction error image)
        distances_to_c0 = np.linalg.norm(reconstruction_error - centroids[0], axis=1)
        distances_to_c1 = np.linalg.norm(reconstruction_error - centroids[1], axis=1)

        for alpha_candidate in alphas: 
            # apply the current alpha to the distance thresholding
            dist_diff = np.abs(distances_to_c0 - distances_to_c1)
            threshold = alpha_candidate * np.max(dist_diff)

            # compute the new labels in function to the new distance
            new_labels = (dist_diff < threshold).astype(int)

            # metrics
            index_healthy = (ground_truths == 0)
            index_unhealthy = (ground_truths == 1)

            tp = ((new_labels == 1) & index_unhealthy).sum()
            fp = ((new_labels == 1) & index_healthy).sum()
            tn = ((new_labels == 0) & index_healthy).sum()
            fn = ((new_labels == 0) & index_unhealthy).sum()

            accuracy_weighted, precision_weighted, recall_weighted, f1_weighted = utils.get_metrics(tp, tn, fp, fn)

            current_params = (alpha_candidate, anomaly_threshold)

            best_params_acc = utils.get_best_metric(best_params_acc, current_params, accuracy_weighted, 'Accuracy')
            best_params_prec = utils.get_best_metric(best_params_prec, current_params, precision_weighted, 'Precision')
            best_params_recall = utils.get_best_metric(best_params_recall, current_params, recall_weighted, 'Recall')
            best_params_f1  = utils.get_best_metric(best_params_f1, current_params, f1_weighted, 'F1-score')

    return best_params_acc, best_params_prec, best_params_recall, best_params_f1

def reconstruction_kmeans(reconstruction_error, ground_truths, tumor_thresholds=(0.01, 0.99), original_dim=128):
    #### HYPER PARAMETERS ####
    tumor_thresholds = np.linspace(*tumor_thresholds, num=50) 

    #### store the best metrics and the corresponding parameters ####
    best_params_acc = ()
    best_params_prec = ()
    best_params_recall = ()
    best_params_f1 = ()

    reconstruction_encoded_position = utils.encode_positions_and_normalize(reconstruction_error)

    for tumor_threshold in tumor_thresholds: 
        predictions = np.asarray([])
        all_clusters = []
        for rec_error in reconstruction_encoded_position:
            rec_error = rec_error.T # n_samples x features (value, x_coord, y_coord). in this context, n_samples is the number of pixels
            kmeans = KMeans(n_clusters=2) # 2 clusters: tumorous (1) and non-tumorous (0)
            kmeans.fit_predict(rec_error)

            clusters = kmeans.labels_
            tumor_index = ((clusters==1).sum())/len(clusters) # check how big is the anomaly
            prediction = (tumor_index >= tumor_threshold).astype(int)
            predictions = np.append(predictions, prediction)

            clusters = clusters.reshape(original_dim, original_dim)
            all_clusters.append(clusters)

        # metrics for a given tumor_threshold
        index_healthy = (ground_truths == 0)
        index_unhealthy = (ground_truths == 1)

        tp = ((predictions == 1) & index_unhealthy).sum()
        fp = ((predictions == 1) & index_healthy).sum()
        tn = ((predictions == 0) & index_healthy).sum()
        fn = ((predictions == 0) & index_unhealthy).sum()

        accuracy_weighted, precision_weighted, recall_weighted, f1_weighted = utils.get_weighted_metrics(tp, tn, fp, fn)

        current_params = (tumor_threshold, clusters, all_clusters)

        best_params_acc = utils.get_best_metric(best_params_acc, current_params, accuracy_weighted, 'Accuracy')
        best_params_prec = utils.get_best_metric(best_params_prec, current_params, precision_weighted, 'Precision')
        best_params_recall = utils.get_best_metric(best_params_recall, current_params, recall_weighted, 'Recall')
        best_params_f1  = utils.get_best_metric(best_params_f1, current_params, f1_weighted, 'F1-score')

    return best_params_acc, best_params_prec, best_params_recall, best_params_f1

def reconstruction_dbscan(reconstruction_error, ground_truths, tumor_thresholds=(0.005, 0.2), epsilon_range=(0.001, 0.01), min_samples_range=(100, 200), original_dim=128):
    #### HYPER PARAMETERS ####
    tumor_thresholds = [0.1] #np.linspace(*tumor_thresholds, num=10) 
    epsilons = [0.005] #np.linspace(*epsilon_range, num=5)
    min_samples_space = [120] #np.linspace(*min_samples_range, num=10)

    #### store the best metrics and the corresponding parameters ####
    best_params_acc = ()
    best_params_prec = ()
    best_params_recall = ()
    best_params_f1 = ()

    tp = fp = tn = fn = 0

    reconstruction_encoded_position = utils.encode_positions_and_normalize(reconstruction_error)

    for min_samples in min_samples_space:
        for epsilon in epsilons:
            predictions = np.asarray([])
            prediction_scores = np.array([])
            clustered_tumors = np.array([])

            for rec_error in reconstruction_encoded_position:
                rec_error = rec_error.T # n_samples x features (value, x_coord, y_coord). in this context, n_samples is the number of pixels
                dbscan = DBSCAN(eps=epsilon, min_samples=int(min_samples)) # 2 clusters: tumorous (1) and non-tumorous (0)
                dbscan.fit_predict(rec_error)

                clusters = dbscan.labels_
                unique_clusters, counts = np.unique(clusters, return_counts=True)
                cluster_counts = dict(zip(unique_clusters, counts))
                cluster_counts.pop(-1, None) # remove 'noise' label
                cluster_prediction = 0 # by default, if in the reconstruction there's no cluster, then we say there's no tumor

                # Initialize brain_cluster with zeros matching the shape of the original image
                cluster_tumor = np.zeros((original_dim * original_dim))

                # Proceed only if cluster_counts is not empty
                if cluster_counts:
                    biggest_cluster_label = max(cluster_counts, key=cluster_counts.get)
                    cluster_count = cluster_counts[biggest_cluster_label]
                    cluster_prediction = (cluster_count / (original_dim * original_dim)).item()
                    cluster_tumor[clusters == biggest_cluster_label] = 1
                    cluster_tumor = cluster_tumor.reshape(original_dim, original_dim)

                prediction_scores = np.append(prediction_scores, cluster_prediction)
                clustered_tumors = np.append(clustered_tumors, cluster_tumor)    

            for tumor_threshold in tumor_thresholds:
                # after iterating on each brain, we have every score prediction
                # the brain is not healthy if we have a number of anomalies above tumor_threshold
                predictions = torch.tensor((prediction_scores >= tumor_threshold).astype(int))

                index_healthy = (ground_truths == 0)
                index_unhealthy = (ground_truths == 1)

                predictions_healthy = predictions == 0
                predictions_unhealthy = predictions == 1

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
                TP_healthy = tn
                FP_healthy = fn
                TN_healthy = tp
                FN_healthy = fp

                accuracy, precision, recall, f1 = utils.get_metrics(tp, tn, fp, fn)
                accuracy_healthy, precision_healthy, recall_healthy, f1_healthy = utils.get_metrics(TP_healthy, FP_healthy, TN_healthy, FN_healthy)
                
                accuracy_weighted = utils.get_weighted_metric(accuracy, accuracy_healthy)
                precision_weighted = utils.get_weighted_metric(precision, precision_healthy)
                recall_weighted = utils.get_weighted_metric(recall, recall_healthy)
                f1_weighted = utils.get_weighted_metric(f1, f1_healthy)

                
                print(f"\ttumor_threshold: {tumor_threshold}, min_samples: {min_samples}, f1: {f1_weighted}")

                current_params = (epsilon, tumor_threshold, clusters, clustered_tumors)

                best_params_acc = utils.get_best_metric(best_params_acc, current_params, accuracy_weighted, 'Accuracy')
                best_params_prec = utils.get_best_metric(best_params_prec, current_params, precision_weighted, 'Precision')
                best_params_recall = utils.get_best_metric(best_params_recall, current_params, recall_weighted, 'Recall')
                best_params_f1  = utils.get_best_metric(best_params_f1, current_params, f1_weighted, 'F1-score')

    return best_params_acc, best_params_prec, best_params_recall, best_params_f1
