from collections import defaultdict
import pandas as pd
import numpy as np


def get_mean_std(csv_lists,
                 arguments=("accuracy", "loss", "val_accuracy", "val_loss"),
                 operators=(max, min, max, min),
                 save_path=None,
                 logger=None,
                 ):
    metrics = defaultdict(list)
    print(f"[INFO] Extracting values from csv files...")
    for csv in csv_lists:
        print(f"[INFO] Getting the values of file {csv}")
        csv_file = pd.read_csv(csv)
        for metric, op in zip(arguments, operators):
            val = op(csv_file[metric])
            metrics[metric].append(val)

    metrics = {metric: {"std": round(np.std(val_list), 4), "mean": round(np.mean(val_list), 4)} for metric, val_list in
               metrics.items()}
    if save_path is not None:
        df = pd.DataFrame([[val['mean'], val['std']] for _, val in metrics.items()], columns=['mean', 'std'],
                          index=list(metrics.keys()))
        df.to_csv(save_path)
    return metrics


def get_conf_mean_std(conf_matrices):
    metrics = defaultdict(list)
    print(f"[INFO] Extracting values from confusion matrices")
    for conf in conf_matrices:
        df = pd.read_csv(conf, index_col=0).values
        print(f"[INFO] Getting the values of df: {conf}")
        tp = df[1, 1]
        tn = df[0, 0]
        fn = df[0, 1]
        fp = df[1, 0]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = (2 * tp) / (2 * tp + fp + fn)
        metrics["sensitivity"].append(sensitivity)
        metrics["specificity"].append(specificity)
        metrics["accuracy"].append(accuracy)
        metrics["f1_score"].append(f1_score)

    metrics = {metric: {"std": round(np.std(val_list), 4), "mean": round(np.mean(val_list), 4)} for metric, val_list in
               metrics.items()}
    print(f'[INFO] Successfully Extracted {metrics.keys()}')
    return metrics


if __name__ == '__main__':
    metrics = get_mean_std(
        csv_lists=["/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_37_26.379079/log.csv",
                   "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_09_40.034447/log.csv",
                   "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_07_52.880768/log.csv",
                   "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_06_43.842120/log.csv",
                   ])
    print(metrics)
