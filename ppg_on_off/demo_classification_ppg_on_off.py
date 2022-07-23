# Classification demo
# Arnulf Graf (2021)
from sklearn.tree import export_text
import random as python_random
import glob
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from ml_classification import clf_sampling, clf_preprocessing, clf_train, clf_test
import pandas as pd
from pathlib import Path
import seaborn as sns
sns.set_style("darkgrid")
# seed randomizers
np.random.seed(42)
python_random.seed(42)


def decision_tree_logic(acc_mean, acc_var):
    if(acc_var <= 0.00082):
        if(acc_mean <= 1.06662):
            if(acc_mean <= 0.93380):
                if(acc_var <= 0.00002):
                    return 0
                else:
                    return 1
            else:
                return 0
        else:
            if(acc_var <= 0.00007):
                return 0
            else:
                return 1
    else:
        return 1



def make_pred_lookback(X, y, clf, preprocessor, day, do_plot, sleep_threshold, awake_threshold):
    n = X.shape[0]
    y_pred = np.zeros(n)
    clf_pred_history = np.zeros(n)

    burnin = 20
    lookback = 20
    majority_pct_sleep = sleep_threshold
    majority_pct_awake = awake_threshold

    for i in range(n):
        # pred = clf_test(preprocessor, clf, X[i].reshape(1, -1))
        pred = [decision_tree_logic(X[i][0], X[i][1])]
        clf_pred_history[i] = pred[0]
        if(i >= burnin):
            last_20 = clf_pred_history[i-lookback:i]
            if(pred[0] == 0): # wants to put us in asleep
                if((lookback - np.count_nonzero(last_20)) / 20 >= majority_pct_sleep): # how badly
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1

            if(pred[0] == 1): #wants to put us in awake
                if(np.count_nonzero(last_20) / 20 >= majority_pct_awake): # how badly
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0

        else:
            y_pred[i] = pred[0]
    if(do_plot):
        fig, axs = plt.subplots(4, 1, figsize = (15, 9), sharex = True)
        axs[0].plot(clf_pred_history, label = "CLF default", drawstyle = "steps-post")
        axs[0].plot(y, label = "True", color = "green", drawstyle = "steps-post")
        axs[0].set_ylabel("isAwake")
        axs[1].plot(y_pred, label = "Lookback", drawstyle = "steps-post")
        axs[1].plot(y, label = "True", color = "green", drawstyle = "steps-post")
        axs[1].set_ylabel("isAwake")
        axs[2].plot(X[:, 0], label = "mean acc")
        axs[2].set_ylabel("mean_acc")
        axs[3].plot(X[:, 1], label = "var acc")
        axs[3].set_ylabel("var_acc")
        axs[0].legend()
        axs[1].legend()
        plt.show()
        # plt.savefig(Path.cwd() / "indv_views" / f"{day.split('/')[-1].split('.parquet')[0]}.png", dpi=200)
        plt.close()
    class_report = classification_report(y,
                                         y_pred,
                                         target_names=["Asleep", "Awake"],
                                         output_dict=True,
                                         digits=4)
    # metric = class_report["macro avg"]["recall"]
    asleep_importance = 10
    metric = (class_report["Asleep"]["recall"] * asleep_importance + class_report["Awake"]["recall"] * 1) / (asleep_importance + 1)
    return y_pred, metric

def train_and_evaluate(scaling_name, model_name, use_lookback_test):
    days = glob.glob("/Users/lselig/Desktop/labeled_sleep_3numtaps/*.parquet")
    python_random.shuffle(days)
    train_test_days = days
    holdout_days = days[-40:]

    data = pd.concat([pd.read_parquet(x) for x in train_test_days])
    data.sort_values(by=["ts"])
    data = data.dropna()

    holdout_data = pd.concat([pd.read_parquet(x) for x in holdout_days])
    holdout_data.sort_values(by=["ts"])
    holdout_data = holdout_data.dropna()

    X = np.array(data[["acc_mean", "acc_var"]])
    y = np.array(data[["label"]])

    assert (X.shape[0] == y.shape[0]), 'Number of samples mismatch'
    n_sample, n_feature = X.data.shape
    print('Number of samples/features: ', n_sample, '/', n_feature)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    assert (X_train.shape[0] == y_train.shape[0]), 'Number of training samples mismatch'
    assert (X_test.shape[0] == y_test.shape[0]), 'Number of testing samples mismatch'
    print('Number of training/testing samples:', X_train.shape[0], '/', X_test.shape[0])

    # over- or undersampling training set
    X_train_sampled, y_train_sampled = clf_sampling('over_SMOTE', X_train, y_train)
    assert (X_train_sampled.shape[0] == y_train_sampled.shape[0]), 'Number of training samples mismatch'
    print('Number of training samples after sampling:', X_train_sampled.shape[0])
    for i in range(len(np.unique(y))):
        print('Train class ', i, 'size: ', len(np.nonzero(y_train == i)[0]), ' / ',
              len(np.nonzero(y_train_sampled == i)[0]))

    # preprocessing and training (learning)
    preprocessor, X_train_preprocessed = clf_preprocessing(scaling_name, X_train_sampled)
    clf = clf_train(model_name, X_train_preprocessed, y_train_sampled) #svm
    r = export_text(clf.best_estimator_, feature_names=['acc_mean', 'acc_var'], decimals = 5)
    print(r)
    # testing (inference)
    if(use_lookback_test):
        y_pred, macro_avg_recall = make_pred_lookback(X_test, y_test, clf, preprocessor, None, do_plot=True,
                                                      sleep_threshold=0.55555, awake_threshold=0.33333)
    else:
        y_pred = clf_test(preprocessor, clf, X_test)

    y_test = y_test.flatten()
    print('Testing accuracy: ' + '%3.2f' % (100 * accuracy_score(y_test, y_pred)) + '%')
    accuracy_mean = 100 * np.mean(1.0 - np.abs(np.sign(y_pred - y_test)))
    accuracy_sem = 100 * stats.sem(1.0 - np.abs(np.sign(y_pred - y_test)))
    conf_mat = confusion_matrix(y_test, y_pred, normalize=None)
    conf_mat_norm = confusion_matrix(y_test, y_pred, normalize='true')

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(9, 9))
    cax = plt.imshow(conf_mat_norm.T, aspect='auto', interpolation='none', origin='lower', cmap=cm.jet)
    cbar = fig.colorbar(cax)
    n_class = conf_mat.shape[0]
    for i in range(n_class):
        for j in range(n_class):
            ax.text(i, j, conf_mat[i, j], va='center', ha='center', color='w')
            ax.grid(False)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(
        'Classification accuracy across classes: ' + '%3.2f' % accuracy_mean + ' +- ' + '%3.2f' % accuracy_sem + '%')
    ax.axis('square')
    plt.show()
    fig.savefig(Path.cwd() / f'{scaling_name}_{model_name}_{"lookback" if use_lookback_test else "default"}.png', dpi = 200)

    for day in holdout_days:
        print(day)
        holdout_day = pd.read_parquet(day)
        holdout_day.sort_values(by=["ts"])
        holdout_day = holdout_day.dropna()

        X = np.array(holdout_day[["acc_mean", "acc_var"]])
        y = np.array(holdout_day[["label"]])
        y_pred = clf_test(preprocessor, clf, X)
        # y_pred, macro_avg_recall = make_pred_lookback(X, y, clf, preprocessor, day, do_plot = True, sleep_threshold=0.55555, awake_threshold=0.33333)

        # sts = np.linspace(0, 1, 10)
        # ats = np.linspace(0, 1, 10)
        # # metrics = []
        # runs = {"ats": [],
        #         "sts": [],
        #         "metrics": []}
        #
        # for s in sts:
        #     for a in ats:
        #         # print(s, a)
        #         y_pred_lb, macro_avg_recall = make_pred_lookback(X, y, clf, preprocessor, day, do_plot = False, sleep_threshold=s, awake_threshold=a)
        #         runs["metrics"].append(macro_avg_recall)
        #         runs["sts"].append(s)
        #         runs["ats"].append(a)
        #
        # fig, axs = plt.subplots(3, 1, figsize = (15, 9))
        # axs[0].plot(runs["metrics"])
        # axs[0].set_ylabel("macro avg recall")
        # axs[1].plot(runs["sts"])
        # axs[1].set_ylabel("Friction awake -> asleep")
        # axs[2].plot(runs["ats"])
        # axs[2].set_ylabel("Friction asleep -> awake")
        # axs[2].set_xlabel("Iteration")
        # opt_metric_idx = np.argmax(np.array(runs["metrics"]))
        # for k in range(3):
        #     axs[k].axvline(opt_metric_idx, ls = "--", color = "red")
        #
        # opt_s = runs["sts"][opt_metric_idx]
        # opt_a = runs["ats"][opt_metric_idx]
        # fig.suptitle(f"Opt awake -> asleep friction: {opt_s}\nOpt asleep -> awake friction: {opt_a}")
        # plt.tight_layout()
        # plt.savefig(Path.cwd() / "opt_thresholds" / f"{day.split('/')[-1].split('.parquet')[0]}.png", dpi = 200)
        # plt.close()

        # y_pred_lb, macro_avg_recall = make_pred_lookback(X, y, clf, preprocessor, day, do_plot = True, sleep_threshold=opt_s, awake_threshold=opt_a)
        y_pred_lb, macro_avg_recall = make_pred_lookback(X, y, clf, preprocessor, day, do_plot = True, sleep_threshold=0.55555, awake_threshold=0.33333)
        preds = [y_pred, y_pred_lb]

        fig, axs = plt.subplots(1, 2, figsize=(9, 9))
        for k, pred in enumerate(preds):
            print('Testing accuracy: ' + '%3.2f' % (100*accuracy_score(y, pred)) + '%')
            y = y.flatten()
            accuracy_mean = np.round(100 * np.mean(1.0 - np.abs(np.sign(pred - y))), 2)
            accuracy_sem = np.round(100 * stats.sem(1.0 - np.abs(np.sign(pred - y))), 2)
            conf_mat = confusion_matrix(y, pred, normalize=None)
            conf_mat_norm = confusion_matrix(y, pred, normalize='true')
            recall_asleep = np.round(conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]) * 100, 4)
            recall_awake = np.round(conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1]) * 100, 4)
            # plot confusion matrix
            cax = axs[k].imshow(conf_mat_norm.T, aspect='auto', interpolation='none', origin='lower', cmap=cm.jet)
            # cbar = plt.colorbar(cax, ax = axs[k])
            n_class = conf_mat.shape[0]
            for i in range(n_class):
                for j in range(n_class):
                    axs[k].text(i, j, conf_mat[i, j], va='center', ha='center', color='w')
            axs[k].set_xlabel('True')
            axs[k].set_ylabel('Predicted')
            # if(k == 1):
            axs[k].set_title(f'{"lookback" if k == 1 else "base"} accuracy: {accuracy_mean} +- {accuracy_sem}\n% correct asleep: {recall_asleep}\n% correct awake: {recall_awake}')
            axs[k].grid(False)
            # else:
            #     axs[k].set_title('Base accuracy: ' +  '%3.2f' % accuracy_mean + ' +- ' + '%3.2f' % accuracy_sem + '%')
            axs[k].axis('square')
        fig.suptitle(f"Model: {m} \nScaling: {s}\nDay: {day.split('/')[-1]}")
        plt.tight_layout()
        # plt.show()
        plt.savefig(Path.cwd() / "confusion_matrices" / f"{day.split('/')[-1].split('.parquet')[0]}.png", dpi = 200)
        plt.close()
        # return

# scalings = ["identity"]
# models = ["decision_tree"]
scalings = ["identity"]
models = ["decision_tree"]



for s in scalings:
    for m in models:
        print("Working on", s, m)
        train_and_evaluate(s, m, use_lookback_test = True)
