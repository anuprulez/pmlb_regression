import sys
import time
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
import os
import numpy as np
import warnings
import seaborn as sb
import operator

import plotly
import plotly.graph_objs as go
from plotly import tools
import plotly.io as pio
import matplotlib.gridspec as gridspec
import seaborn as sns





warnings.filterwarnings('ignore')


########################### 3 panel figure

rcParams['font.family'] = "serif"

size_title = 11
size_xlabel = 10
size_ylabel = 10
size_xticks = 8
size_yticks = 8
size_colorbar = 8

folder_name = "tuning/"
clf_names_dir = os.listdir(folder_name)

excluded_clf = []
excluded_datasets = ['215_2dplanes.tsv.tabular', '344_mv.tsv.tabular', '1201_BNG_breastTumor.tsv.tabular']

clf_names_dir = [clf for clf in clf_names_dir if clf not in excluded_clf]

clf_names = list()
r2_dict = dict()
clf_sum_r2 = dict()
fit_time_dict = dict()


for clf in clf_names_dir:
    data_path = folder_name + clf
    files = [f for f in os.listdir(data_path)]
    r2_scores = list()
    r2_dict[clf] = list()
    fit_time = list()
    for f_name in files:
        if f_name not in excluded_datasets:
            f_path = data_path + "/" + f_name
            df = pd.read_csv(f_path, sep="\t")
            rank_test_score = df[df["rank_test_score"] == 1]
            mean_test_score = rank_test_score["mean_test_score"]
            fit_time_sec = rank_test_score["mean_fit_time"].iloc[0] + rank_test_score["mean_score_time"].iloc[0]
            fit_time.append(fit_time_sec)
            r2_scores.append(mean_test_score.iloc[0])
    r2_dict[clf] = r2_scores
    clf_sum_r2[clf] = np.sum(r2_scores)
    clf_names.append(clf)
    mean_fit_time = np.mean(fit_time)
    fit_time_dict[clf] = mean_fit_time
n_clf = len(clf_names)

sorted_clf = sorted(clf_sum_r2.items(), key=lambda kv: kv[1], reverse=True)
clf_sorted = [a[0] for a in sorted_clf]

model_nice_dict = {
    'XGBoost': 'XGB',
    'ExtraTrees': 'ETs',
    'GradientBoosting': 'GB',
    'RandomForest': 'RF',
    'Bagging': 'BA',
    'KNNeighbours': 'KNN',
    'AdaBoost': 'AB',
    'ExtraTree': 'ET',
    'DecisionTree': 'DT',
    'LinearSVR': 'LSVR',
    'BayesianRidge': 'BR',
    'LinearRegression': 'LR',
    'ElasticNet': 'ENet',
    'Huber': 'HB'
}

model_nice_dict_y = {
    'XGBoost': 'XGBoost (XGB)',
    'ExtraTrees': 'ExtraTrees (ETs)',
    'GradientBoosting': 'GradientBoosting (GB)',
    'RandomForest': 'RandomForest (RF)',
    'Bagging': 'Bagging (BAG)',
    'KNNeighbours': 'KNNeighbours (KNN)',
    'AdaBoost': 'AdaBoost (AB)',
    'ExtraTree': 'ExtraTree (ET)',
    'DecisionTree': 'DecisionTree (DT)',
    'LinearSVR': 'Linear SVR (LSVR)',
    'BayesianRidge': 'BayesianRidge (BR)',
    'LinearRegression': 'LinearRegression (LR)',
    'ElasticNet': 'ElasticNet (ENet)',
    'Huber': 'Huber (HB)'
}

x_labels = list(model_nice_dict.values())
y_labels = list(model_nice_dict_y.values())

# ---------------
# Plot heatmap
# ---------------

performance_datasets = np.zeros(shape=(n_clf, n_clf), dtype=float)

for x, clf_x in enumerate(clf_sorted):
    for y, clf_y in enumerate(clf_sorted):
        x_perf = r2_dict[clf_x]
        y_perf = r2_dict[clf_y]
        n_datasets = len(r2_dict[clf_x])
        x_g_y = len([1 for (a,b) in zip(x_perf, y_perf) if a > b]) / float(n_datasets)
        y_g_x = len([1 for (a,b) in zip(x_perf, y_perf) if b >= a]) / float(n_datasets)
        performance_datasets[x][y] = x_g_y
        performance_datasets[y][x] = y_g_x

mask_matrix = []
for x in range(n_clf):
    for y in range(n_clf):
        mask_matrix.append(x == y)
mask_matrix = np.array(mask_matrix).reshape(n_clf, n_clf)

performance_datasets = np.round((performance_datasets), 2)
performance_datasets = 100 * performance_datasets

gs = gridspec.GridSpec(2,2)
ax = plt.subplot(gs[0,0])
sb.heatmap(performance_datasets,
           fmt='0.0%',
           mask=mask_matrix,
           cmap='Blues',
           square=True, annot=False, vmin=0.0, vmax=100.0,
           xticklabels=x_labels, yticklabels=y_labels, 
           cbar=True,
           cbar_kws={"shrink": 0.8}
          )

cbar = ax.collections[0].colorbar
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=size_colorbar)

plt.xticks(fontsize=size_xticks)
plt.yticks(fontsize=size_yticks)
plt.xlabel('Losses', fontsize=size_xlabel)
plt.ylabel('Wins', fontsize=size_ylabel)
plt.title('(A) % out of '+ str(n_datasets) +' datasets where regressor A outperformed regressor B', fontsize=size_title)

# --------------------
# Plot fit time vs r2 score for regressors
# ------------------

r2_clf_matrix = np.zeros(shape=(n_clf, n_datasets), dtype=float)

folder_name_tuning = "tuning/"
folder_name_no_tuning = "no_tuning/"

clf_names_dir_tuning = os.listdir(folder_name_tuning)
clf_names_dir_tuning = [clf for clf in clf_names_dir_tuning if clf not in excluded_clf]

clf_names_dir_no_tuning = os.listdir(folder_name_no_tuning)
clf_names_dir_no_tuning = [clf for clf in clf_names_dir_no_tuning if clf not in excluded_clf]

clf_names_tuning = list()
clf_mean_r2_tuning = list()
r2_dict_tuning = dict()
fit_time_dict_tuning = dict()

for clf in clf_names_dir_tuning:
    data_path = folder_name_tuning + clf
    files = [f for f in os.listdir(data_path)]
    r2_scores = list()
    opt_fnames = list()
    for f_name in files:
        if f_name not in excluded_datasets:
            opt_fnames.append(f_name)
            f_path = data_path + "/" + f_name
            df = pd.read_csv(f_path, sep="\t")
            rank_test_score = df[df["rank_test_score"] == 1]
            mean_test_score = rank_test_score["mean_test_score"]
        r2_scores.append(mean_test_score.iloc[0])
    clf_names_tuning.append(clf)
    r2_dict_tuning[clf] = r2_scores

clf_names_no_tuning = list()
clf_mean_r2_no_tuning = list()
r2_dict_no_tuning = dict()
fit_time_dict_no_tuning = dict()

for clf in clf_names_dir_no_tuning:
    data_path = folder_name_no_tuning + clf
    files = [f for f in os.listdir(data_path)]
    r2_scores = list()
    no_opt_fnames = list()
    for f_name in files:
        if f_name not in excluded_datasets:
            no_opt_fnames.append(f_name)
            f_path = data_path + "/" + f_name
            df = pd.read_csv(f_path, sep="\t")
            rank_test_score = df[df["rank_test_score"] == 1]
            mean_test_score = rank_test_score["mean_test_score"]
            r2_scores.append(mean_test_score.iloc[0])
    clf_names_no_tuning.append(clf)
    r2_dict_no_tuning[clf] = r2_scores

n_datasets = len(no_opt_fnames)
n_clf = len(clf_names_no_tuning)
clf_datasets_perf = np.zeros(shape=(n_clf, n_datasets), dtype=float)
diff_dict = dict()
negative_diff = dict()
for x, clf in enumerate(clf_names_no_tuning):
    tuning_perf = r2_dict_tuning[clf]
    no_tuning_perf = r2_dict_no_tuning[clf]
    diff = [(a-b) for a,b in zip(tuning_perf, no_tuning_perf)]
    diff_dict[clf] = diff
    clf_datasets_perf[x] = diff
    
clf_df = pd.DataFrame(clf_datasets_perf)
clf_df.to_csv("clf_datasets_perf.csv", sep=",")
        
mean_perf_datasets = np.mean(clf_datasets_perf, axis=0)

c_fname = [(a,b) for (a,b) in zip(opt_fnames, mean_perf_datasets) if b < 0.0]

NUM_COLORS = len(clf_names)
cm = plt.get_cmap('tab20')
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

names_clf = list()
fit_time = list()
r2_scores = list()
ctr = 0

ax = plt.subplot(gs[0, 1])

for item in r2_dict.items():
    ax.scatter(fit_time_dict[item[0]], np.mean(item[1]), c=colors[ctr], label=item[0])
    ctr += 1

plt.title("(B) Fit time vs R2 regression score", size=size_title)
plt.xlabel("Mean fit time (in seconds)", size=size_xlabel)
plt.ylabel("Mean R2 score", size=size_ylabel)
plt.xticks(size=size_xticks)
plt.yticks(size=size_yticks)
ax.legend(bbox_to_anchor=(1.0, 1.00), shadow=True, ncol=1, prop={'size': size_yticks})
plt.grid(True)

# -----------------
# R2 scores for datasets across regressors
# ---------------


regressors_list = {
    'XGBoost': 'XGBoost',
    'ExtraTrees': 'ExtraTrees',
    'GradientBoosting': 'GradientBoosting',
    'RandomForest': 'RandomForest',
    'Bagging': 'Bagging',
    'KNNeighbours': 'KNNeighbours',
    'AdaBoost': 'AdaBoost',
    'ExtraTree': 'ExtraTree',
    'DecisionTree': 'DecisionTree',
    'LinearSVR': 'Linear SVR',
    'BayesianRidge': 'BayesianRidge',
    'LinearRegression': 'LinearRegression',
    'ElasticNet': 'ElasticNet',
    'Huber': 'Huber'
}

ax = plt.subplot(gs[1, 0])
r2_clf_matrix = np.zeros(shape=(n_clf, n_datasets), dtype=float)

for x, clf in enumerate(clf_sorted):
    r2_clf_matrix[x] = r2_dict[clf]

plt.title('(C) R2 scores of datasets across regressors', size=size_title)
plt.xlabel('Number of datasets', size=size_xlabel)
plt.yticks(range(n_clf), list(regressors_list.values()), size=size_yticks)
plt.xticks(size=size_xticks)
plt.imshow(r2_clf_matrix, cmap='Blues', aspect=5.5)
cbar = plt.colorbar(shrink=0.8)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=size_yticks)

# ---------------
# Box-plots for hyperparameter tuning improvements
# ---------------

hyper_improvement_datasets = np.zeros([len(clf_names_no_tuning), n_datasets])
y_labels = list()
for clf_idx, clf in enumerate(clf_names_no_tuning):
    if clf not in [""]:
        diff = [(x - y) for x, y in zip(r2_dict_tuning[clf], r2_dict_no_tuning[clf])]
        hyper_improvement_datasets[clf_idx] = diff
        y_labels.append(clf)

ax = plt.subplot(gs[1, 1])
df = pd.DataFrame(data=hyper_improvement_datasets.T, columns=y_labels)
sns.boxplot(data=df, orient='h', notch=True, palette=[sb.color_palette('Blues', n_colors=2)[1]])
plt.xlabel('R2 score improvement', size=size_xlabel)
plt.title('(D) R2 score improvement due to hyperparameter optimisation', size=size_title)
plt.xticks(size=size_xticks)
plt.yticks(size=size_yticks)
plt.xlim(0., 0.5)
plt.grid(True)

plt.show()
