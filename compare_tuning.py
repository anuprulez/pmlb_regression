import sys
import time
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
import os
import numpy as np
import warnings

import plotly
import plotly.graph_objs as go
from plotly import tools
import plotly.io as pio


warnings.filterwarnings('ignore')

start_time = time.time()
folder_name_tuning = "tuning/"
folder_name_no_tuning = "no_tuning/"
clf_names_dir_tuning = os.listdir(folder_name_tuning)
clf_names_dir_no_tuning = os.listdir(folder_name_no_tuning)

file_obj = open("file_names.txt", "r")
dataset_names = file_obj.read()
dataset_names = dataset_names.split("\n")

clf_names_tuning = list()
clf_mean_r2_tuning = list()
r2_dict_tuning = dict()
fit_time_dict_tuning = dict()

print("Processing results for tuning ...")

for clf in clf_names_dir_tuning:
    data_path = folder_name_tuning + clf
    files = [f for f in os.listdir(data_path)]
    r2_scores = list()
    fit_time = list()
    common_fileset = list()
    for f_name in files:
        if f_name in dataset_names: 
            common_fileset.append(f_name)
            f_path = data_path + "/" + f_name
            df = pd.read_csv(f_path, sep="\t")
            rank_test_score = df[df["rank_test_score"] == 1]
            mean_test_score = rank_test_score["mean_test_score"]
            fit_time_sec = rank_test_score["mean_fit_time"].iloc[0] + rank_test_score["mean_score_time"].iloc[0]
            r2_scores.append(mean_test_score.iloc[0])
            fit_time.append(fit_time_sec)
    clf_names_tuning.append(clf)
    mean_r2 = np.mean(r2_scores)
    clf_mean_r2_tuning.append(mean_r2)
    r2_dict_tuning[clf] = mean_r2
    mean_fit_time = np.mean(fit_time)
    fit_time_dict_tuning[clf] = mean_fit_time
    print("Mean R2 score for %s regressor: %0.2f" % (clf, mean_r2))
    print("Total fit time for %s regressor: %0.4f seconds" % (clf, mean_fit_time))

    print("-------------x------------------x---------------x---------------")
    print(" ")


clf_names_no_tuning = list()
clf_mean_r2_no_tuning = list()
r2_dict_no_tuning = dict()
fit_time_dict_no_tuning = dict()

print("Processing results for no tuning ...")

for clf in clf_names_dir_no_tuning:
    data_path = folder_name_no_tuning + clf
    files = [f for f in os.listdir(data_path)]
    r2_scores = list()
    fit_time = list()
    for f_name in files:
        if f_name in dataset_names:
            f_path = data_path + "/" + f_name
            df = pd.read_csv(f_path, sep="\t")
            rank_test_score = df[df["rank_test_score"] == 1]
            mean_test_score = rank_test_score["mean_test_score"]
            fit_time_sec = rank_test_score["mean_fit_time"].iloc[0] + rank_test_score["mean_score_time"].iloc[0]
            r2_scores.append(mean_test_score.iloc[0])
            fit_time.append(fit_time_sec)
    clf_names_no_tuning.append(clf)
    mean_r2 = np.mean(r2_scores)
    clf_mean_r2_no_tuning.append(mean_r2)
    r2_dict_no_tuning[clf] = mean_r2
    mean_fit_time = np.mean(fit_time)
    fit_time_dict_no_tuning[clf] = mean_fit_time
    print("Mean R2 score for %s regressor: %0.2f" % (clf, mean_r2))
    print("Total fit time for %s regressor: %0.4f seconds" % (clf, mean_fit_time))

    print("-------------x------------------x---------------x---------------")
    print(" ")


NUM_COLORS = 16
cm = plt.get_cmap('tab20c')
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

hyp_tuning_res = list()
no_hyp_tuning_res = list()
ctr = 1
trace_tng_ntng_list = list()
clf_names = list()
for clf in r2_dict_no_tuning:
    hyp_tuning_res.append(r2_dict_tuning[clf])
    no_hyp_tuning_res.append(r2_dict_no_tuning[clf])
    clf_names.append(clf)
    trace_tng_ntng = go.Scatter(
        x = [r2_dict_no_tuning[clf]],
        y = [r2_dict_tuning[clf]],
        name=clf,
        mode = 'markers',
        marker = dict(
            size = 10,
            color = 'rgba' + str(colors[ctr]),
        )
    )
    trace_tng_ntng_list.append(trace_tng_ntng)
    ctr += 1

fontsz = 30
ao = True

# ----------------------------------

trace_tuning = go.Bar(
    x=hyp_tuning_res,
    y=clf_names_no_tuning,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            width=1),
        ),
    name='With optimisation',
    orientation='h'
)

trace_notuning = go.Bar(
    x=no_hyp_tuning_res,
    y=clf_names_no_tuning,
    marker=dict(
        color='rgba(50, 51, 96, 0.6)',
        line=dict(
            width=1),
        ),
    name='Without optimisation',
    orientation='h'
)

layout = dict(
    title='R2 scores of regressors with and without hyperparameter optimisation',
    font=dict(family='Times new roman', size=fontsz),
    yaxis=dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        titlefont=dict(
            family='Times new roman',
            size=fontsz
        )
    ),
    xaxis=dict(
        zeroline=False,
        showline=True,
        showticklabels=True,
        showgrid=True,
        title='Mean R2 scores',
        titlefont=dict(
            family='Times new roman',
            size=fontsz
        )
    ),
    margin=dict(
        l=220,
        r=20,
        t=70,
        b=70,
    ),
)

fig_tp = go.Figure(data=[trace_tuning, trace_notuning], layout=layout)
plotly.offline.plot(fig_tp, filename="r2_scores_bar", auto_open=ao)
pio.write_image(fig_tp, "plots/r2_scores_bar.png", width=1200, height=800) 

end_time = time.time()
print('Total time taken: %d seconds' % int(end_time - start_time))
