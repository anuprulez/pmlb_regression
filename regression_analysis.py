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

warnings.filterwarnings('ignore')

start_time = time.time()
folder_name = "european_days/"
clf_names_dir = os.listdir(folder_name)


def check_file(f_name):
    for clf in clf_names_dir:
        data_path = folder_name + clf
        files = [f for f in os.listdir(data_path)]
        if f_name not in files:
            return False
            break
    return True


clf_names = list()
clf_mean_r2 = list()
r2_dict = dict()
fit_time_dict = dict()
for clf in clf_names_dir:
    data_path = folder_name + clf
    files = [f for f in os.listdir(data_path)]
    r2_scores = list()
    fit_time = list()
    common_fileset = list()
    for f_name in files:
        if_file_present = check_file(f_name)
        if if_file_present is True:
            common_fileset.append(f_name)
            f_path = data_path + "/" + f_name
            df = pd.read_csv(f_path, sep="\t")
            rank_test_score = df[df["rank_test_score"] == 1]
            mean_test_score = rank_test_score["mean_test_score"]
            fit_time_sec = rank_test_score["mean_fit_time"].iloc[0] + rank_test_score["mean_score_time"].iloc[0]
            r2_scores.append(mean_test_score.iloc[0])
            fit_time.append(fit_time_sec)
    clf_names.append(clf)
    mean_r2 = np.mean(r2_scores)
    clf_mean_r2.append(mean_r2)
    r2_dict[clf] = mean_r2
    mean_fit_time = np.mean(fit_time)
    fit_time_dict[clf] = mean_fit_time
    print("Mean R2 score for %s regressor: %0.2f" % (clf, mean_r2))
    print("Total fit time for %s regressor: %0.4f seconds" % (clf, mean_fit_time))

    '''plt.figure()
    plt.plot(r2_scores, color='r')
    plt.grid(True)
    plt.xlabel("Number of datasets")
    plt.ylabel("R2 regression score")
    plt.title(('Best R2 scores for %d datasets (%s regression)' % (len(common_fileset), clf)))
    plt.show()'''
    print("-------------x------------------x---------------x---------------")
    print(" ")
    

fit_times_list = list()
r2_score_list = list()
names_clf = list()
fit_time = list()
trace_time_acc_list = list()
ctr = 0

NUM_COLORS = 16
cm = plt.get_cmap('tab20c')
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

rcParams["font.size"] = 24
rcParams["font.family"] = "Times New Roman"

for item in r2_dict.items():
    names_clf.append(item[0])
    fit_time.append(fit_time_dict[item[0]])
    r2_score_list.append(item[1])
    plt.scatter(fit_time_dict[item[0]], item[1], color=colors[ctr])
    trace_time_acc = go.Scatter(
        x = [fit_time_dict[item[0]]],
        y = [item[1]],
        name=item[0],
        mode = 'markers',
        marker = dict(
            size = 10,
            color = 'rgba' + str(colors[ctr]),
        )
    )
    trace_time_acc_list.append(trace_time_acc)
    ctr += 1

layout_time_acc = dict(
    title='Mean fit time vs R2 regression score',
    font=dict(family='Times new roman', size=24),
    yaxis=dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        title='Mean R2 regression score',
        titlefont=dict(
            family='Times new roman',
            size=24
        )
    ),
    xaxis=dict(
        zeroline=False,
        showline=True,
        showticklabels=True,
        showgrid=True,
        title='Mean fit time (in seconds)',
        #tickangle=-45,
        titlefont=dict(
            family='Times new roman',
            size=24
        )
    ),
    margin=dict(
        l=100,
        r=20,
        t=70,
        b=200,
    ),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig_tp = go.Figure(data=trace_time_acc_list, layout=layout_time_acc)
plotly.offline.plot(fig_tp, filename="fit_time_r2.png", auto_open=True)
#plotly.io.write_image(fig_tp, "plots/fit_time_r2.png")

print("-------------x------------------x---------------")


# plot bar chart for regressors
r2_dict = sorted(r2_dict.items(), key=lambda kv: kv[1])

cf = list()
r2 = list()

for item in r2_dict:
    cf.append(item[0])
    r2.append(item[1])

trace0 = go.Bar(
    x=r2,
    y=cf,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 0.6)',
            width=1),
    ),
    name='R2 regression scores vs regressors',
    orientation='h'
)


layout = dict(
    title='R2 regression scores vs regressors',
    font=dict(family='Times new roman', size=24),
    yaxis=dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        titlefont=dict(
            family='Times new roman',
            size=24
        )
    ),
    xaxis=dict(
        zeroline=False,
        showline=True,
        showticklabels=True,
        showgrid=True,
        title='R2 scores',
        titlefont=dict(
            family='Times new roman',
            size=24
        )
    ),
    margin=dict(
        l=200,
        r=20,
        t=70,
        b=70,
    ),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig_tp = go.Figure(data=[trace0], layout=layout)
plotly.offline.plot(fig_tp, filename="r2_scores.png", auto_open=True)
#plotly.io.write_image(fig_tp, "plots/r2_scores.png")
print("-------------x------------------x---------------")

# plot number of samples in each dataset


data_folder_name = "data_used/"
folder_dir = os.listdir(data_folder_name)
dataset_length = list()

for ds in folder_dir:
    if ds + ".tabular" in common_fileset:
        ds_path = data_folder_name + ds
        df = pd.read_csv(ds_path, sep="\t")
        dataset_length.append(len(df))

trace1 = go.Bar(
    x=np.arange(len(dataset_length)),
    y=sorted(dataset_length),
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 0.6)',
            width=1),
    ),
    name='Size of datasets',
)


layout1 = dict(
    title='Size of datasets',
    font=dict(family='Times new roman', size=24),
    yaxis=dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        title='Size of datasets',
        titlefont=dict(
            family='Times new roman',
            size=24
        )
    ),
    xaxis=dict(
        zeroline=False,
        showline=True,
        showticklabels=True,
        showgrid=True,
        title='Number of datasets',
        titlefont=dict(
            family='Times new roman',
            size=24
        )
    ),
    margin=dict(
        l=100,
        r=20,
        t=70,
        b=70,
    ),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig_tp = go.Figure(data=[trace1], layout=layout1)
plotly.offline.plot(fig_tp, filename="size_datasets.png", auto_open=True)
#plotly.io.write_image(fig_tp, "plots/size_datasets.png")
print("-------------x------------------x---------------")

end_time = time.time()
print('Total time taken: %d seconds' % int(end_time - start_time))
