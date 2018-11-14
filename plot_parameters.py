import plotly
import plotly.graph_objs as go
from plotly import tools
import plotly.io as pio

import pandas as pd

# Get Data: this ex will only use part of it (i.e. rows 750-1500)
df = pd.read_csv('classification_report.csv')
ao = True
fontsz = 16
trace1 = go.Scatter3d(
    x=df['param_regression__loss'],
    y=df['param_regression__max_features'],
    z=df['mean_test_score'],
    mode='markers',
    marker=dict(
        color = df['mean_test_score'],
        colorscale = 'Rainbow',
        colorbar = dict(title = 'R2 score'),
        line=dict(color='rgb(0, 0, 0)')
    )
)

layout=go.Layout(
    width=800,
    height=600,
    title = 'Gradient Boosting Regression',
    font=dict(family='Times new roman', size=fontsz),
    scene = dict(
        xaxis=dict(title='Loss', titlefont=dict(color='green')),
        yaxis=dict(title='Max features', titlefont=dict(color='green')),
        zaxis=dict(title='R2 regression scores', titlefont=dict(color='blue'))
    )
)

fig_tp = go.Figure(data=[trace1], layout=layout)
plotly.offline.plot(fig_tp, filename="gb_parameter", auto_open=ao)
pio.write_image(fig_tp, "plots/gb_parameter.png", height=800, width=1200) 
