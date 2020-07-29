---
layout: post
title:      "COST-BENEFIT ANALYSIS"
date:       2020-07-29 00:29:52 +0000
permalink:  cost-benefit_analysis
---

<a id="index"></a>

Once a model is developed and ready for use in production there are important decisions to be made regarding exactly how the predictions should be used. Some of the main decisions center around the business value and risk of the model's performance. Broadly speaking, value lies in correct predictions and risk lies in incorrect predictions.

---------------------------------------------
**[1. Load Data & Libraries](#load_data)**
<br>
**[2. The Cost of a Prediction](#cost)**
<br> &nbsp;&nbsp;&nbsp; - [Cost-Profit Confusion Matrix](#cost_cm)
<br>
**[3. Confidence Thresholds and Costs](#thresh)**
<br> &nbsp;&nbsp;&nbsp; - [Cost-Profit Confusion Matrix @.8 Cutoff](#cm_cut)
<br> &nbsp;&nbsp;&nbsp; - [Confusion Matrix @.8 Cutoff](#cm)

--------------------------------------------------
# Load Libraries and Data


```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    


```python
# Load libraries and configure notebook settings
%matplotlib inline
import json
from requests.auth import HTTPBasicAuth

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from eval_functions import corr_matrix, get_datasets, multiclass_roc_curve, \
                           plot_confusion_matrix,get_ratios_multiclass, \
                           plot_cm_best_estimator, plot_ngrams, parse_results

pd.set_option('display.max_colwidth', -1)
np.random.seed(0)

with open('G:\\aws_processed\\info.json', 'r') as f:
    d = json.load(f)
    username = d['plotly_username']
    api_key = d['plotly_api_key']

auth = HTTPBasicAuth(username, api_key)
headers = {'Plotly-Client-Platform': 'python'}
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
# Probabilities from the uncorrected dataset were used
predictions = pd.read_json('test_probs_bert.json', orient='column')

predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>neg</th>
      <th>pos</th>
      <th>mixed</th>
      <th>preds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New menu is fabulous!! Quality of food is awesome. Service can be jacked up a bit. Our waitress was either high or naturally "slow".</td>
      <td>2</td>
      <td>0.008629</td>
      <td>0.013405</td>
      <td>0.978027</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lovely food, great atmosphere</td>
      <td>1</td>
      <td>0.000119</td>
      <td>0.999512</td>
      <td>0.000358</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Great service</td>
      <td>1</td>
      <td>0.000123</td>
      <td>0.999512</td>
      <td>0.000273</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Loved the burgers and crispy fries. Recommend the cheese and bacon fries</td>
      <td>1</td>
      <td>0.000180</td>
      <td>0.999023</td>
      <td>0.000847</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Great place for parents with toddlers. Great play area. Great staff. Screens to watch your kids while you eat. Awesome kids menu and great pizza. Kids Heaven</td>
      <td>1</td>
      <td>0.000133</td>
      <td>0.999512</td>
      <td>0.000597</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
cm = confusion_matrix(predictions.label, predictions.preds)
plot_confusion_matrix(cm, ['negative', 'positive', 'mixed'], normalize=True)
```

    Normalized confusion matrix
    


![png](cost_benefit_analysis_files/cost_benefit_analysis_4_1.png)


<a id="cost"></a>
# The Cost of a Prediction

[Return to Index](#index)

While the costs and benefits here are hypothetical in terms of exact monetary amount, the hit:miss ratio is realistic. Correctly classifying a review will earn \\$1 for the majority class and \\$2 for the minority classes.

Incorrectly classifying a review depends on the misclassification:
```
    - A negative review classified as positive will cost $40
    - A negative review classified as mixed will cost $10
    
    - A positive review classified as negative will cost $40
    - A positive review classifed as mixed will cost $10
    
    - A mixed review classified as negative will cost $6
    - A mixed review classified as positive will cost $10
```


```python
cost_matrix = np.array([[2, -40, -10], [-40, 1, -10], [-6, -10, 2]])
cost_matrix
```




    array([[  2, -40, -10],
           [-40,   1, -10],
           [ -6, -10,   2]])



<a id="cost_cm"></a>
## Cost-Profit Confusion Matrix

If we apply this to our confusion matrix, we get:


```python
cm = confusion_matrix(predictions.label, predictions.preds)
cm_cost = np.round(cm*cost_matrix, 2)
plot_confusion_matrix(cm_cost, ['neg', 'pos', 'mixed'], normalize=False)
print('Profit: {}'.format(cm_cost.sum()))
```

    Confusion matrix, without normalization
    Profit: 1565
    


![png](cost_benefit_analysis_files/cost_benefit_analysis_8_1.png)


<a id="thresh"></a>
# Confidence Thresholds and Costs

Using different confidence thresholds we can adjust the proportion of correct:incorrect predictions and see how they impact costs and profits.

[Return to Index](#index)


```python
def plot_cost_cm(df, cols, cost_matrix,
                 neg_cut=0.92, pos_cut=0.92, mix_cut=0.92,
                 return_plot=True):
    """
    Return cost-benefit confusion matrix.
    
    Add cost-matrix with hypothetical/ realistic costs and benefits
    for incorrect and correct predictions. The resulting confusion 
    matrix will multiply the cost-matrix by the confusion matrix,
    taking into consideration chosen probability cutoffs.
    
    The maximum probability or stanard 0.5 threshold is not 
    automatically applied.
    """
    
    neg_certain = list(df.index[(df[cols[0]] > neg_cut)])
    pos_certain = list(df.index[(df[cols[1]] > pos_cut)])
    mix_certain = list(df.index[df[cols[2]] > mix_cut])
    incl_total = list(set(neg_certain + pos_certain + mix_certain))
    
    retained = (len(incl_total)/len(df))*100
    
    uncertain = df[~df.index.isin(incl_total)]
    certain = df[df.index.isin(incl_total)]

    f1_certain = f1_score(certain.label, certain.preds, average='macro')
    
    cm = confusion_matrix(certain.label, certain.preds)
    cm_cost = np.round(cm*cost_matrix, 2)
    profit = cm_cost.sum()
    
    if return_plot:
        print('Retained Data: {:0.2f}%'.format(retained))
        print('Macro-F1 Score: {}'.format(f1_certain))
        print('Total Profits: {}'.format(profit))
        plot_confusion_matrix(cm_cost, ['neg', 'pos', 'mixed'], normalize=False)
    
    return profit, retained, f1_certain

def plot_cost_benefit_thresholds(cost_matrix, predictions):
    """
    Plot the range of probability cutoffs from 0.5 to 0.95
    for the: 
    1) total profit 
    2) f1-scores (recall-precision geometric mean)
    3) amount of data retained after excluding all cases with 
    probabilities below the chosen thresholds.
    """
    
    thresholds = np.arange(0.5, 1.0, 0.05)
    profits = []
    perc_retained = []
    f1_scores = []
    for threshold in thresholds:
        profit, retained, f1 = plot_cost_cm(predictions, 
                                            ['neg', 'pos', 'mixed'],
                                            cost_matrix, 
                                            neg_cut=threshold, 
                                            pos_cut=threshold, 
                                            mix_cut=threshold,
                                            return_plot=False)
        profits.append(profit)
        perc_retained.append(round(retained, 2))
        f1_scores.append(round(f1*100, 3))
        
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing = 0.1,
                        subplot_titles=("Profits by Cutoff", 
                                        "Model Performance vs. Data Retained"))

    fig.append_trace(go.Scatter(x=thresholds, y=profits, text='Profit at Cutoff',
        name=''), row=1, col=1)

    fig.append_trace(go.Scatter(x=thresholds, y=f1_scores, 
        text='F1 score at Cutoff', name=''), row=2, col=1)

    fig.append_trace(go.Scatter(x=thresholds,y=perc_retained,
        text='Data Retained', name=''), row=2, col=1)
    
    fig.update_xaxes(tickfont=dict(size=14),
                     title_text="Probability Cutoffs", row=2, col=1)
    
    fig.update_yaxes(title_text="Profit", tickfont=dict(size=12), row=1, col=1)
    fig.update_yaxes(title_text="Percentage", tickfont=dict(size=12), row=2, col=1)

    fig.update_layout(height=800, width=600, 
                      title_text="Cost-Benefit for Probability Cutoffs",
                      font=dict(family="Courier New, monospace",
                                size=18,
                                color="#7f7f7f"),
                     showlegend=False)
    fig.show()
    
plot_cost_benefit_thresholds(cost_matrix, predictions)
```


<div>


            <div id="088b46e5-4a45-4634-8933-395f256597b7" class="plotly-graph-div" style="height:800px; width:600px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("088b46e5-4a45-4634-8933-395f256597b7")) {
                    Plotly.newPlot(
                        '088b46e5-4a45-4634-8933-395f256597b7',
                        [{"name": "", "text": "Profit at Cutoff", "type": "scatter", "x": [0.5, 0.55, 0.6000000000000001, 0.6500000000000001, 0.7000000000000002, 0.7500000000000002, 0.8000000000000003, 0.8500000000000003, 0.9000000000000004, 0.9500000000000004], "xaxis": "x", "y": [1719, 1927, 2149, 2245, 2396, 2529, 2730, 2831, 3101, 3400], "yaxis": "y"}, {"name": "", "text": "F1 score at Cutoff", "type": "scatter", "x": [0.5, 0.55, 0.6000000000000001, 0.6500000000000001, 0.7000000000000002, 0.7500000000000002, 0.8000000000000003, 0.8500000000000003, 0.9000000000000004, 0.9500000000000004], "xaxis": "x2", "y": [86.301, 86.78, 87.423, 87.687, 88.286, 88.737, 89.31, 89.917, 90.696, 91.446], "yaxis": "y2"}, {"name": "", "text": "Data Retained", "type": "scatter", "x": [0.5, 0.55, 0.6000000000000001, 0.6500000000000001, 0.7000000000000002, 0.7500000000000002, 0.8000000000000003, 0.8500000000000003, 0.9000000000000004, 0.9500000000000004], "xaxis": "x2", "y": [99.83, 99.2, 98.51, 97.82, 97.02, 96.21, 94.99, 93.84, 91.93, 88.81], "yaxis": "y2"}],
                        {"annotations": [{"font": {"size": 16}, "showarrow": false, "text": "Profits by Cutoff", "x": 0.5, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}, {"font": {"size": 16}, "showarrow": false, "text": "Model Performance vs. Data Retained", "x": 0.5, "xanchor": "center", "xref": "paper", "y": 0.45, "yanchor": "bottom", "yref": "paper"}], "font": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}, "height": 800, "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Cost-Benefit for Probability Cutoffs"}, "width": 600, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "matches": "x2", "showticklabels": false}, "xaxis2": {"anchor": "y2", "domain": [0.0, 1.0], "tickfont": {"size": 14}, "title": {"text": "Probability Cutoffs"}}, "yaxis": {"anchor": "x", "domain": [0.55, 1.0], "tickfont": {"size": 12}, "title": {"text": "Profit"}}, "yaxis2": {"anchor": "x2", "domain": [0.0, 0.45], "tickfont": {"size": 12}, "title": {"text": "Percentage"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('088b46e5-4a45-4634-8933-395f256597b7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


--------------------------------------
**Profit continues to increase as we make the confidence threshold higher.** In other words, if we only take confident predictions we see increased profits because misclassifications are minimized and correct classifications are maximized. 

However, as we do so, we also reduce the number of reviews automatically answered. This adds labor costs, and so as volume increases it may become more desirable to have a lower cutoff (add a bit more risk) to keep volume low (reduce need to hire more employees as the company grows). 

<a id="cm_cut"></a>
## Cost-Profit Confusion Matrix @ .8 Cutoff

Confusion matrix for cost-profits after only taking predictions above .8 confidence.


```python
_ = plot_cost_cm(predictions, ['neg', 'pos', 'mixed'],
                 cost_matrix, neg_cut=0.8, pos_cut=0.8, mix_cut=0.8)
```

    Retained Data: 94.99%
    Macro-F1 Score: 0.8931035382420173
    Total Profits: 2730
    Confusion matrix, without normalization
    


![png](cost_benefit_analysis_files/cost_benefit_analysis_13_1.png)


<a id="cm"></a>
## Confusion Matrix @ .8 Cutoff

A normalized confusion matrix showing predictions for each class after a confidence cutoff of .8. 

Keep in mind that the classifier performs better than these current predictions (~6% incorrect labels, ~4% of those have been correctly classified).


```python
def remove_uncertain_samples(df, cols, neg_cut=0.5,
                             pos_cut=0.5, mix_cut=0.5):
    """
    Return normalized confusion matrix with applied thresholds.
    """
    
    neg_certain = list(df.index[(df[cols[0]] > neg_cut)])
    pos_certain = list(df.index[(df[cols[1]] > pos_cut)])
    mix_certain = list(df.index[df[cols[2]] > mix_cut])
    incl_total = list(set(neg_certain + pos_certain + mix_certain))
    print('Retained Data: {:0.2f}%'.format((len(incl_total)/len(df))*100))
    
    uncertain = df[~df.index.isin(incl_total)]
    certain = df[df.index.isin(incl_total)]

    f1_certain = f1_score(certain.label, certain.preds, average='macro')
    
    print('Macro-F1 Score: {}'.format(f1_certain))
    
    cm = confusion_matrix(certain.label, certain.preds)
    plot_confusion_matrix(cm, ['neg', 'pos', 'mixed'], normalize=True)
    
remove_uncertain_samples(predictions, ['neg', 'pos', 'mixed'],
                         neg_cut=0.8, pos_cut=0.8, mix_cut=0.8)
```

    Retained Data: 94.99%
    Macro-F1 Score: 0.8931035382420173
    Normalized confusion matrix
    


![png](cost_benefit_analysis_files/cost_benefit_analysis_15_1.png)


[Return to Index](#index)

