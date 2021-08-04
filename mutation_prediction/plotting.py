import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score

from mutation_prediction.data import Dataset
from mutation_prediction.models import Model

plotly_blue = "#636EFA"
plotly_red = "#EF553B"


def evaluate_performance(title, model, train, test, value_range):
    y_true_train = train.get_y()
    y_pred_train = model.predict(train)
    y_true_test = test.get_y()
    y_pred_test = model.predict(test)

    print(
        "%s:\n"
        "Train: R² = %.3f, RMSE = %.3f\n"
        "Test:  R² = %.3f, RMSE = %.3f"
        % (
            title,
            r2_score(y_true_train, y_pred_train),
            mean_squared_error(y_true_train, y_pred_train, squared=False),
            r2_score(y_true_test, y_pred_test),
            mean_squared_error(y_true_test, y_pred_test, squared=False),
        )
    )

    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True)
    max_mutations = max(np.max(train.get_num_mutations()), np.max(test.get_num_mutations()))
    fig.add_trace(
        go.Scatter(
            x=y_pred_train,
            y=y_true_train,
            mode="markers",
            name="test",
            marker=dict(
                color=train.get_num_mutations(), showscale=True, cmin=0, cmax=max_mutations
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=value_range, y=value_range, mode="lines", showlegend=False, line=dict(color="#000000")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=y_pred_test,
            y=y_true_test,
            mode="markers",
            name="train",
            marker=dict(
                color=test.get_num_mutations(), showscale=False, cmin=0, cmax=max_mutations
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=value_range,
            y=value_range,
            mode="lines",
            showlegend=False,
            line=dict(color="#000000"),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=title,
        xaxis1_title="train prediction",
        xaxis1_range=value_range,
        yaxis1_title="train ground truth",
        yaxis1_range=value_range,
        xaxis2_title="test prediction",
        xaxis2_range=value_range,
        yaxis2_title="test ground truth",
        yaxis2_range=value_range,
        showlegend=False,
        width=800,
        height=500,
    )
    fig.show()

    num_mutations = np.arange(max_mutations + 1)
    r2_per_num_mutations = np.zeros((len(num_mutations), 2), dtype=np.float64)
    count_per_num_mutations = np.zeros((len(num_mutations), 2), dtype=np.int32)
    for n in range(len(num_mutations)):
        mask_train = train.get_num_mutations() == n
        count_per_num_mutations[n, 0] = np.count_nonzero(mask_train)
        if count_per_num_mutations[n, 0] >= 2:
            r2_per_num_mutations[n, 0] = r2_score(
                y_true_train[mask_train], y_pred_train[mask_train]
            )
        mask_test = test.get_num_mutations() == n
        count_per_num_mutations[n, 1] = np.count_nonzero(mask_test)
        if count_per_num_mutations[n, 1] >= 2:
            r2_per_num_mutations[n, 1] = r2_score(y_true_test[mask_test], y_pred_test[mask_test])
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Bar(
            x=num_mutations,
            y=r2_per_num_mutations[:, 0],
            name="train",
            marker_color=plotly_blue,
        ),
        col=1,
        row=1,
    )
    fig.add_trace(
        go.Bar(
            x=num_mutations,
            y=r2_per_num_mutations[:, 1],
            name="test",
            marker_color=plotly_red,
        ),
        col=1,
        row=1,
    )
    fig.add_trace(
        go.Bar(
            x=num_mutations,
            y=count_per_num_mutations[:, 0],
            name="train",
            marker_color=plotly_blue,
        ),
        col=2,
        row=1,
    )
    fig.add_trace(
        go.Bar(
            x=num_mutations,
            y=count_per_num_mutations[:, 1],
            name="test",
            marker_color=plotly_red,
        ),
        col=2,
        row=1,
    )
    fig.update_layout(
        width=800,
        height=500,
        yaxis1_range=[-1, 1],
    )
    fig.show()


def correlation_plot(model: Model, train: Dataset, test: Dataset):
    both_predictions = model.predict(train + test)
    both_true = (train + test).get_y()
    pred_train = model.predict(train)
    pred_test = model.predict(test)
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=pred_train,
            y=train.get_y(),
            mode="markers",
            name="train",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_test,
            y=test.get_y(),
            mode="markers",
            name="test",
        )
    )
    min_y = min(np.min(both_predictions), np.min(both_true)) * 1.05
    max_y = max(np.max(both_predictions), np.max(both_true)) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[min_y, max_y],
            y=[min_y, max_y],
            mode="lines",
            marker=dict(color="black"),
            name="ideal",
        )
    )
    fig.update_layout(
        title="Correlation Plot (Train R² = %.3f, Test R² = %.3f)"
        % (
            r2_score(train.get_y(), pred_train),
            r2_score(test.get_y(), pred_test),
        ),
        xaxis_range=[min_y, max_y],
        xaxis_dtick=0.2,
        xaxis_title="prediction",
        yaxis_range=[min_y, max_y],
        yaxis_dtick=0.2,
        yaxis_title="true value",
        width=620,
        height=600,
    )
    return fig
