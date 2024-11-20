import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from config import MIN_PARAM, MAX_PARAM


def plot(
    posterior_mean_np,
    posterior_variance_np,
    acquisition_samples_np,
    x_values_np,
    design_points_np,
    design_values_np,
    candidate_np,
    candidate_acqu_np,
):
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Posterior", "Acquisition Function"),
    )

    # Add posterior mean line
    fig.add_trace(
        go.Scatter(
            x=x_values_np,
            y=posterior_mean_np,
            mode="lines",
            name="Posterior Mean",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Add variance as a shaded area
    fig.add_trace(
        go.Scatter(
            x=x_values_np,
            y=posterior_mean_np + np.sqrt(posterior_variance_np),
            mode="lines",
            name="Posterior Mean + StD",
            line=dict(color="rgba(255, 255, 255, 0)"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values_np,
            y=posterior_mean_np - np.sqrt(posterior_variance_np),
            mode="lines",
            name="Posterior Mean - StD",
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
        ),
        row=1,
        col=1,
    )

    # Add acquisition function line
    fig.add_trace(
        go.Scatter(
            x=x_values_np,
            y=acquisition_samples_np,
            mode="lines",
            name="Acquisition Function",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    # Add design points
    fig.add_trace(
        go.Scatter(
            x=design_points_np,
            y=design_values_np,
            mode="markers",
            name="Design Points",
            marker=dict(color="green", size=8),
        ),
        row=1,
        col=1,
    )

    # Add next candidate
    fig.add_trace(
        go.Scatter(
            x=candidate_np,
            y=candidate_acqu_np,
            mode="markers",
            name="Next Candidate",
            marker=dict(color="orange", size=10, symbol="x"),
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title="Bayesian Optimization: Posterior and Acquisition Function",
        xaxis_title="Parameter",
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(range=[MIN_PARAM - 1, MAX_PARAM + 1], row=1, col=1)
    fig.update_xaxes(range=[MIN_PARAM - 1, MAX_PARAM + 1], row=2, col=1)

    # Show the figure
    return fig
