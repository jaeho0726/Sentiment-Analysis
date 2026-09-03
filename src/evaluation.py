import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error



def evaluate_mood_alignment(
    data,
    mood_column="Overall Mood",
    sentiment_column="Sentiment Score"
):
    """
    Evaluate alignment between self-reported mood and
    model-generated sentiment scores.

    Parameters
    ----------
    data : Dataframe containing mood and sentiment scores (pandas.DataFrame)

    mood_column : Column containing self-reported mood scores (str)

    sentiment_column : Column containing model sentiment scores (str)

    Returns
    -------
    evaluation_df : Valid observations with residual and absolute discrepancy columns (pandas.DataFrame)

    metrics : MAE, RMSE, mean bias, Pearson correlation, p-value, and sample size (dict)
    """

    evaluation_df = data.dropna(
        subset=[
            mood_column,
            sentiment_column
        ]
    ).copy()

    if len(evaluation_df) == 0:
        raise ValueError(
            "No valid mood/sentiment observations "
            "were found."
        )

    evaluation_df["Residual"] = (
        evaluation_df[sentiment_column]
        - evaluation_df[mood_column]
    )

    evaluation_df["Absolute Discrepancy"] = (
        evaluation_df["Residual"].abs()
    )

    mae = mean_absolute_error(
        evaluation_df[mood_column],
        evaluation_df[sentiment_column]
    )

    rmse = np.sqrt(
        mean_squared_error(
            evaluation_df[mood_column],
            evaluation_df[sentiment_column]
        )
    )

    mean_bias = (
        evaluation_df["Residual"].mean()
    )

    if len(evaluation_df) >= 2:
        correlation, p_value = pearsonr(
            evaluation_df[mood_column],
            evaluation_df[sentiment_column]
        )
    else:
        correlation = np.nan
        p_value = np.nan

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mean_bias": mean_bias,
        "pearson_r": correlation,
        "pearson_p_value": p_value,
        "n": len(evaluation_df)
    }

    return evaluation_df, metrics


def get_largest_discrepancies(
    evaluation_df,
    n=10
):
    # Return observations with the largest absolute difference between sentiment score and self-reported mood.

    columns = [
        "Date",
        "Overall Mood",
        "Sentiment Score",
        "Absolute Discrepancy"
    ]

    return (
        evaluation_df[columns]
        .sort_values(
            "Absolute Discrepancy",
            ascending=False
        )
        .head(n)
        .reset_index(drop=True)
    )


def print_alignment_metrics(metrics):
    # Print alignment metrics in a readable format.

    print(
        f"MAE: "
        f"{metrics['mae']:.3f}"
    )

    print(
        f"RMSE: "
        f"{metrics['rmse']:.3f}"
    )

    print(
        f"Mean Bias: "
        f"{metrics['mean_bias']:.3f}"
    )

    print(
        f"Pearson Correlation: "
        f"{metrics['pearson_r']:.3f}"
    )

    print(
        f"P-value: "
        f"{metrics['pearson_p_value']:.4f}"
    )

    print(
        f"Evaluation Samples: "
        f"{metrics['n']}"
    )