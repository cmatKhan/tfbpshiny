# %%
import logging

import pandas as pd
import plotly.graph_objects as go
from scipy.stats import binom, binomtest
from scipy.stats._result_classes import BinomTestResult

from tfbpshiny.utils.source_name_lookup import get_source_name_dict

logger = logging.getLogger("shiny")

# Global dictionary for responsiveness definitions by expression source
RESPONSIVENESS_DEFINITIONS = {
    "kemmeren_tfko": "p-value < 0.05",
    "mcisaac_oe": "|log2fc| > 0",
}


def parse_binomtest_results(binomtest_obj: BinomTestResult, **kwargs):
    """
    Parses the results of a binomtest into a tuple of floats.

    This function takes the results of a binomtest and returns a tuple of
    floats containing the response ratio, p-value, and confidence interval
    bounds.

    :param binomtest_obj: The results of a binomtest for a single rank bin.
        Additional keyword arguments: Additional keyword arguments are passed
        to the proportional_ci method of the binomtest object.

    :return: A tuple of floats containing the response ratio, p-value, and
        confidence interval bounds.

    :examples:

    .. code-block:: python

        from scipy.stats import binomtest
        from rank_response_plot.logic import parse_binomtest_results

        result = binomtest(k=1, n=2, p=0.5, alternative='greater')
        parse_binomtest_results(result, confidence_level=0.95)
        # Output: (0.5, <p-value>, <ci_lower>, <ci_upper>)

    """
    binom_res = binomtest_obj.proportion_ci(
        confidence_level=kwargs.get("confidence_level", 0.95),
        method=kwargs.get("method", "exact"),
    )
    return (
        binomtest_obj.statistic,
        binomtest_obj.pvalue,
        binom_res.low,
        binom_res.high,
    )


def compute_rank_response(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Computes rank-based statistics and binomial test results for a DataFrame.

    This function groups the DataFrame by 'rank_bin' and aggregates it to
    calculate the number of responsive items in each rank bin, as well as
    various statistics related to a binomial test.  It calculates the
    cumulative number of successes, response ratio, p-value, and confidence
    intervals for each rank bin.

    :param df: DataFrame containing the columns 'rank_bin',
        'responsive', and 'random'. 'rank_bin' is an integer representing
        the rank bin, 'responsive' is a boolean indicating responsiveness,
        and 'random' is a float representing the random expectation.
    :param kwargs: Additional keyword arguments are passed
        to the binomtest function, including arguments to the
        proportional_ci method of the BinomTestResults object (see scipy
        documentation for details)

    :return: A DataFrame indexed by 'rank_bin' with columns for the
            number of responsive items in each bin ('n_responsive_in_rank'),
            cumulative number of successes ('n_successes'), response ratio
            ('response_ratio'), p-value ('pvalue'), and confidence interval
            bounds ('ci_lower' and 'ci_upper').

    :examples:

    .. code-block:: python
        df = pd.DataFrame({'rank_bin': [1, 1, 2],
        ...                    'responsive': [True, False, True],
        ...                    'random': [0.5, 0.5, 0.5]})
        compute_rank_response(df)
        # Returns a DataFrame with rank-based statistics and binomial
        # test results.

    """
    rank_response_df = (
        df.groupby("rank_bin")
        .agg(
            n_responsive_in_rank=pd.NamedAgg(column="responsive", aggfunc="sum"),
            random=pd.NamedAgg(column="random", aggfunc="first"),
        )
        .reset_index()
    )

    rank_response_df["n_successes"] = rank_response_df["n_responsive_in_rank"].cumsum()

    # Binomial Test and Confidence Interval
    rank_response_df[["response_ratio", "pvalue", "ci_lower", "ci_upper"]] = (
        rank_response_df.apply(
            lambda row: parse_binomtest_results(
                binomtest(
                    int(row["n_successes"]),
                    int(row.rank_bin),
                    float(row["random"]),
                    alternative=kwargs.get("alternative", "two-sided"),
                ),
                **kwargs,
            ),
            axis=1,
            result_type="expand",
        )
    )

    return rank_response_df


def binom_ci(trials, random_prob, alpha=0.05):
    """
    Calculate the confidence interval for a binomial distribution. This function
    calculates the confidence interval for a binomial distribution using the binomial
    cumulative distribution function (CDF).

    :param trials: The number of trials (n).
    :param random_prob: The probability of success (p).
    :param alpha: The significance level (default is 0.05).
    :return: A tuple containing the lower and upper bounds of the confidence interval.

    """
    lower_bound = binom.ppf(alpha / 2, trials, random_prob) / trials
    upper_bound = binom.ppf(1 - alpha / 2, trials, random_prob) / trials
    return lower_bound, upper_bound


def process_plot_data(data: pd.DataFrame, n_bins: int = 150, step: int = 5) -> dict:
    """
    Process the data for plotting. This function filters the data for a specific key,
    computes rank-based statistics, and prepares the data for plotting.

    :param data: The DataFrame containing the data to be processed.
    :param n_bins: The number of bins to consider (default is 150).
    :param step: The step size for the x-axis (default is 5).
    :return: A dictionary containing the processed data for plotting.

    """
    if not isinstance(n_bins, int) or n_bins <= 0:
        logger.error(
            f"n_bins {n_bins} is invalid. It must be a positive "
            "integer. Setting to 150."
        )
        n_bins = 150
    subset_data = data[data["rank_bin"] <= n_bins]
    rr_summary = compute_rank_response(subset_data)

    bin_vector = pd.Series(range(5, n_bins + 1, step), name="rank_bin")
    random_vector = [rr_summary["random"][0]] * len(bin_vector)
    plot_data = {
        # don't extract the bins from the df directly b/c there may fewer
        # bins than n_bins and that can make the random line truncated
        "x": bin_vector,
        "y": rr_summary["response_ratio"],
        # ensure that the vector of random is the same length as x
        "random_y": random_vector,
        "ci": (
            bin_vector.apply(lambda n: binom_ci(n, rr_summary["random"][0]))
            if "random" in rr_summary
            else None
        ),
    }

    return plot_data


def prepare_rank_response_data(rr_dict: dict) -> dict:
    """
    Prepare rank response data for plotting.

    :param rr_dict: Dictionary containing rank response data.
    :return: Dictionary containing processed data for plotting.

    """
    metadata = rr_dict.get("metadata", pd.DataFrame())
    data_dict = rr_dict.get("data", {})

    # Use list comprehension to generate plots
    plots: dict = {}
    source_name_dict = get_source_name_dict()
    for _, row in metadata.iterrows():
        id = str(row["id"])
        data = data_dict.get(id)

        expression_id = str(row["expression"])
        promotersetsig_id = str(row["promotersetsig"])
        expression_source = str(row["expression_source"])

        plot_data = process_plot_data(data)
        plot_data["datasource"] = source_name_dict.get(
            row["binding_source"], row["binding_source"]
        )
        plot_data["expression_source"] = expression_source

        plots.setdefault(expression_id, {}).update({promotersetsig_id: plot_data})

    return plots


def add_traces_to_plot(fig, promotersetsig, add_random, **kwargs):
    """Add traces to a Plotly figure based on plot data."""
    # Add the main line for the promoterset signal
    fig.add_trace(
        go.Scatter(
            x=kwargs["x"],
            y=kwargs["y"],
            mode="lines",
            name=f"{kwargs['datasource']}; {promotersetsig}",
            legendrank=-int(promotersetsig),
            meta={
                "promotersetsig": promotersetsig,
            },
        )
    )

    if add_random:
        # Add the random line
        fig.add_trace(
            go.Scatter(
                x=kwargs["x"],
                y=kwargs["random_y"],
                mode="lines",
                name="Random",
                line=dict(dash="dash", color="black"),
                legendrank=-(2**63),
            )
        )

        if kwargs["ci"] is not None:
            ci_lower = kwargs["ci"].apply(lambda x: x[0])
            ci_upper = kwargs["ci"].apply(lambda x: x[1])

            # Add confidence interval lower bound
            fig.add_trace(
                go.Scatter(
                    x=kwargs["x"],
                    y=ci_lower,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

            # Add confidence interval upper bound and shade the area
            fig.add_trace(
                go.Scatter(
                    x=kwargs["x"],
                    y=ci_upper,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.3)",
                    line=dict(width=0),
                    showlegend=False,
                )
            )


def create_rank_response_replicate_plot(plots_dict):
    """Generate a dictionary of Plotly figures from the prepared rank response data."""
    output_dict = {}

    for expression_id, promotersetsig_dict in plots_dict.items():
        fig = go.Figure()
        add_random = True
        expression_source = None

        for promotersetsig_id, plot_data in promotersetsig_dict.items():
            # Get expression_source from the first plot_data entry
            if expression_source is None:
                expression_source = plot_data.get("expression_source")

            # Use the helper function to add traces to the plot
            add_traces_to_plot(fig, promotersetsig_id, add_random, **plot_data)
            add_random = False  # Add random line only once

        # Get responsiveness definition for subtitle
        responsiveness_def = RESPONSIVENESS_DEFINITIONS.get(str(expression_source), "")

        # Create title with subtitle if responsiveness definition exists
        if responsiveness_def:
            title_text = (
                f"Rank Response for Expression ID {expression_id}<br>"
                f"<span style='font-size:14px; color:gray;'>Responsiveness: "
                f"{responsiveness_def}</span>"
            )
        else:
            title_text = f"Rank Response for Expression ID {expression_id}"

        # Update the layout of the figure
        fig.update_layout(
            title={
                "text": title_text,
                "x": 0.5,
            },
            yaxis_title="# Responsive / # Genes",
            xaxis_title="Number of Genes, Ranked by Binding Score",
            xaxis=dict(tick0=0, dtick=5, range=[0, 150]),  # Set x-axis ticks and range
            yaxis=dict(
                tick0=0, dtick=0.1, range=[0, 1.0]
            ),  # Set y-axis ticks and range
            margin=dict(t=80, b=50, l=60, r=30),  # Add more top margin for subtitle
        )

        output_dict[expression_id] = fig

    return output_dict


# ## Example

# Set up the environment
# import dotenv
# from tfbpapi import *

# dotenv.load_dotenv("/home/chase/code/tfbpshiny/.env", override=True)

# # # configure the logger to print to console
# import logging

# logging.basicConfig(level=logging.DEBUG)

# rr_api = RankResponseAPI()

# rr_api.pop_params()
# # "expression_conditions": "expression_source=mcisaac_oe,time=15"
# rr_api.push_params(
#     {
#         "regulator_symbol": "DEP1",
#         "expression_source": "kemmeren_tfko",
#     }
# )

# rr_dict = await rr_api.read(retrieve_files=True)

# plots = prepare_rank_response_data(rr_dict)

# x = create_rank_response_replicate_plot(plots)

# %%
# to show data, do x.get(<id>).show()

# %%
