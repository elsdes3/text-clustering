#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Workflow Utilities to run End-to-End Text Clustering Experiment."""

# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
# pylint: disable=too-many-arguments

from typing import Dict, List

import jsonpickle
import pandas as pd
from prefect import flow, task
from prefect.task_runners import DaskTaskRunner, SequentialTaskRunner
from prefect.utilities.logging import get_logger

import src.clusters.explorers as expl


@flow(name="Preprocess Text Data")
def preprocess_text(pipe_cleaner_str: str, data_str: str) -> List:
    """Perform data cleaning."""
    logger = get_logger()
    logger.info("Cleaning...")
    pipe = jsonpickle.decode(pipe_cleaner_str)
    df = pd.read_json(data_str, orient="split")
    corpus = pipe.fit_transform(df).tolist()
    pipe_clean_str = jsonpickle.encode(pipe)
    logger.info("Done")
    return [corpus, pipe_clean_str]


@flow(task_runner=SequentialTaskRunner(), name="Combine list of DataFrames")
def combine_all_summary_dfs(dfs_str: List[str]) -> str:
    """Combine list of DataFrames into single DataFrame."""
    logger = get_logger()
    logger.info(f"Combining {len(dfs_str)} Summary DataFrames...")
    dfs_list = []
    for df_str in dfs_str:
        df = pd.read_json(df_str, orient="split")
        dfs_list.append(df)
    df_str = pd.concat(dfs_list).to_json(orient="split")
    logger.info("Done")
    return df_str


@task
def combine_all_strs(strs_list: List[str]) -> str:
    """Concatenate all cleaned posts' text."""
    logger = get_logger()
    logger.info("Combining string of posts to read...")
    output_str = "\n\n".join(strs_list)
    logger.info("Done")
    return output_str


@task
def cluster_data(
    pipe_str: str,
    data_str: str,
    cleaned_outputs: List,
    params_dict: Dict,
    num_docs_to_read: int = 5,
) -> List:
    """Cluster data, get top 10 terms and get posts per cluster."""
    pipe_trained_str = expl.train(pipe_str, cleaned_outputs, params_dict)
    d_top_ten = expl.get_top_10_terms(pipe_trained_str, params_dict)
    cluster_assignments = expl.get_cluster_numbers(pipe_trained_str)
    df_cluster_posts_str = expl.get_cluster_posts(
        data_str,
        cluster_assignments,
        d_top_ten,
        params_dict,
        num_docs_to_read,
    )
    astrr = expl.extract_clean_text_from_cluster_posts(
        df_cluster_posts_str, d_top_ten, params_dict
    )
    return {"df_cluster_posts_str": df_cluster_posts_str, "astr": astrr}


@flow(task_runner=DaskTaskRunner(), name="Cluster Data")
def perform_clustering(
    pipe_str: str,
    data_str: str,
    param_grid_list: List[Dict],
    cleaned_outputs_list: List,
    num_docs_to_read: int = 5,
) -> List:
    """Perform Clustering on text data."""
    outputs = []
    for params_dict in param_grid_list:
        output = cluster_data(
            pipe_str,
            data_str,
            cleaned_outputs_list,
            params_dict,
            num_docs_to_read,
        )
        outputs.append(output)
    return outputs


@flow(name="Run through complete Clustering Workflow")
def run_clustering_trials(
    pipe_cleaner_str: str,
    pipe_str: str,
    data_str: str,
    param_grid_list: List[Dict],
    num_docs_to_read: int = 5,
) -> List:
    """Run end-to-end clustering workflow."""
    cleaner_state = preprocess_text(pipe_cleaner_str, data_str)
    # print(len(cleaner_state.result()[0]))
    subflow_state = perform_clustering(
        pipe_str,
        data_str,
        param_grid_list,
        cleaner_state.result(),
        num_docs_to_read,
    )
    df_combo_str = combine_all_summary_dfs(
        [
            subflow_state.result()[k].result()["df_cluster_posts_str"]
            for k, _ in enumerate(param_grid_list)
        ]
    )
    combo_str = combine_all_strs(
        [
            subflow_state.result()[k].result()["astr"]
            for k, _ in enumerate(param_grid_list)
        ]
    )
    return [df_combo_str, combo_str]
