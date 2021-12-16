#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Workflow utilities to explore learned clusters."""

# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
# pylint: disable=dangerous-default-value

import re
from typing import Dict, List, Union

import jsonpickle
import pandas as pd
from prefect.utilities.logging import get_logger


def show_top_ten_words(d_top_ten: Dict) -> str:
    """Print the top ten words in each cluster."""
    astr = ""
    for k, v in d_top_ten.items():
        astr += f"Cluster {k}: {', '.join(v)}\n"
    return astr


def get_top_10_terms(pipe_str: str, params_dict: Dict) -> Dict:
    """Get the top ten words in each cluster."""
    logger = get_logger()
    logger.info("Getting top 10 tokens (by TFIDF weight) per cluster...")
    pipe = jsonpickle.decode(pipe_str)
    n_clusters = params_dict["clusterer__n_clusters"]
    # Get the cluster centroids
    cluster_centers = pipe.named_steps["clusterer"].cluster_centers_
    order_centroids = cluster_centers.argsort()[:, ::-1]

    # Get all words for each cluster
    terms = pipe.named_steps["vectorizer"].get_feature_names_out()

    # Print top 10 words per cluster
    d_top_ten = {}
    for i in range(n_clusters):
        t10_terms = []
        for ind in order_centroids[i, :10]:
            t10_terms.append(terms[ind])
        d_top_ten[i] = t10_terms
    logger.info("Done")
    return d_top_ten


def get_cluster_numbers(pipe_str: str) -> List:
    """Get cluster numbers."""
    logger = get_logger()
    logger.info("Getting assigned cluster numbers from Pipeline attribute...")
    pipe = jsonpickle.decode(pipe_str)
    cluster_numbers = pipe.named_steps["clusterer"].labels_.tolist()
    logger.info("Done")
    return cluster_numbers


def train(
    pipe_str: str,
    cleaned_outputs: List[Union[List, str]],
    params_dict: Dict,
) -> str:
    """Cluster text data."""
    logger = get_logger()
    logger.info(f"Training with {str(params_dict)}...")
    corpus, _ = cleaned_outputs
    pipe = jsonpickle.decode(pipe_str)
    pipe.set_params(**params_dict)
    _ = pipe.fit(corpus)
    pipe_trained_str = jsonpickle.encode(pipe)
    logger.info("Done.")
    return pipe_trained_str


def get_cluster_posts(
    data: str,
    cluster_numbers: List,
    d_top_ten: Dict,
    params_dict: Dict,
    num_docs_to_read: int = 5,
) -> List[Dict]:
    """Get posts that belong to each cluster."""
    logger = get_logger()
    logger.info("Getting posts for each cluster...")
    n_clusters = params_dict["clusterer__n_clusters"]
    df = pd.read_json(data, orient="split")
    cluster_posts = []
    for cluster_idx in range(n_clusters):
        df_with_clusters = df.assign(cluster=cluster_numbers)[
            df.assign(cluster=cluster_numbers)["cluster"] == cluster_idx
        ].iloc[:num_docs_to_read]
        for k, post in df_with_clusters["content"].iteritems():
            d_cluster_post = {
                "index": k,
                "content": post,
                "num_clusters": n_clusters,
                "cluster": cluster_idx,
                "top_10_tokens": d_top_ten[cluster_idx],
                "params_str": str(params_dict),
            }
            cluster_posts.append(d_cluster_post)
    logger.info("Done")
    df_cluster_posts_str = pd.DataFrame.from_records(cluster_posts).to_json(
        orient="split"
    )
    return df_cluster_posts_str


def extract_clean_text_from_cluster_posts(
    df_str: str,
    d_top_ten: Dict,
    params_dict: Dict,
    cluster_numbers_ordered: List[int] = [2, 3, 1, 0, 4, 5],
) -> str:
    """Get text from all posts in a cluster."""
    logger = get_logger()
    df_with_clusters = pd.read_json(df_str, orient="split").set_index("index")
    astr = ""
    for q, cluster_idx in enumerate(cluster_numbers_ordered):
        astr += show_top_ten_words(d_top_ten)
        for post_num, (k, row) in enumerate(
            df_with_clusters.query(f"cluster == {cluster_idx}").iterrows(), 1
        ):
            params_dict_str = str(
                {k.split("__")[-1]: v for k, v in params_dict.items()}
            )
            logger.info(
                f"Getting post {post_num} (row index={k:,}) to read in "
                f"cluster {cluster_idx} found using {params_dict_str}..."
            )
            post_text_without_html = (
                re.sub(r"\<[^<>]*\>", "", row["content"])
                .replace("\n", "")
                .strip()
            )
            astr += (
                f"\nCluster = {cluster_idx}, Raw data index = {k:,}, "
                f"Hyper-Parameters = {str(params_dict)}\n"
                f"{post_text_without_html}\n"
            )
            logger.info("Done")
        if q < len(cluster_numbers_ordered) - 1:
            astr += "\n"
    return astr
