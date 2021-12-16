#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Workflow Utilities to run End-to-End Data Retrieval."""

# pylint: disable=invalid-name,logging-fstring-interpolation

import os
from glob import glob
from typing import List
from zipfile import ZipFile

import nltk
import pandas as pd
from prefect import flow, task
from prefect.task_runners import DaskTaskRunner
from prefect.utilities.logging import get_logger


def get_state_result(state):
    """Get result from Prefect state object."""
    return state.result()


@task
def extract(zip_filepaths: List[str], topics: List[str]) -> pd.DataFrame:
    """Unzip data archives for each topic."""
    logger = get_logger()
    for f, t in zip(zip_filepaths, topics):
        fpath = f"data/raw/{t}.csv"
        if not os.path.exists(fpath):
            with ZipFile(f) as zfile:
                logger.info(f"Unzip {os.path.basename(f)} from archive...")
                zfile.extractall("data/raw")
                logger.info("Done.")
        else:
            logger.info(f"Found extracted file at {fpath}. Did nothing.")
    return glob("data/raw/*.csv")


@task
def read_single_csv(filepath: str, topic: str) -> pd.DataFrame:
    """Load a single CSV file."""
    logger = get_logger()
    logger.info(f"Reading CSV at {filepath}...")
    df = pd.read_csv(filepath).assign(topic=topic)
    logger.info("Done.")
    return df


@flow(task_runner=DaskTaskRunner(), name="Combine data")
def transform(data_filepaths: List[str], topics: List[str]) -> pd.DataFrame:
    """Load multiple CSV files."""
    dfs_state = []
    for f, t in zip(data_filepaths, topics):
        state = read_single_csv(f, t)
        dfs_state.append(state)
    return dfs_state


@task
def load(dfs_list: List[pd.DataFrame], data_dir: str) -> None:
    """Combine list of DataFrames and export to a Parquet file."""
    logger = get_logger()
    parquet_filepath = f"{data_dir}/text_clustering_data.parquet.gzip"
    if not os.path.exists(parquet_filepath):
        logger.info("Exporting data to Parquet file...")
        df = pd.concat(dfs_list, ignore_index=True).sample(frac=1)
        df.to_parquet(
            parquet_filepath,
            engine="auto",
            index=False,
            compression="gzip",
        )
        logger.info("Done.")
    else:
        logger.info(
            f"Found Parquet file locally at {parquet_filepath}. Did nothing."
        )


@task
def delete_file(filepath: str) -> None:
    """Delete file from filepath."""
    logger = get_logger()
    logger.info(f"Deleting intermediate CSV from {filepath}...")
    os.remove(filepath)
    logger.info("Done.")


@flow(task_runner=DaskTaskRunner(), name="Delete intermediate data")
def clean_up_files(data_filepaths: List[str]) -> None:
    """Clean up intermediate data."""
    for f in data_filepaths:
        delete_file(f)


@task
def get_nltk_stopwords(path_to_stopwords_dir: str) -> None:
    """Download NLTK stopwords."""
    logger = get_logger()
    if not os.path.isdir(path_to_stopwords_dir):
        logger.info(
            "Downloading NLTK stopwords locally to "
            f"{path_to_stopwords_dir}..."
        )
        nltk.download("stopwords")
        logger.info("Done.")
    else:
        logger.info(
            f"Found stopwords locally at {path_to_stopwords_dir}. "
            "Did nothing."
        )


@flow(name="Data Extraction Workflow")
def retrieve_data(
    zip_filepaths: List[str],
    topics: List[str],
    data_dir: str,
    path_to_stopwords_dir: str,
) -> None:
    """Run end-to-end data acquisition workflow."""
    # Extract
    data_filepaths = extract(zip_filepaths, topics)

    # Transform
    subflow_state = transform(data_filepaths, topics)

    # Load
    dfs = list(map(get_state_result, tuple(subflow_state.result())))
    load(dfs, data_dir)

    # Data cleanup
    clean_up_files(data_filepaths)

    # Download NLTK stopwords
    get_nltk_stopwords(path_to_stopwords_dir)
