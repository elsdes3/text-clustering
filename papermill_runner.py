#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Programmatic execution of notebooks."""

# pylint: disable=invalid-name

import os
from datetime import datetime
from typing import Dict, List

import papermill as pm

PROJ_ROOT_DIR = os.getcwd()
output_notebook_dir = os.path.join(PROJ_ROOT_DIR, "executed_notebooks")
raw_data_dir = os.path.join(PROJ_ROOT_DIR, "data", "raw")
topics = ["biology", "cooking", "crypto", "diy", "robotics", "travel"]

zero_nb_name = "01_get_data.ipynb"
one_nb_name = "02_eda.ipynb"


zero_dict = dict(
    raw_data_dir="data/raw",
    topics=topics,
    zip_filepaths=[f"{raw_data_dir}/{t}.csv.zip" for t in topics],
    nltk_stopwords_dir=os.path.join(
        os.path.expanduser("~"), "nltk_data", "corpora", "stopwords"
    ),
)
one_dict = dict(
    topics=topics,
    num_samples_to_use=86000,
    raw_data_filepath=f"{raw_data_dir}/text_clustering_data.parquet.gzip",
    n_clusters=6,
    kmeans_random_state=42,
)


def papermill_run_notebook(
    nb_dict: Dict, output_notebook_directory: str = "executed_notebooks"
) -> None:
    """Execute notebook with papermill"""
    for notebook, nb_params in nb_dict.items():
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_nb = os.path.basename(notebook).replace(
            ".ipynb", f"-{now}.ipynb"
        )
        print(
            f"\nInput notebook path: {notebook}",
            f"Output notebook path: {output_notebook_directory}/{output_nb} ",
            sep="\n",
        )
        for key, val in nb_params.items():
            print(key, val, sep=": ")
        pm.execute_notebook(
            input_path=notebook,
            output_path=f"{output_notebook_directory}/{output_nb}",
            parameters=nb_params,
        )


def run_notebooks(
    notebooks_list: List, output_nb_dir: str = "executed_notebooks"
) -> None:
    """Execute notebooks from CLI.
    Parameters
    ----------
    nb_dict : List
        list of notebooks to be executed
    Usage
    -----
    > import os
    > PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
    > one_dict_nb_name = "a.ipynb
    > one_dict = {"a": 1}
    > run_notebook(
          notebook_list=[
              {os.path.join(PROJ_ROOT_DIR, one_dict_nb_name): one_dict}
          ]
      )
    """
    for nb in notebooks_list:
        papermill_run_notebook(
            nb_dict=nb, output_notebook_directory=output_nb_dir
        )


if __name__ == "__main__":
    nb_dict_list = [zero_dict, one_dict]
    nb_name_list = [zero_nb_name, one_nb_name]

    notebook_list = [
        {os.path.join(PROJ_ROOT_DIR, nb_name): nb_dict}
        for nb_dict, nb_name in zip(nb_dict_list, nb_name_list)
    ]

    run_notebooks(
        notebooks_list=notebook_list,
        output_nb_dir=output_notebook_dir,
    )
