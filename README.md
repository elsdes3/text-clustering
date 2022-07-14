# Clustering Text of StackExchange Posts

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/elsdes3/text-clustering)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elsdes3/text-clustering/master/0_get_data.ipynb)
![CI](https://github.com/elsdes3/text-clustering/workflows/CI/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/mit)
![OpenSource](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![prs-welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![pyup](https://pyup.io/repos/github/elsdes3/text-clustering/shield.svg)

## [Table of Contents](#table-of-contents)
1. [About](#about)
2. [Notebooks](#notebooks)
3. [Notes](#notes)
4. [Project Organization](#project-organization)

## [About](#about)

Clustering of the text in posts from [stackexchange data covering multiple topics](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags/data).

## [Notebooks](#notebooks)
1. `0_get_data.ipynb` ([view](https://nbviewer.org/github/elsdes3/text-clustering/blob/main/0_get_data.ipynb))
   - extracts and combines previously download `.zip` files from the [data source](https://www.kaggle.com/competitions/transfer-learning-on-stack-exchange-tags/data)
2. `1_text_clustering.ipynb` ([view](https://nbviewer.org/github/elsdes3/text-clustering/blob/main/1_text_clustering.ipynb))
   - text data processing
   - clustering of processed data

## [Notes](#notes)
1. Text processing using TFIDF Vectorization is completed and shown in `1_text_clustering_tfidf.ipynb`. Processing using other vectorization methods is to be done.

## [Project Organization](#project-organization)

    ├── LICENSE
    ├── .env                          <- environment variables (verify this is in .gitignore)
    ├── .gitignore                    <- files and folders to be ignored by version control system
    ├── .pre-commit-config.yaml       <- configuration file for pre-commit hooks
    ├── .github
    │   ├── workflows
    │       └── main.yml              <- configuration file for CI build on Github Actions
    ├── Makefile                      <- Makefile with commands like `make lint` or `make build`
    ├── README.md                     <- The top-level README for developers using this project.
    ├── environment.yml               <- configuration file to create environment to run project on Binder
    ├── data
    │   ├── raw                       <- Scripts to download or generate data
    |   └── processed                 <- merged and filtered data, sampled at daily frequency
    ├── *.ipynb                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                    and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
    ├── requirements.txt              <- base packages required to execute all Jupyter notebooks (incl. jupyter)
    ├── src                           <- Source code for use in this project.
    │   ├── __init__.py               <- Makes src a Python module
    │   └── *.py                      <- Scripts to use in analysis for pre-processing, visualization, training, etc.
    ├── papermill_runner.py           <- Python functions that execute system shell commands.
    └── tox.ini                       <- tox file with settings for running tox; see https://tox.readthedocs.io/en/latest/

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
