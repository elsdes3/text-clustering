#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Utilities to clean text data."""

# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
# pylint: disable=dangerous-default-value
# pylint: disable=missing-function-docstring
# pylint: disable=unused-argument


from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


def clean_content(
    content, stops=[], join=True, remove_num=False, min_len=None, max_len=None
):
    """Clean text data."""
    # Converting text to lowercase characters
    content = content.str.lower()
    # Removing HTML tags
    # content = content.apply(lambda x: re.sub(r"\<[^<>]*\>", "", x))
    content = content.str.replace(r"\<[^<>]*\>", "", regex=True)
    # Removing any character which does not match to letter, digit or
    # underscore
    # content = content.apply(lambda x: re.sub(r"^\W+|\W+$", " ", x))
    content = content.str.replace(r"^\W+|\W+$", " ", regex=True)
    # Removing space,newline,tab
    # content = content.apply(lambda x: re.sub(r"\s", " ", x))
    content = content.str.replace(r"\s", " ", regex=True)
    # Removing punctuation
    # content = content.apply(lambda x: re.sub(r"[^a-zA-Z0-9]", " ", x))
    content = content.str.replace(r"[^a-zA-Z0-9]", " ", regex=True)
    # Remove numbers
    if remove_num:
        content = content.str.replace(r"\d+", "", regex=True)
    # Tokenizing data
    # content = content.apply(lambda x: word_tokenize(x))
    content = content.apply(word_tokenize)
    # Remove stopwords
    if stops:
        content = content.apply(lambda x: [i for i in x if i not in stops])
    # Filter by length
    if min_len:
        content = content.apply(lambda x: [i for i in x if len(i) >= min_len])
    if max_len:
        content = content.apply(lambda x: [i for i in x if len(i) <= max_len])
    # Join
    if join:
        # Join list into single space-separated string
        content = content.str.join(" ")
    return content


class TextCleaner(BaseEstimator, TransformerMixin):
    """Clean text."""

    def __init__(
        self,
        text_col_name="text",
        join=True,
        remove_num=False,
        stops=[],
        min_len=None,
        max_len=None,
    ):
        self.text_col_name = text_col_name
        self.join = join
        self.remove_num = remove_num
        self.stops = stops
        self.min_len = min_len
        self.max_len = max_len

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = clean_content(
            X[self.text_col_name],
            self.stops,
            self.join,
            self.remove_num,
            self.min_len,
            self.max_len,
        )
        # X_transformed = X_transformed.apply(lambda x: " ".join(x))
        return X_transformed
