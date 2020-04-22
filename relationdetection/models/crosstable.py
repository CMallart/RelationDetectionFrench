#!/usr/bin/env python
"""Plot the table of distribution of entities couples in feature files.

For develop purposes

  Typical usage example:

  cli example
"""

import click
import pandas as pd
import numpy as numpy 
import seaborn as sns
import matplotlib.pyplot as plt


def make_crosstable(feats_file):
    """For a feature_file containing the pairs of entites for each sentence (after running createFeatures), make the crosstable of the entities couples."""
    df = pd.read_csv(feats_file)
    df.fillna(value="", inplace=True)
    df['ent1type'] = df['ent1type'].apply(lambda x: "other" if x=="" else x)
    df['ent2type'] = df['ent2type'].apply(lambda x: "other" if x=="" else x)
    cross = pd.crosstab(df.ent2type, df.ent1type, rownames=['Type e2'], colnames=['Type e1'], dropna=False)
    return(cross)


def plot_crosstable(cross):
    """Plot a crosstable."""
    cm = sns.light_palette("blue", as_cmap=True)
    plt = cross.style.background_gradient(cmap=cm, axis=None, high =-.95)
    plt.show()


@click.command()
@click.argument('--features_file', type=click.File('rb'))
def run(features_file):
    crosstableentities = make_crosstable(feats_file =features_file)
    plot_crosstable(crosstableentities)

if __name__ == "__main__" :
    run()