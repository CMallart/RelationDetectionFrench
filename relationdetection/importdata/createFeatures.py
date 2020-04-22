#!/usr/bin/env python
""" Process raw sentences into a csv file containing the features for both models.

This file calls FeaturesOurs and FeaturesXu. It is to be ru after importData, or after a SQLITE database following the same architecture has been processed.
The csv file with the data used in the article is available in data/featuresOurs.csv and data/featuresXu.csv
To be launched from the command line terminal.

  Typical usage example:

  cli example
"""

from . import FeaturesOurs 
from . import FeaturesXu

@click.command()
@click.argument('--db_file', type=click.File('rb'))
@click.option(
    '--output_file_ours',
    type=type=click.File('wb'),
    help='Path of the file where to keep the .csv containing the features for our model'
)
@click.option(
    '--output_file_xu',
    type=type=click.File('wb'),
    help='Path of the file where to keep the .csv containing the features for the model of Xu2015',
)
def create(db_file, output_file_ours, output_file_xu):
    """Creates the features files.

    Args:
        db_file: a path to a SQlite database containing the sentences and pairs of entities to pass to the model.
        output_file_ours: a path to a file where to store the features created for our model
        output_file_xu: a path to a file where to store the features created for the model of Xu2015

    Returns:
        Nothing

        If ouput_ours is included, we make a selection of sentences from the database, from which to extract features. 
        If output_xu is included, we either:
            - if output_ours is specified, we get the same pairs of entities for output_xu as for output_ours to stay consistent 
            - if no output_ours file is specified, we make a new selection of sentences for features_xu. (not recommended)
    """
    if output_file_ours is not None :
        if output_file_xu is not None:
            FeaturesOurs.FeaturesCreationSQL(dbfile= db_file, first_time=True).execute_SQL(ratio=10, 
                filepath=output_file_ours)

            FeaturesXu.FeaturesCreationSQL(dbfile= db_file, first_time=True).execute_SQL(ratio=10, 
                filepath = output_file_xu, 
                same_as=output_file_ours)
        
        else:
            FeaturesOurs.FeaturesCreationSQL(dbfile= db_file, first_time=True).execute_SQL(ratio=10, 
                filepath=output_file_ours)

    else :
        if output_file_xu is not None:
            FeaturesXu.FeaturesCreationSQL(dbfile= db_file, first_time=True).execute_SQL(ratio=10, 
                filepath = output_file_xu, 
                same_as=None)
        else:
            print("Error: need at least one of Xu or Ours output file to be specified.")


if __name__ == "__main__":
    create()
    