#!/usr/bin/env python

"""Extract data from a dump of Wikipedia articles into a structured SQLite database containing pairs of entities and sentences.

This file is to be run as the first step of the pipeline if not SQLite database containing the right fields is available.
Our pre-processed data, made with this script, is avalaible in data/small_wiki.db.
To be run from the command line interface.
Script can take long. For this, can be stopped qnd re-launched by changing the first_time argument as false in the script (advanced)
"""

from .wikitransformer import WikiTransformer
from .SentencesExtraction import DocumentSentences
from .ExtractEntitiesInSentences import EntitiesBase
from .WikiDataWikiPediaEntitiesLink import LinkWikiPediaData
from .EntitiesCouplesExtraction import CouplesExtraction
from .LabelExtractions import LabelPairs
import click

@click.command()
@click.argument('--wiki_file', type=click.File('rb'))
@click.argument('--db_file', type=click.File('wb'))
@click.option(
    '--number_samples',
    type=click.INT,
    default = 200000,
    help='Number of articles to keep from the dump.'
         'If not provided, it defaults at 200,000.'
)
@click.option('--last_launched',
              type=click.Choice(['none','parse_into_sentences','extraction_of_entities','linking_of_entities','couples_of_entities','labelling'],
              case_sensitive=False)
)
def importFromWikipediaDump(wiki_file, db_file, number_samples, last_launched):
    #booleans for each 'first_time'
    first_time_sent, first_time_ent, first_time_link, first_time_couples, first_time_label = True, True, True, True, True 
    if last_launched == 'parse_into_sentences':
        first_time_sent = False
    elif last_launched =='extraction_of_entities':
        first_time_sent = False
        first_time_ent = False
    elif last_launched == 'linking_of_entities':
        first_time_sent = False
        first_time_ent = False
        first_time_link = False
    elif last_launched == 'couples_of_entities':
        first_time_sent = False
        first_time_ent = False
        first_time_link = False
        first_time_couples = False
    elif last_launched == 'labelling':
        first_time_sent = False
        first_time_ent = False
        first_time_link = False
        first_time_couples = False
        first_time_label = False

    print("Starting extraction of articles of Wikipedia Dump into a labelled database of entities couples in sentences")
    WikiTransformer(wiki_file, workers=6, chunk_size=1000, number_articles_random=number_samples, random_seed=17).stream_articles(database_path=db_file, table="articles")
    print("Import of articles done...")

    DocumentSentences(db_file, first_time = first_time_sent).execute() 
    print("Separation into sentences done...")

    EntitiesBase(db_file, first_time=first_time_ent).execute() 
    print("Extraction of entities done...")

    LinkWikiPediaData(db_file, first_time=first_time_link).execute() 
    print("Linking of entities to Wikidata done...")

    CouplesExtraction(db_file, first_time=first_time_couples).execute() 
    print("Extraction of relations from Wikidata done...")

    LabelPairs(db_file, first_time=first_time_label).execute() 
    print("Labelling of samples done.")

if __name__ == "__main__":
    importFromWikipediaDump()

    #TODO : possibility for first_time = false ?
    #TODO : entity extractor that works..., our is protected.



