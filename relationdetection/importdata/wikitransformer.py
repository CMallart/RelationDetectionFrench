#####################################################
#Use this script only for the links between articles, 
# WikiExtractor cleans the text more thouroughly
#####################################################

import json
import pandas as pd
import sqlite3
from tqdm import tqdm
import re
import numpy as np
from wikipageextractor import WikiExtractor


__doc__ = """
Extraction de liens entre article wiki et lien wiki (e.g. la page Emmanuel_Macron possède un lien vers la page Edouard_Phillipe).
On conserve la forme associée au lien extrait dans la page.
Les pages on été au préalable analysées avec le script wikipediaextractor.py (forké et améliorant https://github.com/sjebbara/Wikipedia2Json).
Les résultats sont stockés dans une base SQLite pour permettre des agrégations et jointures avec d'autres jeux de données.
"""


class WikiTransformer:

    wiki_prefix = "http://fr.wikipedia.org/wiki/"

    def __init__(self, wiki_path, workers=8, chunk_size=1000, number_articles_random=None, random_seed=17):
        self.wiki_path = wiki_path
        self.workers = workers
        self.chunk_size = chunk_size
        np.random.seed(random_seed)
        if number_articles_random is not None:
            self.indices_chosen = np.random.choice(np.array(range(0, 2133000)), number_articles_random, replace=False)
        else:
            self.indices_chosen = None

    def generate_documents(self):
        wiki_extractor = WikiExtractor(prefix=self.wiki_prefix)
        return wiki_extractor.generate_json_docs(self.wiki_path, number_of_workers=self.workers, chunk_size=self.chunk_size)

    def generate_relations(self):
        with tqdm() as pbar:
            for docs in self.generate_documents():
                relations = []
                for item in docs:
                    if item:
                        #(url, doc) = item
                        doc=item
                        node_source = doc.title
                        url = doc.url
                        for a in doc.annotations:
                            node_target = a['uri']
                            surface_form = a['surface_form']
                            relations.append([url, node_source, node_target, surface_form])
                        pbar.update(1)
                if relations:
                    yield relations

    def stream_articles(self, database_path, table="articles"):
        cols = ['url', 'id', 'title', 'text']
        self.conn = sqlite3.connect(database_path,isolation_level=None)
        if self.indices_chosen is not None:
            with tqdm() as pbar:
                articles=[]
                ind = -1
                for doc in self.generate_documents():
                    if doc is not None:
                        ind+=1
                        if ind > max(self.indices_chosen):
                            df_art = pd.DataFrame(articles, columns=cols)
                            df_art.to_sql(name=table, con=self.conn, chunksize=200,
                                    if_exists='append', index=False)
                            break
                        else:
                            id = doc.id
                            title=doc.title
                            text = doc.text
                            url = doc.url
                            if ind in self.indices_chosen:
                                articles.append(
                                    [url, id, title, self.deep_clean_article(text)])
                            pbar.update(1)
                            if (len(articles) > 4000):
                                df_art = pd.DataFrame(articles, columns=cols)
                                df_art.to_sql(name=table, con=self.conn, chunksize=200,
                                        if_exists='append', index=False)
                                articles = []
        else:
            with tqdm() as pbar:
                articles=[]
                for doc in self.generate_documents():
                    if doc is not None:
                        id = doc.id
                        title=doc.title
                        text = doc.text
                        url = doc.url
                        articles.append(
                            [url, id, title, self.deep_clean_article(text)])
                        pbar.update(1)
                    if (len(articles) > 4000):
                        df_art = pd.DataFrame(articles, columns=cols)
                        df_art.to_sql(name=table, con=self.conn, chunksize=200,
                                if_exists='append', index=False)
                        articles = []
    

    def deep_clean_article(self, text):
        text = text.replace(u"\u2019", u"\u0027").replace("\n"," ").replace(u"\u2026", ".")
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = re.sub(r"\">([A-Za-záàâäãåçéèêëíìîïñóòôöõúùûüýÿæœ ])<\/a>\)", "\\1", text)
        text = re.sub(r"\">[A-Za-záàâäãåçéèêëíìîïñóòôöõúùûüýÿæœ ]<\/a>\: \'\)", '', text)
        text = re.sub(r"<\/[A-Za-z]+>", '', text)
        return(text) 


    def stream_to_table(self, table='links'):
        cols = ["url_source", "source", "target", "target_surface_form"]#, "offset_taget_in_source"]
        for relations in self.generate_relations():
            df_rels = pd.DataFrame(relations, columns=cols)
            df_rels.to_sql(name=table, con=self.conn, chunksize=200,
                           if_exists='append', index=False)



if __name__ == '__main__':
    w = WikiTransformer("/home/cyrielle/Codes/clean_code/DataStorage/wiki/frwiki-20200120-pages-articles-multistream.xml.bz2", workers=6, random_seed=17)
    w.stream_articles(database_path='/home/cyrielle/Codes/clean_code/DataStorage/wiki/wiki.db')

