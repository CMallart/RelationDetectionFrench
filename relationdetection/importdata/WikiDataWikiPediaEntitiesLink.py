"""
This script takes the table of textual entities, checks their wikipedia id and writes in the same base their wikidata id
"""

import requests
import sqlite3
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import json
import sys

class LinkWikiPediaData:

    def __init__(self, dbfile = "/home/cyrielle/Codes/clean_code/DataStorage/small_wiki.db", first_time=True):
        self.dbfile = dbfile
        self.first_time = first_time

    def to_str(self, vals):
        s = ""
        for item in vals:
            wiki_art_name = item.replace("\"", "\\\"")
            wiki_art_name = wiki_art_name.replace("_", " ")
            s = s + "\"" + str(wiki_art_name) + "\"" +"@fr "
        return(s)


    def json_to_df(self, data):
        wikidata_ids = []
        for item in data['results']['bindings']:
            wikidata_ids.append(OrderedDict({
                'wikidata_id': item['item']['value']
                    if 'item' in item else None,
                'lemma': item['lemma']['value'].replace(" ", "_")
                }
                ))
        df = pd.DataFrame(wikidata_ids)
        return(df)


    def json_to_dict(self, data):
        wikidata_ids = []
        for item in data['results']['bindings']:
            wikidata_ids.append(OrderedDict({
                'wikidata_id': item['item']['value']
                    if 'item' in item else None,
                'lemma': item['lemma']['value'].replace(" ", "_")
                }
                ))
        return(wikidata_ids)


    def json_to_tuples(self, data):
        wikidata_ids = []
        for item in data['results']['bindings']:
            wikidata_ids.append((item['item']['value'] if 'item' in item else None, item['lemma']['value'].replace(" ", "_")) )
        return(wikidata_ids)


    def query_list_of_articles(self, l, url ='https://query.wikidata.org/sparql'):
        headers = {
            'User-Agent': 'EntityLinkingBot/0.1 (cyrielle.mallart@irisa.fr) python-requests/entity_linking_using_wikidata_knowledge_base/wikipedia_page_to_wikidata/query_list_articles/0.5'
        }
        list_of_values = l
        url = 'https://query.wikidata.org/sparql'
        query = "SELECT ?lemma ?item WHERE {VALUES ?lemma {%s} ?sitelink schema:about ?item; schema:isPartOf <https://fr.wikipedia.org/>; schema:name ?lemma.}" %(self.to_str(list_of_values))
        r = requests.get(url, params = {'format': 'json', 'query': query}, headers = headers)
        try :
            data = r.json()
            df = self.json_to_tuples(data)
            return(df)
        except:
            print(r.content)
        
    #@profile
    def execute(self):
        conn = sqlite3.connect(self.dbfile, isolation_level=None)
        cursor = conn.cursor()
        t=0
        print("Starting")
        cursor.execute('pragma journal_mode=wal')
        if self.first_time:
            cursor.execute("ALTER TABLE entities ADD COLUMN wikidata_id")
    
        ck= 200
        #get the entities by batches of 100 max not to crash wikidata query api
        for df in pd.read_sql_query("SELECT wiki_id FROM entities WHERE wiki_id != '' AND wikidata_id IS null", conn, chunksize=ck):
            links_id = df["wiki_id"].values.tolist()
            unique_link_ids = set(list(links_id))
            links = self.query_list_of_articles(unique_link_ids)
            #print(links)

            t+=ck
            print("Tried to find wikidata id for", t, "entities")

            cursor.executemany("UPDATE entities SET wikidata_id = ? WHERE wiki_id =?", links)



if __name__ == "__main__":
    LinkWikiPediaData(first_time=False).execute() 
    
"""
    test_list = ['Tueur_de_masse', 'Anders_Behring_Breivik', 'Gus_Van_Sant', "Attentats_d'Oslo_et_d'Utøya", 'Compte_de_résultat', 'Flux_de_trésorerie', 'Compte_de_résultat', 'Résultat_exceptionnel', 'Claude-Nicolas_Ledoux', 'Urbaniste', 'Utopie', "Société_d'Ancien_Régime", 'Architecture_néo-classique_en_Belgique', '21_mars', '1736', 'Dormans', 'Sassenage', 'Gravure', 'Jacques-François_Blondel', 'Antiquité', 'Jean-Michel_Chevotet', 'Place_Vendôme', 'Rococo', 'Louis-François_Trouard', '1757', 'Paestum', 'Andrea_Palladio', '1762', 'Rue_Saint-Honoré', '1969', 'Musée_Carnavalet', 'Mauperthuis', 'Aqueduc', 'Orangerie', '1764', "Rue_de_la_Chaussée-d'Antin", 'Andrea_Palladio', 'Ordre_colossal', 'Montfermeil', 'Claude-Nicolas_Ledoux', 'Fronton_(architecture)', "Claude-Louis_d'Aviler", '1764', '1770', 'Tonnerre_(Yonne)', 'Sens_(Yonne)', 'Saint-Étienne', 'Auxerre', 'Abbaye_de_Reigny', '1766']
    def to_str(vals):
        s = ""
        for item in vals:
            wiki_art_name = item.replace("\"", "\\\"")
            wiki_art_name = wiki_art_name.replace("_", " ")
            s = s + "\"" + str(wiki_art_name) + "\"" +"@fr "
        return(s)

    print(to_str(test_list))
    query = "SELECT ?lemma ?item WHERE {VALUES ?lemma {%s} ?sitelink schema:about ?item; schema:isPartOf <https://fr.wikipedia.org/>; schema:name ?lemma.}" %(to_str(test_list))
    r = requests.get('https://query.wikidata.org/sparql', params = {'format': 'json', 'query': query})
    data = r.json()
    print(r.content)
"""
    