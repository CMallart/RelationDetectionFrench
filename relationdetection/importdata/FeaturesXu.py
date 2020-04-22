#!/usr/bin/env python

import re
import Stemmer
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from ShortestPathDepParse import Dependencies
from EntitiesExtraction import EntityExtractor, Entity
from nltk import CoreNLPParser
from LCA_depparse import DependenciesLCA
import numpy as np
np.random.seed(17)

punctuation = r'[\.,;:\?!()\[\]\{\}«»\'\"\—’&\+]' #\- for now
digits = "([0-9])"
fr_stopwords = ["alors","au","aucuns","aussi","autre","avec","car","ce","cet","cela","ces","ceux","ci","comme","comment",
        "dans","des","du","donc","elle","elles","en","est","et","eu","un", "une", "par", "plus", "moins", "aux",
        "ici","il","ils","je","juste","la","le","les","leur","là","ma","mais","mes","mine","moins","mon","mot",
        "ni","notre","nous","ou","où","parce","pas","peut","pour","quand","que","quel","quelle","quelles","on", "ont", "ne", "qu", "vers", "été",
        "était", "être", "avant", "après", "jusque","jusqu","depuis", "avoir", 
        "quels","qui","sa","sans","ses","si","sien","son","sont","sous","sur","ta","tandis","tellement","tels","tes","ton","tous",
        "tout","trop","très","tu","votre","vous","vu","ça","sa", "son", "ses", "de", "a"]

en_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
                "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
                "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
                "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
                "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", 
                "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", 
                "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
                "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", 
                "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
                "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "even"]

pos_tagger = CoreNLPParser('http://localhost:9004', tagtype='pos')


class PairOfEntitiesFeatures():
    """
    For a given pair of entities in a sentence, find the features between them
    Features for now include :
            - surface form entity 1
            - surface form entity 2
            - type entity 1 (PER, ORG, LOC...)
            - type entity 2 (PER, ORG, LOC...)
            - words between entites
            - x words before the entity 1
            - x words after entity 2
            - shortest dependency path between two entities

    Only function to use : get_features()
    """

    def __init__(self, ent1, ent2, text):
        self.ent1=ent1
        self.ent2=ent2
        self.text = text
        self.ent1text = None
        self.ent2text = None
        self.ent1type= None
        self.ent2type=None

        self.N = 20
        self.stemmer = Stemmer.Stemmer('french')

    def _get_id_entities(self):
        """
        Keep the id of the entities,
        So that later we can go find their properties in the 'entities' table
        """
        self.id1 = self.ent1.id
        self.id2 = self.ent2.id

    def _get_entities_surface_form(self):
        self.ent1text = self.ent1.surface_form.replace(" "," ") #re.sub(r'[ ]+', " ", re.sub(punctuation, " ", self.ent1.text))
        self.ent2text = self.ent2.surface_form.replace(" "," ") #re.sub(r'[ ]+', " ", re.sub(punctuation, " ", self.ent2.text))

    def _get_dependency_paths(self):
        self.word_path1, self.word_path2, self.rel_path1, self.rel_path2, self.pos_path1, self.pos_path2 = DependenciesLCA(self.text).branch_paths(self.ent1text, self.ent2text )
    
    #@profile
    def get_features(self):
        """
        Outputs the PairOfEntities object attributes as a list

        List of :
            - surface form entity 1
            - surface form entity 2
            - type entity 1 (PER, ORG, LOC...)
            - type entity 2 (PER, ORG, LOC...)
            - words between entites
            - x words before the entity 1
            - x words after entity 2
            - shortest dependency path between two entities

        """
        self._get_entities_surface_form()
        self._get_dependency_paths()
        self._get_id_entities()
        pairfeatures = [self.id1, self.id2, self.ent1text, self.ent2text, self.word_path1, self.word_path2, self.rel_path1, self.rel_path2, self.pos_path1, self.pos_path2, self.text]
        return(pairfeatures)

class FeaturesCreationSQL():

    def __init__(self,  dbfile= "/home/cyrielle/Codes/clean_code/DataStorage/small_wiki/small_wiki.db", first_time=True ):
        self.db = dbfile
        self.first_time = first_time

    def data_sample_chose(self, ratio=1):
        """
        Count how many ones we have in the entities pairs and where
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        cursor.execute("SELECT rowid FROM entities_pairs \
                    WHERE relation_id is not null\
                    AND entities_pairs.entity1_wikidata_id is not null\
                    AND entities_pairs.entity2_wikidata_id is not null")
        indexes_1 = [c[0] for c in cursor.fetchall()]

        cursor.execute("SELECT rowid FROM entities_pairs \
                    WHERE relation_id is null\
                    AND entities_pairs.entity1_wikidata_id is not null\
                    AND entities_pairs.entity2_wikidata_id is not null")
        indexes_0_initial = [c[0] for c in cursor.fetchall()]
        indexes_0 = list(np.random.choice(np.array(indexes_0_initial), 200000, replace=False))
    
        #all the indics we finall take in final :
        
        indexes = indexes_0 + indexes_1
        print(len(indexes_0), len(indexes_1))
        return(indexes)
       

    def execute_SQL(self, ratio, filepath, same_as=None):
        """
        Given a list of pairs of entities, get the features of said pair
        """
        if same_as is None :
            id_pairs = self.data_sample_chose(ratio = ratio)

            conn = sqlite3.connect(self.db, isolation_level=None)
            cursor = conn.cursor()
            query = "SELECT entities_pairs.entity1_id, entities_pairs.entity2_id, ent1.start, ent1.end, ent1.entity_type, ent1.wikidata_id,\
                                    ent2.start, ent2.end, ent2.entity_type, ent2.wikidata_id, sentences.text, sentences.rowid, entities_pairs.relation_id\
                                    FROM entities_pairs \
                                    LEFT JOIN entities as ent1 ON entities_pairs.entity1_id==ent1.rowid \
                                    LEFT JOIN entities as ent2 ON entities_pairs.entity2_id==ent2.rowid \
                                    LEFT JOIN sentences ON sentences.rowid= entities_pairs.id_sentence\
                                    WHERE entities_pairs.rowid in ({})".format(','.join(str(ind) for ind in id_pairs))
            cursor.execute(query)

        else :
            df_to_copy = pd.read_csv(same_as)
            id_pairs = df_to_copy['id_entities_pair'].tolist()
            
            cursor = conn.cursor()
            query = "SELECT entities_pairs.entity1_id, entities_pairs.entity2_id, ent1.start, ent1.end, ent1.entity_type, ent1.wikidata_id,\
                                    ent2.start, ent2.end, ent2.entity_type, ent2.wikidata_id, sentences.text, sentences.rowid, entities_pairs.relation_id\
                                    FROM entities_pairs \
                                    LEFT JOIN entities as ent1 ON entities_pairs.entity1_id==ent1.rowid \
                                    LEFT JOIN entities as ent2 ON entities_pairs.entity2_id==ent2.rowid \
                                    LEFT JOIN sentences ON sentences.rowid= entities_pairs.id_sentence\
                                    WHERE entities_pairs.rowid in ({})".format(','.join(str(ind) for ind in id_pairs))
            cursor.execute(query)

        total_list=[]

        while True:
            pair = cursor.fetchone()
            if pair == None:
                break
            entid1, entid2, start1, end1, type1, wikidata_id1, start2, end2, type2, wikidata_id2, sentence, id_sentence, relation = pair

            if start1 < start2:
                e1 = Entity(entid1, int(end1), int(start1), sentence, type1, wikidata_id1)
                e2 = Entity(entid2, int(end2), int(start2), sentence, type2, wikidata_id2)
            else :
                e2 = Entity(entid1, int(end1), int(start1), sentence, type1, wikidata_id1)
                e1 = Entity(entid2, int(end2), int(start2), sentence, type2, wikidata_id2)

            feature_pair = PairOfEntitiesFeatures(e1, e2, sentence).get_features()
            are_related = 1 if (relation is not None) else 0
            feature_pair.append(int(are_related))
            total_list.append(feature_pair)
        
        feats = pd.DataFrame(total_list, columns= ["id1", "id2", 'entity1', 'entity2', 
                                        "word_path1", "word_path2", "dep_path1", "dep_path2", "pos_path1", "pos_path2",
                                        "original_sentence",
                                        "relation"])

        feats.to_csv(filepath, index=False)
        return feats


####################################################################################################################################################################

####################################################################################################################################################################

if __name__ == "__main__":
    FeaturesCreationSQL(dbfile= "/home/cyrielle/Codes/clean_code/DataStorage/small_wiki/small_wiki.db", 
                        first_time=True).execute_SQL(ratio=1, filepath = "/home/cyrielle/Codes/clean_code/Models/Features/features_Xu.csv", 
                        same_as="/home/cyrielle/Codes/clean_code/Features/data/features_ours.csv")




