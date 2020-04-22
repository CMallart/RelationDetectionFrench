#!/usr/bin/env python3

import re
import tqdm
import sqlite3
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from ShortestPathDepParse import Dependencies
from EntitiesExtraction import EntityExtractor, Entity
from nltk import CoreNLPParser
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

#def pairwise(iterable):
 #   "s -> (s0,s1), (s1,s2), (s2, s3), ..."
  #  a, b = itertools.tee(iterable)
   # next(b, None)
    #return zip(a, b)
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

    def __init__(self, ent1, ent2, text, insecables):
        self.ent1=ent1
        self.ent2=ent2
        self.text = text
        self.ent1text = None
        self.ent2text = None
        self.ent1type= None
        self.ent2type=None
        self.dependency_graph = Dependencies(self.text, port = 9004, insecables=insecables)

        self.words =None
        self.words_before=None
        self.words_after=None
        self.shortest_dependency_path_w = None
        self.shortest_dependency_path_p = None
        self.shortest_dependency_path_t = None
        self.num_verbs = None #number of verbs between two entities
        self.num_punct = None #number of punctuation signs between entities (we assume a lot will say it's a dirent sentence)
        self.num_sconj = None #number of subornative conjunction(qui, que, qu'tel que...)

        self.N = 20

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

    def _get_type_entities(self):
        self.ent1type = self.ent1.label
        self.ent2type = self.ent2.label
    
    #@profile
    def _get_shortest_dependency_path(self):
        self.shortest_dependency_path_w, self.shortest_dependency_path_p, self.shortest_dependency_path_t = self.dependency_graph.shortest_path(self.ent1text, self.ent2text, 0, 0)
    
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
        self._get_type_entities()
        self._get_shortest_dependency_path()
        self._get_id_entities()
        pairfeatures = [self.id1, self.id2, self.ent1text, self.ent2text, self.ent1type, self.ent2type, 
                        self.shortest_dependency_path_p, self.shortest_dependency_path_w, self.shortest_dependency_path_t, self.text]
        return(pairfeatures)


    #@profile
    def clean_text(self, text):
        """
        With the context given as list of strings (list of words), we return a list of stemmed words, removing stop words.
        """
        text_list = [item.lower() for item in text if (item.lower() not in fr_stopwords)]
        text_clean = [item for item in text_list if ((len(item)>1)&(re.search(digits, item)==None))] #re.search(punctuation, item)==None)&
        return(text_clean)


class FeaturesCreationSQL():

    def __init__(self,  dbfile= "/home/cyrielle/Codes/clean_code/DataStorage/small_wiki/small_wiki.db", first_time=True):
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
        indexes_0 = list(np.random.choice(np.array(indexes_0_initial), 200000, replace=False)) #ratio*len(indexes_0_initial)
    
        #all the indics we finall take in final :
        
        indexes = indexes_0 + indexes_1
        print(len(indexes_0), len(indexes_1))
        return(indexes)
       



    def execute_SQL(self, ratio, filepath):
        """
        Given a list of pairs of entities, get the features of said pair
        """
        
        
        id_pairs = self.data_sample_chose(ratio = ratio)

        total_list =[]

        conn = sqlite3.connect(self.db, isolation_level=None)
        cursor = conn.cursor()
        query = "SELECT entities_pairs.entity1_id, entities_pairs.entity2_id, ent1.start, ent1.end, ent1.entity_type, ent1.wikidata_id,\
                                ent2.start, ent2.end, ent2.entity_type, ent2.wikidata_id, sentences.text, sentences.rowid, entities_pairs.relation_id, entities_pairs.rowid\
                                FROM entities_pairs \
                                LEFT JOIN entities as ent1 ON entities_pairs.entity1_id==ent1.rowid \
                                LEFT JOIN entities as ent2 ON entities_pairs.entity2_id==ent2.rowid \
								LEFT JOIN sentences ON sentences.rowid= entities_pairs.id_sentence\
                                WHERE entities_pairs.rowid in ({})".format(','.join(str(ind) for ind in id_pairs))
        cursor.execute(query)

        cursor2 = conn.cursor()
        query2 ="SELECT entities.surface_form, entities.id_sentence FROM entities"
        cursor2.execute(query2)
        dic_fetch = 0
        dic_entities_sentences = {}
        while True:
            dic_fetch = cursor2.fetchone()
            if dic_fetch == None:
                break
            entity_surface_form, id_sent = dic_fetch
            if id_sent in dic_entities_sentences :
                dic_entities_sentences[id_sent].append(entity_surface_form)
            else :
                dic_entities_sentences[id_sent] = [entity_surface_form]


        while True:
            pair = cursor.fetchone()
            if pair == None:
                break
            entid1, entid2, start1, end1, type1, wikidata_id1, start2, end2, type2, wikidata_id2, sentence, id_sentence, relation, id_pair = pair
            #quick fix for weirdo sentences :
            if "├" in sentence :
                break
            #elif len(list(re.finditer(',', sentence)))>7 : #very likely an enumeration, would probs no help
             #   break
            if start1 < start2:
                e1 = Entity(entid1, int(end1), int(start1), sentence, type1, wikidata_id1)
                e2 = Entity(entid2, int(end2), int(start2), sentence, type2, wikidata_id2)
            else :
                e2 = Entity(entid1, int(end1), int(start1), sentence, type1, wikidata_id1)
                e1 = Entity(entid2, int(end2), int(start2), sentence, type2, wikidata_id2)

            #insecables are the entities that are made of several words (Saint-Exupery, Francois Hollande...) but that need to be seen as one node in the dependency graph
            insecables = dic_entities_sentences[id_sentence]

            feature_pair = PairOfEntitiesFeatures(e1, e2, sentence, insecables).get_features()
            feature_pair.append(id_pair)
            feature_pair.append(relation)
            total_list.append(feature_pair)
        
        feats = pd.DataFrame(total_list, columns= ["id1", "id2", 'entity1', 'entity2', 'ent1type', 'ent2type', 
                                        'shortest_dependency_path_p', "shortest_dependency_path_w","shortest_dependency_path_t",
                                         "original_sentence","id_entities_pair",
                                         "relation"])

        entity_types = ["", "association",
                        "commune",
                        "date",
                        "departement",
                        "epci",
                        "institution",
                        "lieu",
                        "ong",
                        "organisation",
                        "partipolitique",
                        "pays",
                        "personne",
                        "region",
                        "societe",
                        "syndicat"]

        feats_0 = feats[pd.isna(feats["relation"])]
        feats_1 = feats[pd.notna(feats["relation"])]

        #Get the max number of any combinations 
        print(feats_1.groupby(["ent1type", "ent2type"]).count())
        lim1 = int(feats_1.groupby(["ent1type", "ent2type"]).count().quantile(0.95)[0])
        print("Quartile 0.95 of the distribution of number of examples per couples of entities, label 1: ", lim1)
        lim0 = lim1
        
        #Now the selection of the types of entities
        df = pd.DataFrame()
        for (type1, type2) in itertools.combinations(entity_types, 2):
            tempdf0 = feats_0[(feats_0['ent1type'] == type1)&(feats_0['ent2type'] == type2)]
            tempdf0 = tempdf0.iloc[0:lim0]
            tempdf1 = feats_1[(feats_1['ent1type'] == type1)&(feats_1['ent2type'] == type2)]
            tempdf1 = tempdf1.iloc[0:lim1]
            df= pd.concat([df, tempdf0, tempdf1])

        df.to_csv(filepath, index=False)
        return feats


####################################################################################################################################################################

####################################################################################################################################################################

if __name__ == "__main__":
    FeaturesCreationSQL(dbfile= "/home/cyrielle/Codes/clean_code/DataStorage/small_wiki/small_wiki.db", first_time=True).execute_SQL(ratio=1, 
        filepath="/home/cyrielle/Codes/clean_code/Features/data/features_ours.csv")




