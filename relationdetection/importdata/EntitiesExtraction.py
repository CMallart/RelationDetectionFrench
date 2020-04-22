#!/usr/bin/env python

import re
import tqdm
import pandas as pd
import requests
import json
import spacy

punctuation = (r'[\.,;:\?!()\[\]\{\}«»\'\"\-\—\/’&]')
digits = "([0-9])"
stopwords = ["alors","au","aucuns","aussi","autre","avec","car","ce","cet","cela","ces","ceux","ci","comme","comment",
        "dans","des","du","donc","elle","elles","en","est","et","eu","un", "une", "par", "plus", "moins", "aux",
        "ici","il","ils","je","juste","la","le","les","leur","là","ma","mais","mes","mine","moins","mon","mot",
        "ni","notre","nous","ou","où","parce","pas","peut","pour","quand","que","quel","quelle","quelles","on", "ont", "ne", "qu", "vers", "été",
        "était", "être", "avant", "après", "jusque","jusqu","depuis", "avoir",
        "quels","qui","sa","sans","ses","si","sien","son","sont","sous","sur","ta","tandis","tellement","tels","tes","ton","tous",
        "tout","trop","très","tu","votre","vous","vu","ça","sa", "son", "ses", "de", "a"]

class Entity():
    """
    Class to model an entity
    Mandatory args: 
        a start :int  (number of chars in the sentence)
        an end : int 
        text : the sentence from which we extract
    Adds :
        surface_form : the form taken by the entity
        tokens = # of words that compose this entity (ex, Cotes-d'Armor = 3 or Antoine de Saint-Exupery=4)

    """
    def __init__(self, id, end, start, text, label=None, wikidata_id=None):
        self.id = id
        self.label=label
        self.start= start
        self.surface_form = text[self.start:end]

        if ((self.surface_form[-1] in punctuation)&(re.search(r"([A-Z]\.)([A-Z]\.)+", self.surface_form)==None)&(re.search(r"([0-9]+\.)[0-9]+", self.surface_form)==None)):
            self.end = end-1
            self.surface_form = text[self.start:self.end].rstrip(".")
        else:
            self.end = end
            self.surface_form = text[self.start:self.end]
        self.tokens = re.split(r"[\'\- \/.]", self.surface_form)
        self.wikidata_id = wikidata_id if wikidata_id != "" else None

    def print_ents(self):
        return(self.surface_form + "---"+ "/".join(self.tokens) + " ---" + str(self.start) + " \\ "+ str(self.end))



class EntityExtractor():
    """
    returns the details of the entities extracted from the text
    under the form of lists, df or str(to be input in a db)
    """

    def __init__(self, extraheaders = None, model = 'BDC_2019-05-20_16-36-02'):
        self.headers = {'Content-Type':'application/json', 'accept': 'application/json'}
        self.url = 'http://d1bdcreco01.ouest-france.fr:5000/modele/%s/annotation' % model #d1bdcreco01.ouest-france.fr car pb de resolution dns quand utilisation du vpn


    def callapi(self, text):
        response = requests.post(self.url, headers = self.headers, data = text)
        doc = json.loads(response.content)

        entitiesinfo=[]
        if 'annotations' in doc:
            for annotation in doc['annotations']:
                temp = []
                temp.extend([str(annotation["start"]), str(annotation["end"]), annotation["entiteForme"]])

                if ("wikiId" in annotation) & ("scoreWiki" in annotation) :
                    if (annotation['scoreWiki']> 0.6):
                        temp.append(annotation["wikiId"])
                    else:
                        temp.append('')
                else :
                    temp.append('')

                if 'entiteType' in annotation:
                    temp.append(annotation['entiteType'])
                else :
                    temp.append('')

                entitiesinfo.append(temp)

        else:
            entitiesinfo = []

        dataframe = pd.DataFrame(entitiesinfo, columns = ['start', 'end', 'surface_form', 'wiki_id', 'entity_type'])
        return(dataframe)

    def get_entities_spacy(text):


    def get_entities(self, text):
        text =  text.encode("utf-8")
        dataframe = self.callapi(text)
        returnlist = [dataframe.columns.values.tolist()] + dataframe.values.tolist()
        return(returnlist)

    def get_str_list_entities(self, text):
        text =  text.encode("utf-8")
        dataframe = self.callapi(text)
        returnlist = dataframe.values.tolist()
        returnlist = ["__".join(l) for l in returnlist]
        returnlist = "\\".join(returnlist)
        return(returnlist)

    def get_df_entities(self, text):
        text =  text.encode("utf-8")
        dataframe = self.callapi(text)
        return(dataframe)
    
    def get_forms_entities(self, text):
        text =  text.encode("utf-8")
        dataframe = self.callapi(text)
        list_surface_forms = dataframe.surface_form.tolist()
        return(list_surface_forms)

if __name__ == "__main__":
    ents= EntityExtractor()
    print(ents.get_df_entities(text="Dans le Larousse, à la lettre R comme « rumeur », on lit : « Nouvelle, bruit qui se répand dans le public, dont l'origine est inconnue ou incertaine et la véracité douteuse. » Appliqué à la Tunisie, on peut dire que, en une semaine, l'actualité a connu plusieurs épisodes salés, dignes d'un mauvais feuilleton, genre télévisuel très prisé pendant la période ramadanesque. Inventaire."))