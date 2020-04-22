import spacy
from spacy import displacy
import pandas as pd
from spacy.tokens import Span
import re


punctuation = (r'[\.,;:\?!()\[\]\{\}«»\'\"\-\—\/’&]')

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
    def __init__(self, id, end, start, text, label=None):
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

    def print_ents(self):
        return(self.surface_form + "---"+ "/".join(self.tokens) + " ---" + str(self.start) + " \\ "+ str(self.end))



class EntityExtractor():
    """
    returns the details of the entities extracted from the text
    under the form of lists, df or str(to be input in a db)
    """

    def __init__(self):
        self.punct_not_in_french_entities = [";", ",", "\"", ":", "--", "...", ".", "+", "!", "?", "/"] #punctuation signs not usually associated with French entities
        self.nlp = spacy.load('fr_core_news_sm')

    def merge_entities(self, e1, e2, doc, keep_first_label = True):
        """
        Take two neighbouring entities and merge them into one span (almost an entity).
        
        e1 : first entity (nlp(text).ents[1])
        e2 : second entity
        keep_first_label : if we keep as label the one of the first entity, otherwise the second
        """
        if keep_first_label:
            #consider, from looking at French, that the label of the reunion of the entities separated by an ' 
            #is the label of the first one : this is why it's default
            new_label_ = e2.label_ #(unicode)
            #new_label = e2.label # get hash value of entity label (int)
        else :
            new_label_ = e1.label_ 
            #new_label = e1.label

        #create a Span with the start and end index of the token, not the start and end index of the entity in the document 
        #start and end are the token offset, while start_char and end_char are the character offset
        start_token = e1.start
        end_token = e2.end
        new_entity =  Span(doc, start_token, end_token, label_=new_label_)
        return new_entity


    def is_real_name(self, text, punct_not_in_french_entities):
        """
        Check if a text contains the punctuation we defined as 'not pertaining to an entity text'
        """
        #detect the entities that have a weird punctuation in them
        #the only punctuation sign accepted is, in the end, the apostrophe and the hyphen
        
        #barbaric
        is_ok = True
        for punct in punct_not_in_french_entities:
            if punct+" " in text:
                is_ok = is_ok & False
            else: is_ok = is_ok & True
        return is_ok
        ##TODO : make that better, it's too brute with that for loop


    def probably_split_apostrophe_hyphen(self, entity, next_entity, texte):
        """
        Checks if two consecutive entities are not actually one single entity split in 2
        """
        split = False
        if (entity.text.endswith("'")):
            split = True
        if (texte[next_entity.start_char-1] == "'")or(texte[next_entity.start_char-1] == "-"):
            split=True
        return(split)

    def extract_clean_entities(self,texte, punct_not_in_french_entities):
        """
        Take a single text and a dictionnary of 'forbidden' punctuation (punctuation that should not appear in an entity name)
        Returns a list of entities extracted from that text
        """
        doc = self.nlp(texte)
        extracted_entities = []
        ignore_next = False

        for num, entity in enumerate(doc.ents):
            if ignore_next : 
                ignore_next = False
                continue
            else :
                if entity.end_char - entity.start_char > 1 :#If the length of the entity is longer than 1 character (eliminate the -- abheration)
                    if self.is_real_name(entity.text, punct_not_in_french_entities) :#If the entity name looks like a real word (eliminate the ''[-- 2006] LOC' kind of case)
                        if num < len(doc.ents)-1 :
                            next_entity = doc.ents[num+1]
                            if self.probably_split_apostrophe_hyphen(entity, next_entity, texte) :# If there is a single apostrophe between the two entities, it is split wrongly
                                ignore_next = True
                                new_entity = self.merge_entities(entity, next_entity, doc, keep_first_label=True)
                                extracted_entities.append(new_entity)

                            else :
                                extracted_entities.append(entity)
                        else:
                            extracted_entities.append(entity)
        return(extracted_entities)

    def get_entity_spec(self, text):
        l = []
        entity_list = self.extract_clean_entities(text, self.punct_not_in_french_entities)
        for ent in entity_list:
            l.append((ent.text, ent.start_char, ent.end_char, ent.label_))
        dataframe = pd.DataFrame(l, columns = ['surface_form', 'start', 'end', 'entity_type'])
        return dataframe

if __name__ == "__main__":
    ents= EntityExtractor()
    print(ents.get_df_entities(text="Dans le Larousse, à la lettre R comme « rumeur », on lit : « Nouvelle, bruit qui se répand dans le public, dont l'origine est inconnue ou incertaine et la véracité douteuse. » Appliqué à la Tunisie, on peut dire que, en une semaine, l'actualité a connu plusieurs épisodes salés, dignes d'un mauvais feuilleton, genre télévisuel très prisé pendant la période ramadanesque. Inventaire."))