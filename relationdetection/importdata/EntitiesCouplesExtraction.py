import sqlite3
import pandas as pd
import itertools
from EntitiesExtraction import EntityExtractor, Entity



class FindCouplesSentence():

    def __init__(self, entities):
        self.entities = entities

    def combinate(self,list_entities):
        """
        To extract all the possible combinations of two entities identified in a sentence,
        But only if they have at most two entities between them, 
        Other than that the link is too weak to think it can be a relation
        """

        def pairwise(iterable):
        #"s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        def threewise(iterable):
        #"s -> (s0,s2), (s1,s3), (s2, s4), ..."
            a, b = itertools.tee(iterable)
            next(b, None)
            next(b, None)
            return zip(a, b)

        def fourwise(iterable):
        #"s -> (s0,s2), (s1,s3), (s2, s4), ..."
            a, b = itertools.tee(iterable)
            next(b, None)
            next(b, None)
            next(b, None)
            return zip(a, b)

        pairs = list(pairwise(list_entities))
        threes = list(threewise(list_entities))
        fours = list(fourwise(list_entities))
        return(pairs+threes+fours)

    def order_entities_by_sentence_order(self):
        self.entities.sort(key=lambda ent: ent.start, reverse=True)

    def df_couple_entities(self):
        """
        Once the entities are ordered in terms of apparition in the sentence,
        We keep only the couples of entities that are separated by at most 2 other entities
        And whose wikidata id exists
        """
        self.order_entities_by_sentence_order()
        entities_pairs=[(ent1.id, ent1.surface_form, ent1.wikidata_id, ent2.id, ent2.surface_form, ent2.wikidata_id) for (ent1,ent2) in self.combinate(self.entities) if (ent1.wikidata_id != "")&(ent2.wikidata_id!="")]
        pairs = pd.DataFrame(entities_pairs, columns=['entity1_id', 'entity1_surface_form', 'entity1_wikidata_id', 'entity2_id', 'entity2_surface_form', 'entity2_wikidata_id'])
        return(pairs)


class CouplesExtraction():

    def __init__(self,  dbfile= "/home/cyrielle/Codes/clean_code/DataStorage/small_wiki.db", first_time=True ):
        self.db = dbfile
        self.first_time = first_time

        
    def entities_from_sql(self, fetchall_request, text):
        """
        Once we have the lines from the database, 
        create an "Entity" object from those fields
        """
        entities =[]
        for tup in fetchall_request :
            rowid, start, end, label, wikidata_id = tup
            entities.append(Entity(rowid, int(end), int(start), text, label, wikidata_id))
        return entities


    def execute(self):
        """
        Prendre les phrases, les entities extraites,
        et une base des faire les couples d'entites qui ont
            -au plus 2 entites entre elles (sinon les liens sont bien trop tenus)
            -chacune un wikidata id 
        """

        conn = sqlite3.connect(self.db, isolation_level=None)
        cursor = conn.cursor()
        cursor.execute('pragma journal_mode=wal')
        
        if self.first_time :
            query = "SELECT rowid, text FROM sentences"
        else:
            cursor.execute("select distinct id_sentence from entities_pairs")
            list_id = [el[0] for el in list(cursor.fetchall())]
            query = "SELECT rowid, text FROM sentences WHERE rowid not in ({})".format(','.join(str(ind) for ind in list_id))

        print("Started")
        i=0
        cksize = 1000
        for df in pd.read_sql_query(query, conn, chunksize=cksize):
            i+=1
            for id_sent, sentence in df.itertuples(index=False):
                cursor.execute("SELECT rowid, start, end, entity_type, wikidata_id FROM entities WHERE id_sentence= ?", (id_sent,))
                entities = self.entities_from_sql(cursor.fetchall(), sentence)

                ent_couples = FindCouplesSentence(entities).df_couple_entities()
                ent_couples['id_sentence'] = id_sent

                ent_couples.to_sql('entities_pairs', conn, if_exists='append', index=False,chunksize=cksize)
            print("Found the couples of ", i*cksize, " sentences analysed")

if __name__ == "__main__":
    CouplesExtraction(first_time=False).execute()