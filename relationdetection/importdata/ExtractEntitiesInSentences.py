import sqlite3
import pandas as pd
from EntitiesExtraction import EntityExtractor
import numpy as np
np.random.seed(17)

class EntitiesBase:

    def __init__(self, dbfile="/home/cyrielle/Codes/clean_code/DataStorage/small_wiki.db",first_time=True):
        self.dbfile = dbfile
        self.first_time = first_time

    def execute(self):
        newconn = sqlite3.connect(self.dbfile, isolation_level=None)
        EntityExtract = EntityExtractor()

        if self.first_time:
            query = "SELECT rowid, id_article, text FROM sentences"
        else:
            query = "SELECT rowid, id_article, text FROM sentences WHERE id_article NOT IN (SELECT DISTINCT id_article FROM entities)"

        i = 0
        t = 0
        print("Starting")

        chksize = 2000
        # create dataframe placeholder
        totaldata = pd.DataFrame()
        for df in pd.read_sql_query(query, newconn, chunksize=chksize):
            # for each article text, get the entities under form of a dataframe
            for id_sent, id_article, sent in df.itertuples(index=False):
                i += 1
                t +=1
                dfentities = EntityExtract.get_df_entities(sent)
                dfentities["id_article"] = id_article
                dfentities["id_sentence"] = id_sent
                totaldata = pd.concat(
                    [totaldata, dfentities], axis=0, ignore_index=True
                )
                if i >= 100:
                    totaldata.to_sql(
                        "entities",
                        newconn,
                        if_exists="append",
                        index=False,
                    )
                    i = 0
                    totaldata = pd.DataFrame()
            
                if t%chksize ==0 : print("Found the entities in ", t, "sentences")


if __name__ == "__main__":
    EntitiesBase(first_time=False).execute()

