
from tqdm import tqdm
import sqlite3
import re
import pandas as pd
import sys 

class DocumentSentences():

    def __init__(self, dbfile= "/home/cyrielle/Codes/clean_code/DataStorage/small_wiki.db", first_time = True):
        self.first_time = first_time
        self.dbfile = dbfile 

        self.alphabets= "([A-Za-z])"
        self.prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
        self.suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        self.starters = "(Mr|Mrs|Ms|Dr)"
        self.acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        self.websites = "[.](com|net|org|io|gov|me|edu)"
        self.digits = "([0-9])"
        self.punctuation = (r'[\.,;:\?!()\[\]\{\}«»\'\"\-\—\/’&\+\~]')
        
        #self.corenlpparser = CoreNLPParser(url='http://localhost:'+ str(corenlpserverport), tagtype='pos')

    #@profile
    def split_into_sentences(self, article):
        """
        Given a text presented as a succession of sentences, split it into a list of sentences.
        Returns the start index of the sentences.
        """
        
        text= article
        text = re.sub( self.prefixes,"\\1<prd>",text)
        text = re.sub( self.websites,"<prd>\\1",text)
        text = re.sub("Ph.D.","Ph<prd>D<prd>", text)
        #text = re.sub(r"\s" +  self.alphabets + "[.] "," \\1<prd> ",text)
        #text = re.sub( self.alphabets + "[.]" +  self.alphabets + "[.]" +  self.alphabets + "[.]"+  self.alphabets + "[.]"+  self.alphabets + "[.]"+  self.alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>\\4<prd>\\5<prd>\\6<prd>",text)
        text = re.sub( self.alphabets + "[.]" +  self.alphabets + "[.]" +  self.alphabets + "[.]"+  self.alphabets + "[.]"+  self.alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>\\4<prd>\\5<prd>",text)
        text = re.sub( self.alphabets + "[.]" +  self.alphabets + "[.]" +  self.alphabets + "[.]"+  self.alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>\\4<prd>",text)
        text = re.sub( self.alphabets + "[.]" +  self.alphabets + "[.]" +  self.alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub( self.alphabets + "[.]" +  self.alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+ self.suffixes+"[.]"," \\1<prd>",text)
        text = re.sub( self.digits + "[.]" +  self.digits,"\\1<prd>\\2",text)
        text = re.sub( self.digits + "[.]" +  self.digits,"\\1<prd>\\2",text)
        text = re.sub(r"(<prd>)(\s[A-Z])","\\1<stop>\\2", text) #If we hav an acronym as the last word of a sentence, such as "...entre chez P.C.L. Au cours de ..."
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("...", ".")
        
        
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        #sentences = sentences[:-1]
        #sentences = [s.strip(" ") for s in sentences if (len(s.split(" "))>6)] #&(self.has_verb(s))]
        return(sentences)


    def sentences_df(self, df_articles, sent_id):
        df_sents = pd.DataFrame()
        s_id = sent_id
        for id_article, text in df_articles.itertuples(index=False):
            sentences = self.split_into_sentences(text)
            sent_infos =[]
            for sent in sentences:
                try:
                    start_sent = text.index(sent)
                    end_sent = start_sent+len(sent)
                    s_id+=1
                except :
                    print(sent +"--------", text, "-----------------")
                    start_sent = None
                    end_sent = None
                sent_infos.append((id_article, start_sent, end_sent, sent))
            df_sents_temp = pd.DataFrame(sent_infos, columns = ['sent_id','id_article', 'start_in_article', 'end_in_article', 'text','preceding_sent'])
            df_sents = pd.concat([df_sents, df_sents_temp])
        return(df_sents)   



    #@profile
    def execute(self):
        conn = sqlite3.connect(self.dbfile, isolation_level=None)
        cursor = conn.cursor()
        if self.first_time:
            query = 'SELECT id_version, text FROM articles'
        else:
            cursor.execute("select distinct id_article from sentences")
            list_id = ['"'+el[0]+'"' for el in list(cursor.fetchall())]
            query = 'SELECT id_version, text FROM articles WHERE id_version not in ({})'.format(' , '.join(str(ind) for ind in list_id))

        cksize = 500
        i=0
        print("Starting")
        sent_id = -1
        for df in pd.read_sql_query(query, conn, chunksize=cksize):
            sent_id, df_sents = self.sentences_df(df, sent_id)
            df_sents.to_sql('sentences', conn, if_exists='append', index=False,chunksize=10)
            i+=cksize
            print("Extracted sentences from", i, "articles")

        

if __name__ == "__main__":
    DocumentSentences(dbfile= "/home/cyrielle/Codes/clean_code/DataStorage/of/2019_data.db", first_time = False).execute()