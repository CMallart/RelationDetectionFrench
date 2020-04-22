#!/usr/bin/env python
import numpy as np
np.random.seed(7)

import collections
import pandas as pd
import matplotlib.pyplot as plt
import time
import click

from gensim.models.keyedvectors import FastTextKeyedVectors
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, f1_score, precision_score, recall_score

from keras import optimizers, metrics
from keras.layers import LSTM, Dense, Dropout, Input, Activation, LeakyReLU, Dot, concatenate
from keras.models import Model
from keras.callbacks import History, ReduceLROnPlateau, EarlyStopping
import keras.backend as K
from keras.engine.topology import Layer

import warnings
warnings.simplefilter('ignore')


properties_names = {"":"None",
"http://www.wikidata.org/prop/direct/P1001":"juridiction concernée",
"http://www.wikidata.org/prop/direct/P101":"domaine d'activité",
"http://www.wikidata.org/prop/direct/P1018":"institution de normalisation",
"http://www.wikidata.org/prop/direct/P102":"parti politique",
"http://www.wikidata.org/prop/direct/P103":"langue maternelle",
"http://www.wikidata.org/prop/direct/P1038":"parentèle",
"http://www.wikidata.org/prop/direct/P1040":"monteur",
"http://www.wikidata.org/prop/direct/P105":"rang taxinomique",
"http://www.wikidata.org/prop/direct/P106":"occupation",
"http://www.wikidata.org/prop/direct/P1066":"élève de",
"http://www.wikidata.org/prop/direct/P1071":"lieu de fabrication",
"http://www.wikidata.org/prop/direct/P108":"employeur",
"http://www.wikidata.org/prop/direct/P110":"illustrateur",
"http://www.wikidata.org/prop/direct/P112":"fondateur",
"http://www.wikidata.org/prop/direct/P1142":"idéologie politique",
"http://www.wikidata.org/prop/direct/P118":"ligue",
"http://www.wikidata.org/prop/direct/P119":"lieu de sépulture",
"http://www.wikidata.org/prop/direct/P1269":"aspect de",
"http://www.wikidata.org/prop/direct/P127":"propriétaire",
"http://www.wikidata.org/prop/direct/P1302":"principales localités desservies",
"http://www.wikidata.org/prop/direct/P1303":"instrument de musique pratiqué",
"http://www.wikidata.org/prop/direct/P131":"localisation administrative",
"http://www.wikidata.org/prop/direct/P1321":"lieu d'origine (Suisse)",
"http://www.wikidata.org/prop/direct/P1327":"partenaire professionnel ou sportif",
"http://www.wikidata.org/prop/direct/P1336":"territoire revendiqué par",
"http://www.wikidata.org/prop/direct/P1344":"participant à",
"http://www.wikidata.org/prop/direct/P1346":"vainqueur",
"http://www.wikidata.org/prop/direct/P135":"mouvement",
"http://www.wikidata.org/prop/direct/P136":"genre",
"http://www.wikidata.org/prop/direct/P1365":"remplace",
"http://www.wikidata.org/prop/direct/P1366":"remplacé par",
"http://www.wikidata.org/prop/direct/P137":"opérateur",
"http://www.wikidata.org/prop/direct/P1376":"capitale de",
"http://www.wikidata.org/prop/direct/P138":"nommé en référence à",
"http://www.wikidata.org/prop/direct/P1382":"coïncide avec",
"http://www.wikidata.org/prop/direct/P1389":"certificat de produit",
"http://www.wikidata.org/prop/direct/P140":"religion",
"http://www.wikidata.org/prop/direct/P1408":"ville de licence",
"http://www.wikidata.org/prop/direct/P1412":"langues parlées, écrites ou signées",
"http://www.wikidata.org/prop/direct/P1431":"producteur délégué",
"http://www.wikidata.org/prop/direct/P1435":"statut patrimonial",
"http://www.wikidata.org/prop/direct/P144":"basé sur",
"http://www.wikidata.org/prop/direct/P1441":"présent dans l'œuvre",
"http://www.wikidata.org/prop/direct/P150":"contient les subdivisions territoriales administratives",
"http://www.wikidata.org/prop/direct/P1532":"nationalité sportive",
"http://www.wikidata.org/prop/direct/P1542":"a pour effet",
"http://www.wikidata.org/prop/direct/P155":"précédé par",
"http://www.wikidata.org/prop/direct/P1552":"caractérisé par",
"http://www.wikidata.org/prop/direct/P156":"suivi par",
"http://www.wikidata.org/prop/direct/P1589":"point le plus bas",
"http://www.wikidata.org/prop/direct/P159":"siège social",
"http://www.wikidata.org/prop/direct/P161":"distribution",
"http://www.wikidata.org/prop/direct/P162":"producteur",
"http://www.wikidata.org/prop/direct/P166":"distinction reçue",
"http://www.wikidata.org/prop/direct/P169":"directeur général",
"http://www.wikidata.org/prop/direct/P17":"pays",
"http://www.wikidata.org/prop/direct/P170":"créateur",
"http://www.wikidata.org/prop/direct/P171":"taxon supérieur",
"http://www.wikidata.org/prop/direct/P172":"groupe ethnique",
"http://www.wikidata.org/prop/direct/P175":"interprète",
"http://www.wikidata.org/prop/direct/P176":"fabricant",
"http://www.wikidata.org/prop/direct/P178":"développeur",
"http://www.wikidata.org/prop/direct/P179":"dans la série",
"http://www.wikidata.org/prop/direct/P180":"dépeint",
"http://www.wikidata.org/prop/direct/P1830":"propriétaire de",
"http://www.wikidata.org/prop/direct/P184":"directeur de thèse",
"http://www.wikidata.org/prop/direct/P185":"doctorant",
"http://www.wikidata.org/prop/direct/P186":"matériau",
"http://www.wikidata.org/prop/direct/P1877":"d'après une œuvre de",
"http://www.wikidata.org/prop/direct/P1889":"à ne pas confondre avec",
"http://www.wikidata.org/prop/direct/P19":"lieu de naissance",
"http://www.wikidata.org/prop/direct/P190":"jumelage ou partenariat",
"http://www.wikidata.org/prop/direct/P197":"gare voisine",
"http://www.wikidata.org/prop/direct/P1995":"spécialité médicale",
"http://www.wikidata.org/prop/direct/P20":"lieu de mort",
"http://www.wikidata.org/prop/direct/P205":"pays souverain sur le bassin versant",
"http://www.wikidata.org/prop/direct/P206":"baigné par",
"http://www.wikidata.org/prop/direct/P22":"père",
"http://www.wikidata.org/prop/direct/P2341":"autochtone de",
"http://www.wikidata.org/prop/direct/P2389":"organisation dont la fonction est à la tête",
"http://www.wikidata.org/prop/direct/P2416":"discipline sportive pratiquée en compétition",
"http://www.wikidata.org/prop/direct/P25":"mère",
"http://www.wikidata.org/prop/direct/P2541":"zone opérationnelle",
"http://www.wikidata.org/prop/direct/P26":"conjoint",
"http://www.wikidata.org/prop/direct/P2632":"lieu de détention",
"http://www.wikidata.org/prop/direct/P264":"label discographique",
"http://www.wikidata.org/prop/direct/P2650":"intéressé par",
"http://www.wikidata.org/prop/direct/P27":"pays de nationalité",
"http://www.wikidata.org/prop/direct/P272":"société de production",
"http://www.wikidata.org/prop/direct/P276":"lieu",
"http://www.wikidata.org/prop/direct/P2789":"se connecte avec",
"http://www.wikidata.org/prop/direct/P279":"sous-classe de",
"http://www.wikidata.org/prop/direct/P282":"système d'écriture",
"http://www.wikidata.org/prop/direct/P2925":"domaine du saint ou de la divinité",
"http://www.wikidata.org/prop/direct/P2936":"langue utilisée",
"http://www.wikidata.org/prop/direct/P30":"continent",
"http://www.wikidata.org/prop/direct/P3075":"religion officielle",
"http://www.wikidata.org/prop/direct/P3094":"se développe de",
"http://www.wikidata.org/prop/direct/P3095":"pratiqué par",
"http://www.wikidata.org/prop/direct/P31":"nature de l'élément",
"http://www.wikidata.org/prop/direct/P3373":"frère ou sœur",
"http://www.wikidata.org/prop/direct/P344":"directeur de la photographie",
"http://www.wikidata.org/prop/direct/P35":"chef d'État",
"http://www.wikidata.org/prop/direct/P355":"organisation filiale",
"http://www.wikidata.org/prop/direct/P36":"capitale",
"http://www.wikidata.org/prop/direct/P361":"partie de",
"http://www.wikidata.org/prop/direct/P366":"usage",
"http://www.wikidata.org/prop/direct/P37":"langue officielle",
"http://www.wikidata.org/prop/direct/P371":"présentateur",
"http://www.wikidata.org/prop/direct/P3729":"rang immédiatement inférieur",
"http://www.wikidata.org/prop/direct/P3730":"rang immédiatement supérieur",
"http://www.wikidata.org/prop/direct/P375":"lanceur",
"http://www.wikidata.org/prop/direct/P376":"sur le corps astronomique",
"http://www.wikidata.org/prop/direct/P38":"monnaie",
"http://www.wikidata.org/prop/direct/P3828":"élément de parure",
"http://www.wikidata.org/prop/direct/P39":"fonction",
"http://www.wikidata.org/prop/direct/P40":"enfant",
"http://www.wikidata.org/prop/direct/P403":"se jette dans",
"http://www.wikidata.org/prop/direct/P407":"langue de l'œuvre, du nom ou du terme",
"http://www.wikidata.org/prop/direct/P410":"grade militaire",
"http://www.wikidata.org/prop/direct/P412":"tessiture",
"http://www.wikidata.org/prop/direct/P413":"position de jeu/spécialité",
"http://www.wikidata.org/prop/direct/P449":"diffuseur original",
"http://www.wikidata.org/prop/direct/P451":"partenaire",
"http://www.wikidata.org/prop/direct/P457":"texte fondateur",
"http://www.wikidata.org/prop/direct/P460":"réputé identique à",
"http://www.wikidata.org/prop/direct/P461":"contraire",
"http://www.wikidata.org/prop/direct/P463":"membre de",
"http://www.wikidata.org/prop/direct/P495":"pays d’origine",
"http://www.wikidata.org/prop/direct/P4969":"œuvre dérivée",
"http://www.wikidata.org/prop/direct/P50":"auteur",
"http://www.wikidata.org/prop/direct/P501":"enclavé dans",
"http://www.wikidata.org/prop/direct/P520":"armement",
"http://www.wikidata.org/prop/direct/P527":"comprend",
"http://www.wikidata.org/prop/direct/P53":"famille",
"http://www.wikidata.org/prop/direct/P533":"cible",
"http://www.wikidata.org/prop/direct/P54":"membre de l'équipe de sport",
"http://www.wikidata.org/prop/direct/P551":"résidence",
"http://www.wikidata.org/prop/direct/P559":"extrémité",
"http://www.wikidata.org/prop/direct/P562":"banque émettrice",
"http://www.wikidata.org/prop/direct/P57":"réalisateur ou metteur en scène",
"http://www.wikidata.org/prop/direct/P58":"scénariste",
"http://www.wikidata.org/prop/direct/P6":"chef de l'exécutif",
"http://www.wikidata.org/prop/direct/P607":"conflit",
"http://www.wikidata.org/prop/direct/P61":"découvreur ou inventeur",
"http://www.wikidata.org/prop/direct/P610":"point culminant",
"http://www.wikidata.org/prop/direct/P611":"ordre religieux",
"http://www.wikidata.org/prop/direct/P641":"sport",
"http://www.wikidata.org/prop/direct/P647":"repêchage par",
"http://www.wikidata.org/prop/direct/P664":"organisateur",
"http://www.wikidata.org/prop/direct/P674":"personnages",
"http://www.wikidata.org/prop/direct/P676":"parolier",
"http://www.wikidata.org/prop/direct/P681":"composant cellulaire",
"http://www.wikidata.org/prop/direct/P6885":"région historique",
"http://www.wikidata.org/prop/direct/P69":"scolarité",
"http://www.wikidata.org/prop/direct/P706":"localisation géographique",
"http://www.wikidata.org/prop/direct/P710":"participant",
"http://www.wikidata.org/prop/direct/P737":"influencé par",
"http://www.wikidata.org/prop/direct/P740":"lieu de fondation",
"http://www.wikidata.org/prop/direct/P749":"organisation mère",
"http://www.wikidata.org/prop/direct/P750":"distributeur",
"http://www.wikidata.org/prop/direct/P793":"événement clé",
"http://www.wikidata.org/prop/direct/P800":"œuvre notable",
"http://www.wikidata.org/prop/direct/P802":"élève",
"http://www.wikidata.org/prop/direct/P807":"séparé de",
"http://www.wikidata.org/prop/direct/P828":"a pour cause",
"http://www.wikidata.org/prop/direct/P837":"jour de l'année",
"http://www.wikidata.org/prop/direct/P84":"architecte",
"http://www.wikidata.org/prop/direct/P840":"lieu de l'action",
"http://www.wikidata.org/prop/direct/P859":"sponsor",
"http://www.wikidata.org/prop/direct/P86":"compositeur",
"http://www.wikidata.org/prop/direct/P87":"librettiste",
"http://www.wikidata.org/prop/direct/P88":"commanditaire",
"http://www.wikidata.org/prop/direct/P921":"sujet ou thème principal",
"http://www.wikidata.org/prop/direct/P931":"localité desservie par cette infrastructure de transport",
"http://www.wikidata.org/prop/direct/P937":"lieu de travail",
"http://www.wikidata.org/prop/direct/P945":"allégeance",
"http://www.wikidata.org/prop/direct/P974":"affluent",
"http://www.wikidata.org/prop/direct/P98":"éditeur scientifique"}

###############################################################################################################
#Format data to the right input

def flatten(coll):
    for i in coll:
        if isinstance(i, collections.Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i

def get_lookup_tables(df):
    
    #for relation number
    relations = set(df.relation.tolist())
    relations = list(relations)
    global num_classes
    num_classes = len(relations)
    
        ##relation to number : 
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(relations)
    
    global relation_code_to_name
    relation_code_to_name = dict(zip(integer_encoded, relations))
    
        ##number to one-hot-encoded vector
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        
        ##relation name to encoding
    lookup_table_relations = dict(zip(relations, onehot_encoded))
    

    #for dependency tags
    deptags = ['', 'parataxis', 'ROOT', 'nsubj', 
    'nsubjpass', 'nmod', 'dep', 'cop',
    'ccomp', 'acl', 'name', 'det', 'nummod', 'conj', 'expl', 
     'aux', 'amod', 'compound', 'auxpass', 'advmod', 'cc', 'mwe', 'xcomp', 
    'advcl', 'mark', 'punct',  'acl:relcl', 'case', 'csubj', 
     'iobj', 'neg', 'nmod:poss', 'dobj', 'appos',  'discourse']
    deptags_codes = list(range(1, len(deptags)+1))
    lookup_table_deptags = dict(zip(deptags, deptags_codes))
    
    #for part-of-speech tags
    postags = ['', 'SCONJ', 'NUM', 'CONJ', 'NOUN', 'VERB', 'PRON', 'ADP', 'DET', 'ADV', 'PUNCT', 
               'INTJ', 'SYM', 'PROPN', 'PART', 'AUX', 'X', 'ADJ']
    postags_codes = list(range(1, len(postags)+1))
    lookup_table_postags = dict(zip(postags, postags_codes))
    
    #for the words, they are 1-hot encoded too. This time taken straigth from the features, no list
    vocab = set()
    col1 = df.word_path1.apply(lambda x : x.split("/"))
    batchvocab = set(list(flatten(col1)))
    vocab = batchvocab | vocab
    
    col2 = df.word_path2.apply(lambda x : x.split("/"))
    batchvocab = set(list(flatten(col2)))
    vocab = batchvocab | vocab
    
    vocab = list(vocab)+["UKNWN"]
    vocab_codes = list(range(1, len(vocab)+1))

    global vocab_size
    vocab_size = len(vocab)

    lookup_table_vocab = dict(zip(vocab, vocab_codes))
    
    return(lookup_table_relations,lookup_table_deptags, lookup_table_postags, lookup_table_vocab)


def data_reshape(df, lookup_table_relations, lookup_table_deptags, lookup_table_postags, lookup_table_vocab, maxlen=30):
    y = []
    features_words1 = []
    features_words2 = []
    features_pos1 = []
    features_pos2 = []
    features_dep1 = []
    features_dep2 = []

    for row in df.itertuples(): #ie for each sample
        y.append(row.relation!='') 
        seq_word1 =  [lookup_table_vocab[word] if ((type(word)==str) & (word in lookup_table_vocab)) else lookup_table_vocab["UKNWN"] for word in row.word_path1.split("/")]
        seq_word2 =  [lookup_table_vocab[word] if ((type(word)==str) & (word in lookup_table_vocab)) else lookup_table_vocab["UKNWN"] for word in row.word_path2.split("/")]
        seq_dep1 =  [lookup_table_deptags[dep.split(':')[0]] if "conj" in dep else lookup_table_deptags[dep] for dep in row.dep_path1.split("/")]
        seq_dep2 =  [lookup_table_deptags[dep.split(':')[0]] if "conj" in dep else lookup_table_deptags[dep] for dep in row.dep_path2.split("/")]
        seq_pos1 =  [lookup_table_postags[pos_tag] for pos_tag in row.pos_path1.split("/")]
        seq_pos2 =  [lookup_table_postags[pos_tag] for pos_tag in row.pos_path2.split("/")]
        features_words1.append(seq_word1)
        features_words2.append(seq_word2)
        features_dep1.append(seq_dep1)
        features_dep2.append(seq_dep2)
        features_pos1.append(seq_pos1)
        features_pos2.append(seq_pos2)
    padded_features_words1 = pad_sequences(features_words1, padding='post', maxlen=maxlen)
    padded_features_words2 = pad_sequences(features_words2, padding='post', maxlen=maxlen)
    padded_features_dep1 = pad_sequences(features_dep1, padding='post', maxlen=maxlen)
    padded_features_dep2 = pad_sequences(features_dep2, padding='post', maxlen=maxlen)
    padded_features_pos1 = pad_sequences(features_pos1, padding='post', maxlen=maxlen)
    padded_features_pos2 = pad_sequences(features_pos2, padding='post', maxlen=maxlen)
    X = [padded_features_words1, padded_features_words2, padded_features_dep1, padded_features_dep2, 
         padded_features_pos1, padded_features_pos2]               
    return(X, y)


def sets_test_train(feats_file = "/home/cyrielle/Codes/clean_code/Features/data/features_ratio1_definitve_XuBiLSTM.csv"):
    df = pd.read_csv(feats_file)
    df.fillna(value="", inplace=True)
    df.set_index(["entity1","entity2", "original_sentence"], inplace=True)

    with open('indexes_test.json') as json_file:
        ind_test = json.load(json_file)
        df_test = df[df.id_entities_pair.isin(ind_test)]
        
    with open('indexes_train.json') as json_file:
        ind_train = json.load(json_file)
        df_train = df[df.id_entities_pair.isin(ind_train)]

    #train
    lookup_table_relations2, lookup_table_deptags2, lookup_table_postags2, lookup_table_vocab2 = get_lookup_tables(df_train)
    
    dataX, datay = data_reshape(df_train, lookup_table_relations2, lookup_table_deptags2, lookup_table_postags2, lookup_table_vocab2) 
    index_train, index_val = train_test_split(range(0, len(datay)), test_size=0.25, random_state=7)
    X_train = [[dataXj[i] for i in index_train] for dataXj in dataX]
    X_val = [[dataXj[i] for i in index_val] for dataXj in dataX]#[dataXi[index_val] for dataXi in dataX]
    y_train = [datay[i] for i in index_train]
    y_train = np.array(y_train)
    y_val = [datay[i] for i in index_val]
    y_val = np.array(y_val)

    #test
    X_test, y_test = data_reshape(df_test, lookup_table_relations2, lookup_table_deptags2, lookup_table_postags2, lookup_table_vocab2)

    return(X_train, X_val, X_test, y_train, y_val, y_test)



###############################################################################################################
#Define and train models

class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
            
    author : sergeyf
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


def build_model_xu(X_train):
    word_embedding_dim = 200
    pos_embedding_dim = 50
    dep_embedding_dim = 50

    word_vocab_size = vocab_size+1
    pos_vocab_size = 18+1
    dep_vocab_size = 35+1

    hidden_layer_size =100
    word_state_size = 200
    other_state_size =50

    #inputs
    input_word1 = Input(shape = (None,))
    input_word2 = Input(shape = (None,))
    input_dep1 = Input(shape =  (None,))
    input_dep2 = Input(shape =  (None,))
    input_pos1 = Input(shape = (None,))
    input_pos2 = Input(shape = (None,))

    #embeddings
    embed_words = Embedding(word_vocab_size, word_embedding_dim, mask_zero=True)
    embed_pos = Embedding(pos_vocab_size, pos_embedding_dim, mask_zero=True)
    embed_dep = Embedding(dep_vocab_size, dep_embedding_dim, mask_zero=True)

    #embed the inputs of both sides with the same embedding layers
    embedded_words1 = embed_words(input_word1)
    embedded_words2 = embed_words(input_word2)
    embedded_pos1 = embed_pos(input_pos1)
    embedded_pos2 = embed_pos(input_pos2)
    embedded_dep1 = embed_dep(input_dep1)
    embedded_dep2 = embed_dep(input_dep2)

    #lstms for each channel
    lstm_word1 = LSTM(units = word_state_size, return_sequences=True)(embedded_words1)
    lstm_word1_nomask = ZeroMaskedEntries()(lstm_word1)
    pool_word1 = GlobalMaxPooling1D()(lstm_word1_nomask) #shape = num samples, num_features

    lstm_pos1 = LSTM(units = other_state_size, return_sequences=True)(embedded_pos1)
    lstm_pos1_nomask = ZeroMaskedEntries()(lstm_pos1)
    pool_pos1 = GlobalMaxPooling1D()(lstm_pos1_nomask)

    lstm_dep1 = LSTM(units = other_state_size, return_sequences=True)(embedded_dep1)
    lstm_dep1_nomask = ZeroMaskedEntries()(lstm_dep1)
    pool_dep1 = GlobalMaxPooling1D()(lstm_dep1_nomask)

    lstm_word2 = LSTM(units = word_state_size, return_sequences=True)(embedded_words2)
    lstm_word2_nomask = ZeroMaskedEntries()(lstm_word2)
    pool_word2 = GlobalMaxPooling1D()(lstm_word2_nomask)

    lstm_pos2 = LSTM(units = other_state_size, return_sequences=True)(embedded_pos2)
    lstm_pos2_nomask = ZeroMaskedEntries()(lstm_pos2)
    pool_pos2 = GlobalMaxPooling1D()(lstm_pos2_nomask)

    lstm_dep2 = LSTM(units = other_state_size, return_sequences=True)(embedded_dep2)
    lstm_dep2_nomask = ZeroMaskedEntries()(lstm_dep2)
    pool_dep2 = GlobalMaxPooling1D()(lstm_dep2_nomask)

    hidden_word_pre = Dot(axes =-1, normalize = True)([pool_word1, pool_word2]) #dot product of the two tensors 
    hidden_word = Dense(word_state_size, name = "Dense_word")(hidden_word_pre)

    hidden_pos_pre = Dot(axes =-1, normalize = True)([pool_pos1, pool_pos2])
    hidden_pos = Dense(other_state_size, name= "Dense_pos")(hidden_pos_pre)

    hidden_dep_pre = Dot(axes =-1, normalize = True)([pool_dep1, pool_dep2])
    hidden_dep = Dense(other_state_size, name = "Dense_dep")(hidden_dep_pre)

    main_hidden_pre = concatenate([hidden_word, hidden_pos, hidden_dep])
    main_hidden = Dense(hidden_layer_size, name = "Dense_main")(main_hidden_pre)

    output = Dense(num_classes, activation='softmax')(main_hidden)

    #define model
    model = Model(inputs=[input_word1, input_word2, input_dep1, input_dep2, input_pos1,input_pos2], outputs=output)
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss= "categorical_crossentropy",metrics = [metrics.mean_squared_error], optimizer = adam)
    return(model)


def train_model_xu(model, X_train, X_val, y_train, y_val, result_folder):
    #callbacks
    history = History()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    callback_list = [history, reduce_lr]

    #fit model
    model.fit(x= [X_train[0],X_train[1],X_train[2],X_train[3],X_train[4], X_train[5]],
          y= y_train,
          epochs=20,
          batch_size=1,
          validation_data=([X_val[0],X_val[1],X_val[2],X_val[3],X_val[4], X_val[5]], y_val),
          callbacks=callback_list)

    #save or not the model
    if result_folder is not None :
        checkpoint_filepath= result_folder +"model_{}.model".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
        model.save(checkpoint_filepath)
    return(model)


###############################################################################################################
#Test and results

def save_classif_reports(y_test, y_pred,report_file= None, print=True):
    if report_file is not None :
        with open(report_file, 'w') as f :
            f.write(classification_report(y_test, y_pred))
            f.write("F1_score macro: "+str(f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro')))
            f.write("F1_score weigthed: "+str(f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted')))
            f.write("Precision score macro: "+str(precision_score(y_test, y_pred, average="macro")))
            f.write("Precision score weighted: "+str(precision_score(y_test, y_pred, average="weighted")))
            f.write("Recall score macro: "+str(recall_score(y_test, y_pred, average="macro")))
            f.write("Recall score weighted: "+str(recall_score(y_test, y_pred, average="weighted")))
    if print == True :
        print("F1_score macro: "+str(f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro')))
        print("F1_score weigthed: "+str(f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted')))
        print("Precision score macro: "+str(precision_score(y_test, y_pred, average="macro")))
        print("Precision score weighted: "+str(precision_score(y_test, y_pred, average="weighted")))
        print("Recall score macro: "+str(recall_score(y_test, y_pred, average="macro")))
        print("Recall score weighted: "+str(recall_score(y_test, y_pred, average="weighted")))


def test_model(trained_model, X_test, y_test, result_folder):
    #predict the Y with model
    y_pred = trained_model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)
    predictions = [properties_names[relation_code_to_name[i]] for i in y_pred_classes]

    #turn y_test back into variable
    y_true_classes = y_test.argmax(axis=-1)
    reel = [properties_names[relation_code_to_name[i]] for i in y_true_classes]
    if result_folder is not None :
        report_path = result_folder +"classif_report_{}".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
        save_classif_reports(reel, predictions, report_file = report_path, print=True)
    else:
        save_classif_reports(reel, predictions, print=True)



###############################################################################################################
#Build, train and test 
@click.command()
@click.argument(
    '--feats_file', 
    type=click.File('rb'),
    help = 'Path where to find the .csv with the features for our model.')
@click.option(
    '--result_folder',
    type=click.File('wb'),
    help='Path of the file where to keep the .csv containing the features for our model.'
         'If not provided, no output of anything -not even the model- will be retained.',
)
def build_and_train_xu(feats_file, result_folder):
    X_train, X_val, X_test, y_train, y_val, y_test = sets_test_train(feats_file)
    model = build_model_xu(X_train)
    trained_model = train_model_xu(model, X_train, X_val,  y_train, y_val, result_folder)
    test_model(trained_model,X_test, y_test, result_folder)


if __name__ == "__main__":
    build_and_train_xu()
    















