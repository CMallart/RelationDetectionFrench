#!/usr/bin/env python
import numpy as np
np.random.seed(7)

import collections
import pandas as pd
import matplotlib.pyplot as plt
import time

from gensim.models.keyedvectors import FastTextKeyedVectors
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, f1_score, precision_score, recall_score

from keras import optimizers, metrics
from keras.layers import LSTM, Dense, Dropout, Input, Activation, LeakyReLU, Dot, concatenate
from keras.models import Model
from keras.callbacks import History, ReduceLROnPlateau, EarlyStopping

import click
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

def get_lookup_tables(df):
    """
    Create the tables of encoding for the relations, the depednency tags, the part-of-speech tags and the entity types
    """
        #for relation number
    relations = set(df.relation.tolist())
    relations = list(relations)
    
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
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list(deptags))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    lookup_table_deptags = dict(zip(list(deptags), onehot_encoded))
    
        #for part-of-speech tags
    postags = ['', 'SCONJ', 'NUM', 'CONJ', 'NOUN', 'VERB', 'PRON', 'ADP', 'DET', 'ADV', 'PUNCT', 'INTJ', 'SYM', 'PROPN', 'PART', 'AUX', 'X', 'ADJ']
    label_encodert = LabelEncoder()
    integer_encodedt = label_encodert.fit_transform(list(postags))
    integer_encodedt = integer_encodedt.reshape(len(integer_encodedt), 1)
    onehot_encodert = OneHotEncoder(sparse=False)
    onehot_encodedt = onehot_encodert.fit_transform(integer_encodedt)
    lookup_table_postags = dict(zip(list(postags), onehot_encodedt))

        #for entity types
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
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(entity_types)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    lookup_table_ent_types = dict(zip(entity_types, onehot_encoded))

    return(lookup_table_relations, lookup_table_ent_types, lookup_table_deptags, lookup_table_postags)

def split_if(x):
    if type(x) != float :
        return x.split("/")
    else :
        return ""

def flatten(coll):
    for i in coll:
        if isinstance(i, collections.Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i



def data_reshape(df, lookup_table_relations, lookup_table_ent_types, lookup_table_deptags, lookup_table_postags, len_max_seq=30):
    wordvectors = FastTextKeyedVectors.load("/home/cyrielle/Codes/clean_code/DataStorage/pretrained/model.bin")

    #Create array X and y
    features_flat = []
    labels =[]
    seq_flat=[]
    
    for row in df.itertuples():
        labels.append(int(row.relation!= ''))
        #'flat' features
        ent1t = lookup_table_ent_types[row.ent1type]
        ent2t = lookup_table_ent_types[row.ent2type] 
        vec = np.concatenate((ent1t, ent2t))
        features_flat.append(vec)

        len_seq=0
        for i in range(0, len(row.shortest_dependency_path_p.split("/"))):
            #'sequence' features
            current_word = row.shortest_dependency_path_w.split("/")[i]
            current_dependency = row.shortest_dependency_path_p.split("/")[i]
            current_pos_tag= row.shortest_dependency_path_t.split("/")[i]
            seq_sdp_w = wordvectors[current_word] if type(current_word)==str else np.zeros(100)
            seq_sdp_p = lookup_table_deptags[current_dependency.split(':')[0]] if "conj" in current_dependency else lookup_table_deptags[current_dependency]
            seq_sdp_t = lookup_table_postags[current_pos_tag]
            vec_seq = np.concatenate((seq_sdp_w, seq_sdp_p, seq_sdp_t)) 
            seq_flat.append(vec_seq)
            len_seq+=1
        while len_seq<len_max_seq:
            seq_flat.append(np.zeros(len(vec_seq)))
            len_seq+=1
    #add labels and reshpare into 3-dimensional tensors
    labels = np.array(labels)
    features_words = np.array(features_flat) 
    seq = np.reshape(seq_flat, (df.shape[0], len_max_seq, len(vec_seq)))  
    return(features_words, seq, labels)


def sets_test_train(feats_file):
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
    lookup_table_relations,lookup_table_ent_types, lookup_table_deptags, lookup_table_postags = get_lookup_tables(df_train)
    dataX, dataXseq, datay = data_reshape(df_train, lookup_table_relations, lookup_table_ent_types, lookup_table_deptags, lookup_table_postags) 
    X_train, X_val, X_train_seq, X_val_seq, y_train, y_val = train_test_split(dataX, dataXseq, datay, test_size=0.25, random_state=7)
    #test
    lookup_table_relations2,lookup_table_ent_types2, lookup_table_deptags2, lookup_table_postags2 = get_lookup_tables(df_test)
    X_test, X_test_seq, y_test = data_reshape(df_test, lookup_table_relations2,lookup_table_ent_types2, lookup_table_deptags2, lookup_table_postags2) 
     
    return(X_train, X_val, X_test, X_train_seq, X_val_seq, X_test_seq, y_train, y_val, y_test)





###############################################################################################################
#Define and train models


def build_model_ours(X_train, X_train_seq):
    #Branch with only the embedding vectors, 2D
    input_vec1 = Input(shape = X_train[0].shape)
    branch1 = Dense(1024)(input_vec1)
    branch1_2 = LeakyReLU(alpha=0.005)(branch1)
    branch1_3 = Dropout(0.5)(branch1_2)

    #Branch with the sequences only
    input_vec2 = Input(shape=(X_train_seq.shape[1],X_train_seq.shape[2]))
    branch2 = LSTM(units = 1024)(input_vec2)
    branch2_1 = LeakyReLU(alpha=0.5)(branch2)
    branch2_2 = Dropout(0.07)(branch2_1)

    #Merge the two branches
    merge = Dot(axes =1, normalize = True)([branch1_3, branch2_2])
    dense_merged = Dense(np.power(2, 7))(merge)
    leakyrelu =  LeakyReLU(alpha=0.5)(dense_merged)
    dropout = Dropout(0.5)(leakyrelu)
    output = Dense(1, activation='sigmoid')(dropout)

    #final model
    model = Model(inputs=[input_vec1, input_vec2], outputs=output)

    #optimization strategy
    adam = optimizers.Adam(lr=0.04)
    model.compile(loss='binary_crossentropy', metrics=[metrics.binary_accuracy], optimizer=adam)
    return(model)


def train_model_ours(model, X_train, X_val,  X_train_seq, X_val_seq,  y_train, y_val, result_folder):
    #callbacks
    history = History()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    callback_list = [history, reduce_lr]

    #fit model
    model.fit(x=[X_train,X_train_seq] ,
            y= y_train,
            epochs=25,
            batch_size=16,
            validation_data=([X_val, X_val_seq], y_val),
            callbacks=callback_list)

    #save or not the model
    if result_folder is not None :
        checkpoint_filepath= result_folder +"model_{}.model".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
        model.save(checkpoint_filepath)
    return(model)


###############################################################################################################
#Test and results

def threshold(y_test, y_pred, report_file=None):
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    cutting_point = threshold[np.argmax(tpr - fpr)]
    y_pred = y_pred > cutting_point
    if report_file is not None :
        with open(report_file, 'w') as f :
                f.write("Optimal cutting point ROC :" + str(cutting_point))
    print("Optimal cutting point: ", cutting_point)
    return(fpr, tpr, y_pred)
    

def rocplot(fpr, tpr, rocplot_path=None):
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if rocplot_path is not None:
        plt.savefig(rocplot_path)
    plt.show()

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


def test_model(trained_model,X_test, X_test_seq, y_test, result_folder):
    #predict the Y with model
    y_pred = trained_model.predict([X_test, X_test_seq])
    if result_folder is not None :
        report_path = result_folder +"classif_report_{}".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
        rocplot_path = result_folder +"roc_{}".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
        fpr, tpr, y_pred = threshold(y_test, y_pred, report_path)
        rocplot(fpr, tpr, rocplot_path)
        save_classif_reports(y_test, y_pred, report_file= report_path, print=True)
    else:
        fpr, tpr, y_pred = threshold(y_test, y_pred)
        rocplot(fpr, tpr)
        save_classif_reports(y_test, y_pred, print=True)



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
def build_and_train_ours(feats_file, result_folder):
    X_train, X_val, X_test, X_train_seq, X_val_seq, X_test_seq, y_train, y_val, y_test = sets_test_train(feats_file)
    model = build_model_ours(X_train, X_train_seq)
    trained_model = train_model_ours(model, X_train, X_val,  X_train_seq, X_val_seq,  y_train, y_val, result_folder)
    test_model(trained_model,X_test, X_test_seq, y_test, result_folder)




if __name__ == "__main__":
    #TODO :check where indexes files are beigng kept
    build_and_train_ours()
    















