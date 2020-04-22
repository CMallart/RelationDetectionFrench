########################################################
#
#This file to create collapsed dependencies for French,
#
# Based on Stanford CoreNLP, requires their server 
# 
# 
# Author : Cyrielle Mallart
#
#A simple tool for extracting the shortest dependency path
##########################################################

#from nltk.parse.corenlp import CoreNLPDependencyParser
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

import re
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import sys
from tqdm import tqdm 
from EntitiesExtraction import EntityExtractor
import sqlite3

from EnhancedPPDepParser import CoreNLPDependencyParser

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore")



mapping_pos = {'ADJ':'a', #see https://github.com/sammous/spacy-lefff for list of lefff tags
    'ADV':'r',  #see https://universaldependencies.org/u/pos/ for list of universal dependencies tags
    'VERB':'v', 
    'NOUN':'n',
    'ADP':'prep',
    'AUX':'all',
    'CONJ':'coo',
    'DET':'det',
    'INTJ':'all',
    'NUM':'exclude',
    'PART':'all',
    'PRON':'all',
    'PROPN':'np',
    'PUNCT':'all',
    'SCONJ':'prel',
    'SYM':'exclude',
    'X':'all',
    'TOP':'exclude' }

pronouns_tags = ['cln',#pronoms personnels sujet
                 'cla', #pronoms personnels complément d'objet direct
                 'cld', #pronoms personnels complément d'objet indirect (ou 'forme disjointe')
                 'clg',#pronoms personnels 'en' et 'y'
                 #'cll', #same ?!, does bullshit
                 'prel', #pronom relatif
                 'pro',
                 'clr']#random pronom, idk anymore

lemmatizer = FrenchLefffLemmatizer()


class Dependencies():

    def __init__(self, sentence, port=9004, insecables=[], fr_lemmatize = True):
        self.sentence = sentence.rstrip('.')
        self.sentence = re.sub(r'(.?)([\.,;:\?!()\[\]\{\}«»\'\"\—\/’&])', '\\1 \\2 ', self.sentence)

        self.insecables= insecables
        self.fr_lemmatize = fr_lemmatize
        self.corenlpparser = CoreNLPDependencyParser(url='http://localhost:'+ str(port))

        try :
            temp_parse = list(self.corenlpparser.raw_parse(self.sentence))[0]
            self.parse = {i:temp_parse.nodes[i] for i in range(0, len(temp_parse.nodes))}
            self.number_nodes = len(self.parse)

            self.graph = self.create_graph()
            #self.replace_punct()
        except AssertionError:
            self.parse= None
            self.graph= None
            print("NLTK ERROR : something it did not like in sentence ", self.sentence)
        except StopIteration:
            self.parse = None
            self.graph = None
            print("No parse in the sentence", self.sentence)
        
        
        
        

    #tools for graph creation
    ############################################################################# 
    def unique_list(self, l):
        x=[]
        x.append(l[0])
        for i in range(1,len(l)):
            if l[i] != l[i-1]:
                x.append(l[i])
        return x
    
    def deal_with_duplicates(self):
        """As enhancedPLusPLusDependencies sometimes have duplicates, deal with that
        It can be up to 4 or 5 times the word, but always consecutively (to the best og my knowledge)

        So if the next node has a word that is exactly the same as the previous node, we consider that we have a duplicate

        We keep the last word met in memory and compare to the next node's word each time.
        """
        to_remove=[]
        replacements = []
        last_seen_word = self.parse[0]["word"]
        last_index_seen = 0
        for i in range(1, self.number_nodes):
            current_node = self.parse[i]
            if current_node['word'] == last_seen_word: #if we have a duplicate
                #we remove the next nodes with appearance of the word
                to_remove.append(i)
                #and we indicate to redirect all the edges to the first node that displays that word
                replacements.append(last_index_seen)
            else :
                last_seen_word = current_node["word"]
                last_index_seen = i

        return(to_remove, replacements)
    

    def find_sub_list(self,sl,l, duplicates):
        """
        Helper function to gt he indices of a sublist in a list
        Adapted to our case due to the shift of +1 indices in the list of nodes ouput by CoreNLPParser
        """
        duplicate_indices = [i-1 for i in duplicates] #to account for the shift in indices as the nodes start at 1
        for index in duplicate_indices : l[index] = ' ' #replace the duplicates by a blank string
        results=[]
        sll=len(sl)
        try :
            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                if l[ind:ind+sll]==sl:
                    results.append((ind+1,ind+sll))#remember that the nodes count as 0 the top node, it moves everything by 1
                elif [elmt for elmt in l[ind:ind+sll+1] if elmt != ' '] == sl:
                    results.append((ind+1,ind+sll+1))
                elif [elmt for elmt in l[ind:ind+sll+2] if elmt != ' '] == sl:
                    results.append((ind+1,ind+sll+2))
                elif [elmt for elmt in l[ind:ind+sll+3] if elmt != ' '] == sl: #entities made of 4 words, we can push until 5 if I see any #TODO
                    results.append((ind+1,ind+sll+3))   
        except :
            print("Finding the sublist for entities regroupment : issue")
            if sl == [] :
                print("The entites to find is an empty list")
            else :
                print(list(enumerate(l)), sl)
        return results


    def overlaps_entity(self, span, other_spans):
        same_beginning = [s for s in other_spans if s[0]==span[0]]
        bigger_end = [s for s in same_beginning if s[1] > span[1]]
        if len(bigger_end)>0:
            return(True)
        else :
            return(False)

                    
    def get_list_with_punct(self, l, punct):
        res=[]
        for item in l:
            res.extend([el+punct for el in item.split(punct)[:-1]] + [item.split(punct)[-1]])
        return res


    def get_lemma(self, tag, word, stanford_lemma):
        if self.fr_lemmatize :
            postag = mapping_pos[tag]
            if (postag != 'exclude'):
                lemma = lemmatizer.lemmatize(word, postag)
            elif word != None:
                lemma = word
            else :
                lemma = "TOP"
            try:
                if type(lemma)==list:
                    #pronouns special
                    if (len(lemma) >1) & (tag=='PRON'):
                        possible_lemma_forms = [lem for lem, pos in lemma if pos in pronouns_tags]
                        lem = possible_lemma_forms[0]
                    elif len(lemma)==0:
                        lem = word
                    else:
                        lem = lemma[0][0]
                elif type(lemma)==tuple:
                    lem= lemma[0]
                else:
                    lem = lemma
            except IndexError:
                print("WARNING: lemmatization approximation, in the sentence %s: word %s has Stanford tag %s, but lemmatized as %s. Taking original word" %(self.sentence, word,tag,lemma))
                lem = word
        else :
            lem = stanford_lemma
        return(lem.lower())



    def create_graph(self):
        """Convert the data in a list of nodes and edges into a networkx labeled directed graph."""
        nx_nodelist, nx_edgelist = self.get_list_real_nodes()
        g = nx.DiGraph()
        g.add_nodes_from(nx_nodelist)
        g.add_edges_from(nx_edgelist)

        return(g)



    def get_list_real_nodes(self):
        """
        This function retuns a node list and an edge list to build the graph
        The node list is the list of the nodes output by the CoreNLPParser, except that the 'insecables' phrases (ie entities)
            are gathered in a single node

        The original nodes as ouput by the parsing are stocked in self.parse.nodes.

        We maintain three -normally small- dictionaries and four lists:
            -one dict for just the entities, which gives a key to every surface form of an entity. ex {1:"Collège de France, 2:"Stéphane Bern"}
            -one dict that keeps for each entity the positions they have in the nodes. ex {1:[(0,2), (1(,17))], 2:[(7,10)]} 
            -one dic that keps the new adresses of the nodes that we reunite. ex : nodes 12,13,14 united in node 12 => {12:12, 13:12, 14:12}

            -one list with all the nodes that we don't need to transform
            -one list of all the nodes after treatment
            -one list of all the relations
            -one list of the duplicates of nodes in the parse, resulted from a problem in enhanced++ dependencies parsing. Used to solve double dependencies
        
        Returns two lists : one of edge nodes, one of edges
        """

        list_nodes = []
        words_in_original_nodes= [self.parse[i]['word'] for i in range(1, self.number_nodes)]
        
        duplicates, replacements =self.deal_with_duplicates()

        #########################################################################
        #Get the words that are close to each other in the sentence

        next_to_in_sentence = {}
        for i in range(1,self.number_nodes-1): #duplicates are taken care of with the replacements happening at the end
            next_to_in_sentence[i]=i+1

        #########################################################################
        #Regroup the entities 

        #Those in two words have been separated with CoreNLP, we need to regroup them and see them as one single entity

        #get a dictionary listing the entities and giving them a temporary code
        #result {0 : "Elizabeth II", 1:"Royaume Uni"...}
        entity_code = {}
        for i in range(0, len(self.insecables)):
            entity_code[i] = self.insecables[i]

        #for each entity identified by its code, find the nodes of the parse that express these entities
        #result { 0 : [(1,2), (10,11)],  1 : [(5,6)]}
        positions_for_each_entity={}
        for key in entity_code:
            value = entity_code[key]
            value = re.sub(r'(.?)([\.,;:\?!()\[\]\{\}«»\'\"\—\/’&])', '\\1 \\2 ', value)
            parse_insec = list(self.corenlpparser.raw_parse(value))[0]
            parse_insec = {i:parse_insec.nodes[i] for i in range(0, len(parse_insec.nodes))}
            split_entity_text = [parse_insec[i]['word'] for i in range(1, len(parse_insec))] #we parse the enity form to have the same split as the CoreNLP gave for the whole sentence
            split_entity_text = self.unique_list(split_entity_text) #remove the duplicates from the parse ... again
            positions_for_each_entity[key] = self.find_sub_list(split_entity_text, words_in_original_nodes, duplicates) #returns a list of tuples, ex 1:[(1,3), (5,7)]

        #if when finding indices of an entity, the ranges of two entities overlap, such as 'Le "Los Angeles Times", basé à Los Angeles,',
        #it is likely that the longest entity is the one at hand : here we have 'Los Angeles Times' not just 'Los Angeles' and 'Times'
        #so if they overlap we check who's the longest and keep that one
        flattened_positions = [item for sublist in positions_for_each_entity.values() for item in sublist]
        to_remove = []
        for key in positions_for_each_entity:
            value = positions_for_each_entity[key]
            for tup in value:
                if self.overlaps_entity(tup, flattened_positions):
                    to_remove.append(tup)
            for tup2 in to_remove:
                positions_for_each_entity[key].remove(tup2)
            to_remove=[]

        #########################################################################
        #Creation of the list of nodes

        #list  all the nodes that don't contain the "easy to graph" entities
        #note, the list_of_problem_nodes is never used after because it also contains all the duplicate nodes : it is not a list of only the splitted entities, it's a list of the rubbish
        list_of_problem_nodes = [i for i in duplicates]
        for key in positions_for_each_entity:
            value = positions_for_each_entity[key]
            for val in value : # for each tuple of 'range' in the times we found this entity, ie for each (x,y) in {k:[(x,y), (x,y), (x,y)...]}
                list_of_problem_nodes.extend( list(range(val[0], val[1]+1)) )

        ##easy ones
        #get the easy entities
        list_good_nodes = [i for i in range(1,self.number_nodes) if i not in list_of_problem_nodes]


        #create the definitive list of nodes, initializing with the easy nodes
        list_nodes.extend( [(self.parse[i]['address'], 
                    {'lemma':self.get_lemma(self.parse[i]['tag'], self.parse[i]['word'], self.parse[i]['lemma']), 
                        'tag':self.parse[i]['tag'], 
                        'word':self.parse[i]['word'],
                        'dep': self.parse[i]['rel']} 
                    ) for i in list_good_nodes] )

        ##start dealing with the hard ones

        #create and initilaize a dict that contains the "redirections" of all the grouped entities
        #ie, it will stock that if
        #  3:Los 4:Angeles 
        #  now 3: Los Angeles and the dicitionary takes {4:3}, 4 redirects to 3
        #Will be useful to redirect the dependency links further on
        new_adresses = {i: i for i in list_good_nodes}

        #for all the 'entities in two parts', create a new node to store them      
        for key in positions_for_each_entity:
            value = positions_for_each_entity[key]
            for val in value :
                address = self.parse[val[0]]['address'] #the new node will have as id (or adress) the one of the first word
                pos = self.parse[val[0]]['tag'] #the tag is often the one of the first word (often a NOUN)
                word = entity_code[key] #the original surface form of the entity is the one we input
                lemma = entity_code[key] #no need to lemmatize this, as it will be searched for in the graph later
                rel = self.parse[val[0]]['rel'] #the first word is the main one normally
                list_nodes.append( (address,
                                    {"lemma": lemma,
                                    'tag': pos,
                                    "word": word,
                                    "dep" : rel}) )
                for i in range(val[0], val[1]+1) : 
                #and not forget to link every node of the parse to this new node's adress
                    new_adresses[i] = address

        #for all the duplicates
        for ind, n in enumerate(duplicates): #that way, if a dependency link points to a duplicated node, we make it point to the duplicate node we kept
            potential_replacement = replacements[ind]
            if potential_replacement in new_adresses : #if the replacement already exists in the dictionnary and points to a new node, we echo these changes
                new_adresses[n] = new_adresses[potential_replacement]
            else :
                new_adresses[n] = potential_replacement

        #########################################################################
        #Creation of the list of edges

        #dependency edges
        edges =[]
        for n in range(1, self.number_nodes) :
            if n not in duplicates: #we do not care about adding the duplicate nodes, it'd only add useless noise
                if (self.parse[n]['head'] != 0) :
                    edges.append( (new_adresses[self.parse[n]['head']], new_adresses[n], {'type' :'dep', "weight": 0.2})) #use the 'redirecting' dictionary
                if n in next_to_in_sentence : 
                    edges.append( (new_adresses[n], new_adresses[next_to_in_sentence[n]], {'type' :'before', "weight":1}))  #add the edges for words that were next to each other
        
        #we do not allow for loops
        edges = [item for item in edges if item[0]!=item[1]] 
        return list_nodes, edges

        


    #remedy to some issues on the graph : conjunctions and punctuation
    ############################################################################

    def replace_punct(self):
        """
        Not to use, it's full of holes and issues, makes the path very often missing one or two entities
        """
        #first pass to remove the punctuation - cannot do this in the main loop or else we run in issues for replacing the edges further on
        add = []
        remove = []
        for node in self.graph.nodes:
            try :
                if (self.graph.nodes[node]['dep']=='punct') & (len(self.graph.nodes[node]['word']) <2):
                    sons = list(self.graph.successors(node))
                    if len(sons)>0:
                        son= sons[0]
                        #if that punct sign was followed by something, we add to the head of the punctuation
                        add.append((list(self.graph.predecessors(node))[0], son, {'type' :'dep', "weight": 0.2}))
                    remove.append(node)
            except KeyError:
                print("Replace punct error")
                print(self.parse, self.sentence)
                #self.plot_tree_layout()
                sys.exit(1)
        self.graph.add_edges_from(add)
        self.graph.remove_nodes_from(remove)


    #plotting toolse.search(r"<e1>(.*?)</e1>", s).start() + 4
    ############################################################################
    
    def plot_tree_layout(self):
        """
        Plotting function that recreates an almost tree-like structure, more legible than the basic output

        The lemmas of the words are labels of the nodes and the relations are labels of the edges
        """
        write_dot(self.graph,'test.dot')

        edge_lab = dict([((frm, to), rel['weight'])
                for frm, to, rel in self.graph.edges(data=True)])
        node_lab=nx.get_node_attributes(self.graph, 'word')

        pos = graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos=pos, with_labels=False)
        nx.draw_networkx_labels(self.graph, pos=pos, labels=node_lab, font_size=16)
        nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_lab)
        plt.show()



    #shortest path 
    #############################################################################
    def shortest_path(self, entity1, entity2, index_e1, index_e2):
        """
        Given two nodes, find the shortest dependency path and print it, with edge labels and node labels
        Also take into account the 'number' of the entity, 
            that is whether it is the first or second or third... occurence of the entity surface form in the sentence
        """
        if self.graph ==None:
            print("Error in graph creation, the dependency parser did not create a parse")
            return((None, None, None))
        else :
            try :
                ent1 = [node for node in self.graph.nodes if self.graph.node[node]['word'] == entity1][index_e1]
                ent2 = [node for node in self.graph.nodes if self.graph.node[node]['word'] == entity2][index_e2]

            except IndexError:
                print("Error while finding shortest path : seems we can't find one or both of the entities")
                print(entity1, index_e1, entity2, index_e2,  [self.graph.node[node]['word'] for node in self.graph.node], 
                    self.sentence,
                    [node for node in self.graph.nodes if self.graph.node[node]['word'] == entity1],
                    [node for node in self.graph.nodes if self.graph.node[node]['word'] == entity2])
                #self.plot_tree_layout()
                return((None, None, None))

            except KeyError :
                print("Error while finding shortest path : the 'word' attribute of a node is not found")
                print(entity1, "/", entity2,"/", [node for node in self.graph.nodes], "/", self.sentence)
                print(self.plot_tree_layout())
                return((None, None, None))


            try :
                temp_graph = self.graph.to_undirected()
                parcours = nx.dijkstra_path(temp_graph, source=ent1, target=ent2, weight = 'weight' )
                words_path =[temp_graph.node[origin]['lemma'] for origin in parcours[1:-1]]
                dep_path = [temp_graph.node[origin]['dep'] for origin in parcours[1:-1]]
                pos_path = [temp_graph.node[origin]['tag'] for origin in parcours[1:-1]]

                shortest_dependency_path_words = "/".join(words_path)
                shortest_dependency_path_deps = "/".join(dep_path)
                shortest_dependency_path_postags = "/".join(pos_path)
                return(shortest_dependency_path_words,shortest_dependency_path_deps, shortest_dependency_path_postags )

            except nx.exception.NetworkXNoPath:
                print("No Path found between %s and %s, in the sentence %s" %(entity1, entity2, self.sentence))
                print(self.parse)
                #self.plot_tree_layout()
                return((None, None, None))


    #the hubs ?
    #############################################################################
    def measures(self):
        """
        Fun with measures, to see if anything is worth it 
        """
        pos = nx.spring_layout(self.graph)
        def draw(G, pos, measures, measure_name):
            
            nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.get_cmap('plasma'), 
                                        node_color=list(measures.values()),
                                        nodelist=measures.keys())
            nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
            
            node_lab=nx.get_node_attributes(G, 'word')
            nx.draw_networkx_labels(G, pos=pos, labels=node_lab, font_size=16)
            nx.draw_networkx_edges(G, pos)

            plt.title(measure_name)
            plt.colorbar(nodes)
            plt.axis('off')
            plt.show()


        A = nx.to_numpy_matrix(self.graph)

        bc = nx.betweenness_centrality(self.graph, weight='weight')
        dc = nx.degree_centrality(self.graph)
        idc = nx.in_degree_centrality(self.graph)
        odc = nx.out_degree_centrality(self.graph)
        ec = nx.eigenvector_centrality(self.graph)
        
        draw(self.graph, pos, bc, 'betweeness centrality')
        draw(self.graph, pos, dc, 'degree centrality')
        draw(self.graph, pos, idc, 'in degree centrality')
        draw(self.graph, pos, odc, 'out degree centrality')
        draw(self.graph, pos, ec, 'eigenvector centrality')


        return(A, [self.graph.node[i]['word'] for i in self.graph.nodes()], bc, dc, idc, odc)


if __name__ == "__main__":

    #port = 9004 for french, 9000 for english

    #exemple of the time on a complicated sentence
    #import time
    #start = time.time()
    
    mydep = Dependencies("Un hypothétique prolongement de la ligne 7 du métro de Paris est possible au nord,  jusqu ' au Musée de l'air et de l'espace au Bourget et profiterait aux Drancéens puisqu'une station serait prévue à la gare du Bourget, située aux abords de Drancy",
                        fr_lemmatize=False, insecables = ["Musée de l'air et de l'espace", "Drancy"])

    print(mydep.shortest_path("Musée de l'air et de l'espace", "Drancy", 0, 0))
    #mydep.plot_tree_layout()

    #end = time.time()
    #print(end - start)
