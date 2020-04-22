import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

import re
import sys
from tqdm import tqdm 

from nltk.parse.corenlp import CoreNLPDependencyParser

class DependenciesLCA():

    def __init__(self, sentence, port=9004):
        self.sentence = sentence.rstrip('.')
        self.sentence = re.sub(r'(.?)([\.,;:\?!()\[\]\{\}«»\'\"\-\—\/’&])', '\\1 \\2 ', self.sentence)

        self.corenlpparser = CoreNLPDependencyParser(url='http://localhost:'+ str(port))
        parse =  self.corenlpparser.raw_parse(self.sentence)
        self.tree = next(parse)

    def lca(self, index1, index2):
        path1 = []
        path2 = []
        path1.append(index1)
        path2.append(index2)

        node = index1
        while(node != self.tree.root):
            node = self.tree.nodes[node['head']]
            path1.append(node)

        node = index2
        while(node != self.tree.root):
            node = self.tree.nodes[node['head']]
            path2.append(node)
        
        for l1, l2 in zip(path1[::-1],path2[::-1]):
            if(l1==l2):
                temp = l1
        return temp

    def path_lca(self, node, lca_node):
        path = []
        path.append(node)
        while(node != lca_node):
            node = self.tree.nodes[node['head']]
            path.append(node)
        return path

    def branch_paths(self, ent1, ent2):

        entity1 = re.split(r"[ .',\-0-9]", ent1)[-1]
        entity2 = re.split(r"[ .',\-0-9]", ent2)[-1]

        node1=None
        node2=None
        for node in self.tree.nodes:
            if (self.tree.nodes[node]["word"] == entity1) & (node1==None):
                node1 = self.tree.nodes[node]
            elif (self.tree.nodes[node]["word"] == entity2) & (node2==None):
                node2 = self.tree.nodes[node]

        try :
            if node1['address']!=None and node2['address']!=None:
                lca_node = self.lca(node1, node2)
                path1 = self.path_lca(node1, lca_node)
                path2 = self.path_lca(node2, lca_node)

                word_path1 = "/".join([p["word"] for p in path1])
                word_path2 = "/".join([p["word"] for p in path2])
                rel_path1 = "/".join([p["rel"] for p in path1])
                rel_path2 = "/".join([p["rel"] for p in path2])
                pos_path1 = "/".join([p["tag"] for p in path1])
                pos_path2 = "/".join([p["tag"] for p in path2])
            else:
                print(entity1, entity2, self.sentence)
        except AssertionError :
            print("Node none, Entity 1 :", node1, entity1, ent1, " /  Entity2 :", node2, entity2, ent2, " /  Phrase :", self.sentence)
        except:
            if (bool(re.search(r'\d', entity1)) == True) | (bool(re.search(r'\d', entity1)) == False):
                return (None,None,None,None,None,None)
            print("Node none, Entity 1 :", node1, entity1, ent1, " /  Entity2 :", node2, entity2, ent2, " /  Phrase :", self.sentence, "  / Tree : ", self.tree)

        return(word_path1, word_path2, rel_path1, rel_path2, pos_path1, pos_path2)

if __name__ == "__main__":

    dep= DependenciesLCA("L\'actrice Ovidie, le dessinateur Manu Larcenet, ainsi que Florent Grospart qui deviendra adjoint au maire Vert de Vendôme et dirigeant d\'Attac, ont fait partie du SCALP.")
    
    print(dep.branch_paths("Vendôme", "Florent Grospart"))
