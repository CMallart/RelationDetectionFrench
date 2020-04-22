"""
Take the features, getting the corresponding entities, and seach their link in wikidata
"""
import sqlite3
import pandas as pd
import sys
import requests
import time

class LabelPairs:

	def __init__(self, dbfile = "/home/cyrielle/Codes/clean_code/DataStorage/small_wiki.db", first_time=False ):
		self.dbfile = dbfile
		self.first_time = first_time

	def format_query(self,list_tuples):
		start = "SELECT ?e1 ?e2 ?p  ?l ?id WHERE {"
		end = "}"
		middle = ["{ values ?id {%s} values ?e1 {wd:%s} values ?e2 {wd:%s}  ?e1 ?p ?e2 . ?property ?ref ?p . ?property rdf:type wikibase:Property .\
			 ?property rdfs:label ?l FILTER (lang(?l) = \"fr\")} " %(tup) for tup in list_tuples]
		middle_union = " UNION ".join(middle)
		return(start + middle_union + end)

		#SELECT ?item WHERE {
  #?item rdfs:label "Saint-Simon"@fr.
  #minus {?item wdt:P31 wd:Q4167410 .}
  
}

	def json_to_tuples(self, data):
		results = []
		for item in data['results']['bindings']:
			wiki_id_rel = item['p']['value'] if data['results']['bindings'] else ''
			name_rel = item['l']['value'] if data['results']['bindings'] else ''
			id_features =  item['id']['value'] if data['results']['bindings'] else ''
			results.append((wiki_id_rel,name_rel,id_features))
		return(results)

	def query_links(self, df, url = 'https://query.wikidata.org/sparql'):
		query = self.format_query(list(df.itertuples(index=False)))
		headers = {
            'User-Agent': 'EntityLinkingBot/0.1 (cyrielle.mallart@irisa.fr) python-requests/entity_linking_using_wikidata_knowledge_base/wikipedia_page_to_wikidata/query_list_articles/0.5'
        }
		r = requests.get(url, params = {'format': 'json', 'query': query}, headers = headers)
		#print(r.headers)
		if 'Retry-After' in r.headers:
			sys.exit(1)
			print("403, Retry-after", r.content, r.headers)

		data = r.json()
		vals = self.json_to_tuples(data)
		return(vals)


	def execute(self):
		conn = sqlite3.connect(self.dbfile, isolation_level=None)
		cursor = conn.cursor()
		if self.first_time:
			cursor.execute("ALTER TABLE entities_pairs ADD COLUMN relation_id")
			cursor.execute("ALTER TABLE entities_pairs ADD COLUMN relation_name")

		t=0
		print("Starting")

		for df in pd.read_sql_query("select rowid, entity1_wikidata_id, entity2_wikidata_id FROM entities_pairs", conn, chunksize=20):
			df["entity1_wikidata_id"] = df["entity1_wikidata_id"].apply(lambda x: x.split("/")[-1] if x else '')
			df["entity2_wikidata_id"] = df["entity2_wikidata_id"].apply(lambda x: x.split("/")[-1] if x else '')
			#print(df[["e1", "e2", "rowid"]])
			links = self.query_links(df)
			time.sleep(10)
			#print(links)

			cursor.executemany("UPDATE entities_pairs SET relation_id = ?, relation_name = ? WHERE rowid = ?", links)
			t+=20
			if t % 1000 == 0 : print("Labelled ", t, " pairs")

if __name__ == "__main__":
  LabelPairs(first_time=False).execute()