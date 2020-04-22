## Contents

+ `data` : our dataset, as well as the feature files to feed our models
+ `importdata` : ETL pipeline to get Wikipedia data in SQL database, and files creating the features. Does not work *as is* on new data, as the entity extractor is proprietory and no alternative has been implemented so far.
+ `models` : our models for comparing benchmark and our approach of detection before classification

## Data

Dataset before any subsampling or balancing is in `relationdetection/data/wikidata_relations.db`.
It is a large SQL database file, containing the following tables and fields :
+ articles : url, id, title, text, language, cats
+ entities : start, end, surface_form, wiki_id, entity_type, id_article, id_sentence, wikidata_id
+ entities_pairs : entity1_id, entity1_surface_form, entity1_wikidata_id, entity2_id, entity2_surface_form, entity2_wikidata_id, id_sentence, relation_id, relation_name
+ sentences : id_article, start_in_article, end_in_article, text, follows
