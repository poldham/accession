# When adding sector labels with an existing biospacy model
# ensure that overwrite_ents is false or it will overwrite the 
# taxa and related labels. 

import spacy

nlp = spacy.load("training/model-best")

config = {
   "phrase_matcher_attr": None,
   "validate": True,
   "overwrite_ents": True,
   "ent_id_sep": "||",
}

ruler = nlp.add_pipe("entity_ruler", config=config).from_disk("./assets/accession_patterns_short.jsonl")

text = "Genbank, ncbi, ddbj, embl, ebi, ena, SEQ ID NO: 1, SEQ NO 1"
doc = nlp(text)

print([(ent.text, ent.label_) for ent in doc.ents])

# this will need to be created at the time of the preprocess
nlp.to_disk("models/entity_ruler/")
