import spacy

nlp = spacy.load("training/model-best")

config = {
   "phrase_matcher_attr": None,
   "validate": True,
   "overwrite_ents": True,
   "ent_id_sep": "||",
}

ruler = nlp.add_pipe("entity_ruler", config=config).from_disk("./assets/accession_label_patterns.jsonl")

text = "GenBank, genbank, EMBL, ebi, DDBJ, ncbi, NCBI"
doc = nlp(text)

print([(ent.text, ent.label_) for ent in doc.ents])

# this will need to be created at the time of the preprocess
nlp.to_disk("accession_model")