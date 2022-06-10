import pandas as pd
import spacy

def text_labels(path,model,dest):

    nlp = spacy.load(model)
    df = pd.read_csv(path,usecols=['text','id'])

    tup = list(df.itertuples(index=False, name='meta'))

    entities = []
    for doc, context in list(nlp.pipe(tup, n_process = 8, batch_size = 1000, as_tuples=True)):
        entities.extend({
            "id":context, 
            "entity_id":ent.ent_id_,
            "entity_text":ent.text, 
            "entity_label":ent.label_, 
            "entity_start":ent.start_char, 
            "entity_end":ent.end_char,
            "entity_id_no":ent.ent_id 
            } for ent in doc.ents)
    entities_df = pd.DataFrame.from_records(entities)
    #return entities_df.to_csv(f"./{str(dest)}.csv")
    return entities_df
