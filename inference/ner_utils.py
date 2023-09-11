import spacy
import sys
sys.path.insert(1, '/data/1/mmm_test/mm-locate-news')
from utils import *
# python -m spacy download 'fr_core_news_sm', 'it_core_news_sm' , ...
# https://spacy.io/api/annotation

def convert_tag( tag_in, tag_map):
    converted_tag = tag_in
    tag = [tm for tm in tag_map if tag_in in tag_map[tm]]
    if tag != []:
        converted_tag = tag[0]

    return converted_tag

class ner_spacy():

    def __init__(self, language ):
        
        self.tag_map = {
        "PERSON": ["person","Person","PER","per","Per"],
        "NORP": ["Nationality","nationality","religion","Religion"],
        "FAC": ["faculty","Faculty","FACULTY","building", "airport", "highway", "bridge"],
        "ORG": ["ORGANIZATION","Org","Organization","organization"],
        "LOC": ["GPE","location","Location","LOCATION","river","mountain"],
        "PRODUCT": ["Object","OBJ","OBJECT","object", "vehicle", "food"],
        "EVENT": ["event","Event","hurricane", "battle", "war", "sport"],
        "WORK_OF_ART": ["book","song"],
        "LAW": ["Law","law"],
        "LANGUAGE": ["lang","Language","language"],
        "DATE": ["date","Date"],
        "TIME": ["time","Time"],
        "PERCENT": ["Percent","percent"],
        "MONEY": ["money","Money"],
        "QUALITY": ["Quality","quality","weight","distance"],
        "ORDINAL": ["Ordinal","ordinal"],
        "CARDINAL": ["Cardinal","cardinal","NUMBER"]
        }

        self.has_model = True
        model = None
        supported_languages = {"eng":"eng", "deu":"deu", "fra":"fra", "por":"por", "ita":"ita"}.keys() 
        if language in supported_languages:
            model = "en_core_web_sm"

        if model != None:
            self._nlp = spacy.load(model)
        else:
            self.has_model = False

    def annotate(self, text):
        doc = self._nlp(text)
        entities = doc.ents
        output = self.wrapper(entities)
        return output



    def wrapper(self, annotations_in):
        annotations = []

        for ent in annotations_in:
            annotations.append({
                'text': ent.text,
                'label': convert_tag(ent.label_, self.tag_map),
                'start_char': ent.start_char,
                'end_char': ent.end_char,
            })
        return annotations


