import requests
import json
import copy
import os
import sys
sys.path.insert(1, '../mm-locate-news')
from utils import *

class wikifier():
    def __init__(self):
        pass

    def annotate(self, ner_outputs, text0, language, config):

        langin = config['wikifier']['lang'][language]
        params_in = config['wikifier']['params']
        user_key = config['wikifier']['user_key']
        API_ENDPOINT = config['wikifier']['api_url']
        text = text0.replace('.',' ')

        self.params = {
            'userKey': user_key,
            "text": text,
            "lang": langin,
            "secondaryAnnotLanguage": langin,
            "extraVocabularies": params_in['extraVocabularies'],
            "wikiDataClasses": params_in['wikiDataClasses'],
            "wikiDataClassIds": params_in['wikiDataClassIds'],
            "support": params_in['support'],
            "ranges": params_in['ranges'],
            "includeCosines": params_in['includeCosines'],
            "maxMentionEntropy": params_in['maxMentionEntropy'],
            "maxTargetsPerMention": params_in['maxTargetsPerMention'],
            "minLinkFrequency": params_in['minLinkFrequency'],
            "pageRankSqThreshold": params_in['pageRankSqThreshold'],
            "applyPageRankSqThreshold": params_in['applyPageRankSqThreshold'],
            "partsOfSpeech": params_in['partsOfSpeech'],
            "verbs": params_in['verbs'],
            "nTopDfValuesToIgnore": params_in['nTopDfValuesToIgnore']
        }
        out_put = ["error"]
        try:
          out0 = requests.get(API_ENDPOINT, params=self.params)
          out = json.loads(out0.text)
          out_put = self.wrapper(text, out, ner_outputs, ["ORG","LOC","FAC"])
        except Exception as e:
          print(f"error occured [{e}] ...")

        return out_put


    def wrapper(self,doc, named_links_in, ner_outputs, limiting_labels):

        annotations = []

        for ner in ner_outputs:
            if limiting_labels !=[]:
                if ner['label'] not in limiting_labels:
                    continue
            temp = []
            pgrmk = 0
            annotation_main = ""

            for nl in named_links_in['annotations']:

                url = nl['url']
                if url[:29] == "http://en.wikipedia.org/wiki/":  # convert wikipedia to dbpedia
                    url = "http://dbpedia.org/resource/" + url[29:]

                for s in nl['support']:
                    text = doc[s['chFrom']:s['chTo']+1]
                    dic_annotation = {'uri': url, 'text': text, 'page_rank': s['pageRank'], 'start_char': s['chFrom'], 'end_char': s['chTo']+1}

                    if ner['start_char'] - 2 <= dic_annotation['start_char'] <= ner['end_char'] + 2:
                        if ner['start_char'] - 2 <= dic_annotation['end_char'] <= ner['end_char'] + 2:
                            dic_annotation['label'] = ner['label']
                            
                            if pgrmk < dic_annotation['page_rank']:  # keep track of the link with highest page rank
                                annotation_main = dic_annotation
                                pgrmk = dic_annotation['page_rank']
                
            if annotation_main != "":   
                annotations.append(annotation_main)


        return annotations

