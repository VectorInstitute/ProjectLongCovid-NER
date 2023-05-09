import json
import re
from pathlib import Path
import random
import itertools

from pigeon import annotate

import pandas as pd
import numpy as np
from sklearn import metrics

import stanza
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForTokenClassification
import torch
from tqdm.notebook import tqdm

class LH():
    def __init__(self, path_to_data):
        """
        Load json file containing Tweet text.
        """
        with open(path_to_data) as sample_data:
            self.json_dict = json.load(sample_data)
            

    def get_all_text(self): 
        """
        Most tweets appear as a dict object, with the 'text' key storing a str value.
        However, sometimes the 'text' key stores a list value instead.
        Here, we parse both cases.
        """
        all_text = []
#         all_geo = []
        for idx, obj in tqdm(enumerate(self.json_dict)):
            if type(obj) == list:
                return obj
                nested_text = []
                for item in obj:
                    nested_text.append(item['text'])
                all_text.extend([nested_text])
            else:
                all_text.append([obj['text']])
            
            user = obj['user']
            timestamp = obj['created_at']
            if 'location' in user.keys():
                all_text[-1].append(user['location'])
            else:
                all_text[-1].append('')
            all_text[-1].append(timestamp)
        return all_text
    
    
    def clean_data(self, input_texts):
        """
        Clean text, including:
        - Removing URLs
        - Removing contractions
        - Removing special characters
        """
        clean_texts = []
        for item in tqdm(input_texts):
            item[0] = re.sub(r"https?://\S+", "", item[0])

#             item[0] = contractions.fix(item[0])

            item[0] = re.sub("@\S*", "", item[0]) # Remove mentions
            item[0] = re.sub("\n", " ", item[0]) # Remove new line characters
            item[0] = re.sub("&amp;", "&", item[0]) # Replace "&amp" with "&"
            item[0] = item[0].strip() # Remove whitespace at either end

            clean_texts.append(item)

        return clean_texts


    def filter_for_unique_text(self, input_texts):
        """
        To have predictable ordering of unique texts,
        loop through list explicitly and test against
        temporary set.
        """
        temp_lookup = set()
        unique_text = []
        for i, item in enumerate(input_texts):
            text = item[0]
            if text not in temp_lookup and temp_lookup.add(text) is None:
                unique_text.append(item)
        #unique_text = [text for text in input_texts if text not in temp_lookup and temp_lookup.add(text) is None]
        return unique_text
    

    def filter_for_self_reports_with_regex(self, input_texts, regexp_list = []):  
        """
        Apply custom regex expressions to each text,
        store match results in dataframe.
        updated to add timestamp
        """
        self_reports = []
        regex_matched = []
        geos = []
        dates = []
        if len(regexp_list) == 0:
            print('Please provide a list of regular expressions.')
            return []

        for item in tqdm(input_texts):
            text = item[0]
            geo = item[1]
            try:
                timestamp = item[2]
            except:
                timestamp = 'NA'
            
            self_reports.append(text)
            geos.append(geo)
            dates.append(timestamp)
            
            these_matches = []
            for r in regexp_list:
                regexp = re.compile(r)
                if regexp.search(text):
                    these_matches.append(r)
                
            if len(these_matches) > 0:
                regex_matched.append(', '.join(these_matches))
            else:
                regex_matched.append('N/A')
        
        matches = pd.DataFrame({'report_text':self_reports, 'regex_matched':regex_matched, 'geo':geos, 'timestamp':dates})
        return matches

    def filter_for_self_reports_with_mnli(self, input_texts,
                                          candidate_labels = [],
                                          hypothesis_template ='Sharing {}.'):
        results = []
        classifier = pipeline("zero-shot-classification", model="digitalepidemiologylab/covid-twitter-bert-v2-mnli", device=0)
        for t in tqdm(input_texts, total=len(input_texts)):
            res = classifier(t[:511],
                                 candidate_labels,
                                 hypothesis_template=hypothesis_template,
                                 multi_label=True)
            results.append(candidate_labels.index(res['labels'][0]))  
        return results

    def filter_for_symptom_reports_with_stanza(self, input_text, num_texts_to_process = -1):
        """
        Apply pre-trained Stanza NER model to each text,
        store match results in dataframe.
        """
        if num_texts_to_process == -1:
            loop_range = len(input_text)
        else:
            loop_range = num_texts_to_process
        print(f"Applying filter to first {loop_range} texts.")

        symptom_reports = []
        entities = []
        entity_types = []
        for i in range(loop_range):
            doc = nlp(input_text[i])

            symptom_reports.append(input_text[i])
            if len(doc.entities) > 0:
                these_entities = []
                these_entity_types = []
                for ent in doc.entities:
                    these_entities.append(ent.text)
                    these_entity_types.append(ent.type)
                entities.append(', '.join(these_entities))
                entity_types.append(', '.join(these_entity_types))
            else:
                entities.append('N/A')
                entity_types.append('N/A')

        df = pd.DataFrame(list(zip(symptom_reports, entities, entity_types)), \
                              columns = ['symptom_reps', 'entities', 'entity_types'])
        return df
    
    def clean(self,text):
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        return text.lower()
    

#     def normalize_symptoms(self, df, mapping_dict, remove_list):
#         df['replaced'] = False
#         symptom_lists = []
#         replaced_list= []
#         for s_list in tqdm(df['symptoms'].tolist()):
#             symptom_row = []
#             for s in eval(s_list):
#                 s = self.clean(s)
#                 if s not in remove_list:
#     #                 print(s)
#     #                 print(mapping_dict[s])
#                     try: 
#                         symptom_row.append(mapping_dict[s].replace(' ','_'))
#                         if not mapping_dict[s] == s :
#                             replaced = True
#                     except:
#                          symptom_row.append(s.replace(' ','_'))
#                 else:
#                     replaced.append('False')

#             symptom_lists.append(list(set(symptom_row)))

#         df['normalized_symptom_list'] = symptom_lists
#         df['normalized_symptom_str'] = [' '.join(list(set(l))) for l in df['normalized_symptom_list'].tolist()]            
#         df['replaced'] = replaced_list
#         flattened = [s for l in df['normalized_symptom_list'].tolist() for s in l]
#         print('# unique symptoms', len(set(flattened)))
#         return df, flattened
    def normalize_symptoms(self, df, mapping_dict, remove_list):
        df['replaced'] = False
        replaced_list= []
        symptom_lists = []
        for s_list in tqdm(df['symptoms'].tolist()):
            symptom_row = []
            replaced = False
            for s in eval(s_list):
                s = self.clean(s)
                if s not in remove_list:
    #                 print(s)
    #                 print(mapping_dict[s])
                    try: 
                        symptom_row.append(mapping_dict[s].replace(' ','_'))
                        if not mapping_dict[s] == s :
                            replaced = True
                    except:
                        symptom_row.append(s.replace(' ','_'))

    #             else:
    #                 replaced.append('False')

            symptom_lists.append(list(set(symptom_row)))
            replaced_list.append(replaced)

        df['normalized_symptom_list'] = symptom_lists
        df['normalized_symptom_str'] = [' '.join(list(set(l))) for l in df['normalized_symptom_list'].tolist()]
        df['replaced'] = replaced_list

        flattened = [s for l in df['normalized_symptom_list'].tolist() for s in l]
        print('# unique symptoms', len(set(flattened)))
        return df, flattened
    
    def normalize_treatments(self,df, mapping_dict, remove_list):
        treatment_lists = []
        for t_list in df['treatments'].tolist():
            treatment_row = []
            for t in eval(t_list):
                t = self.clean(t)
                if t not in remove_list:
                    try: 
                        treatment_row.append(mapping_dict[t].replace(' ','_'))
                    except:
                         treatment_row.append(t.replace(' ','_'))

            treatment_lists.append(list(set(treatment_row)))

        df['normalized_treatment_list'] = treatment_lists
        df['normalized_treatment_str'] = [' '.join(list(set(l))) for l in df['normalized_treatment_list'].tolist()]            

        flattened = set([s for l in df['normalized_treatment_list'].tolist() for s in l])
        print('# unique treatments', len(flattened))
        return df, flattened

    def filter_for_symptom_reports_with_zero_shot(self, input_texts, labels_and_thresh = {}, hypothesis = "This is {}."):
        """
        Apply pre-trained zero-shot model to each text,
        store results in a dictionary.
        """
        if len(labels_and_thresh) == 0:
            print('Please provide a dict of labels and thresholds for the zero-shot classifier.')
            return self_reports

        results = {}
        scores = []
        preds = []

        for i, input_text in enumerate(input_texts):
            this_result = zero_shot_classifier(input_text,\
                                               list(labels_and_thresh.keys()),\
                                               hypothesis_template = hypothesis, multi_label = True)
            label_confidence = list(zip(this_result['labels'], this_result['scores']))
            above_thresh = []
            for j, this_label in enumerate(this_result['labels']):
                if this_result['scores'][j] > labels_and_thresh[this_label]:
                    above_thresh.append(True)
                else:
                    above_thresh.append(False)
            scores.append(this_result['scores'])
            preds.append(int(all(above_thresh)))

        results['labels_and_thresh'] = labels_and_thresh
        results['scores'] = scores
        results['preds'] = preds

        return results
    
    def extract_symptoms_treatments_with_stanza(self, input_all, num_texts_to_process = -1, 
                                                save_checkpoints = True, check_point_path = 'temp_results/'):
        """
        Apply pre-trained Stanza NER model to each text,
        store match results in dataframe.
        creates separate column for treatment and symptoms for easy post processing
        This also extracts time aware information
        Use save_checkpoints arg to save intermediate results
        """
        try:
            print(nlp)
        except:
            # load models
            stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
            nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, use_gpu = True)
            
        if num_texts_to_process == -1:
            loop_range = len(input_all)
        else:
            loop_range = num_texts_to_process
        print(f" Applying stanza to  {loop_range} texts.")

        symptom_reports = []
        symptoms_list = []
        treatments_list = []
        geos_list = []
        dates_list = []
        for i in tqdm(range(loop_range)):
            doc = nlp(input_all[i][0])
            dates_list.append(input_all[i][3]) # take care of date because it is always present
            geos_list.append(input_all[i][2])
            symptom_reports.append(input_all[i][0])
 
            if len(doc.entities) > 0:
                symptoms = []
                treatments = []
                for ent in doc.entities:
                    if ent.type == 'PROBLEM':
                        symptom = ent.text
                        symptoms.append(symptom)
    
                    elif ent.type == 'TREATMENT':
                        treatment = ent.text
                        treatments.append(treatment)
                if symptoms:
                    symptoms_list.append(symptoms)
                else:
                    symptoms_list.append(None)
                if treatments:    
                    treatments_list.append(treatments)
                else:
                    treatments_list.append(None)
            else:
                symptoms_list.append(None)
                treatments_list.append(None)
            # save intermediate results
            if save_checkpoints:
                if (i % 10000 == 0)  & (i !=0):
                    df = pd.DataFrame(list(zip(symptom_reports, symptoms_list, treatments_list, dates_list)), \
                                      columns = ['symptom_reps', 'symptoms', 'treatments', 'date'])
                    df.to_csv(check_point_path + 'checkpoint_pr_count_' + str(i) + '_.csv')

        df = pd.DataFrame(list(zip(symptom_reports, symptoms_list, treatments_list, dates_list)), \
                              columns = ['symptom_reps', 'symptoms', 'treatments', 'date'])
        df.to_csv(check_point_path + 'checkpoint_pr_final.csv')
        return df
    
    def extract_symptoms_treatments_with_new_model(self, input_all, modelname = 'ctbert' , num_texts_to_process = -1, 
                                                save_checkpoints = True, check_point_path = 'temp_results/'):
        """
        Apply fine-tuned models to each text,
        store results in dataframe.
        This also extracts time aware information
        Use save_checkpoints arg to save intermediate results
        Extracts location as well
        """


        if num_texts_to_process == -1:
            loop_range = len(input_all)
        else:
            loop_range = num_texts_to_process
        if modelname == 'ctbert':
            model_path =  '/ssd003/projects/nlpcovidshare/checkpoints/ctbert_finetuned/tweetbert_aug1/'
        elif modelname == 'umlsbert':
            model_path = '/ssd003/projects/nlpcovidshare/checkpoints/umlsbert_finetuned/'
        else:
            raise ('model not identified')
        print('loading model.....')    
        # load fintuned model 
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)

        ner_extract = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True, device=0) # gpu enabled with device = 0
        print(f" Model loaded, applying fine tuned model {modelname} to  {loop_range} texts.")
        symptom_reports = []
        symptoms_list = []
        treatments_list = []
        geos_list = []
        dates_list = []
        for i in tqdm(range(loop_range)):
            doc = ner_extract(input_all[i][0])
            dates_list.append(input_all[i][3]) # take care of date because it is always present
            geos_list.append(input_all[i][2])
            symptom_reports.append(input_all[i][0])
            ner_tokens = [i['word'] for i in doc if i['entity_group'] == 'problem']
            # Merge the partial tokens
            symptoms = []
            for item in ner_tokens:
                word = item
                try:
                    if word.startswith('##'):
                        word = symptoms[len(symptoms)-1] + word.replace('##','')
                        symptoms.pop()
                except IndexError:
                    word = item
                symptoms.append(word)
            if symptoms:
                symptoms_list.append(symptoms)
            else:
                symptoms_list.append(None)
            # save intermediate results
            if save_checkpoints:
                        if (i % 10000 == 0)  & (i !=0):
                            df = pd.DataFrame(list(zip(symptom_reports, symptoms_list,  dates_list, geos_list)), \
                                              columns = ['symptom_reps', 'symptoms', 'date','geo'])
                            df.to_csv(check_point_path + 'checkpoint_new_model_'+modelname+'_count_' + str(i) + '_.csv')
            df = pd.DataFrame(list(zip(symptom_reports, symptoms_list, dates_list,geos_list)), \
                                      columns = ['symptom_reps', 'symptoms', 'date','geo'])
            df.to_csv(check_point_path + 'checkpoint_new_model_'+modelname+'_final.csv')
        return df
   
 

    

class LHdataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)