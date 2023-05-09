# LongCOVID_Insights
This repository includes a set of tools and models used in the NLP pipeline for automatic extraction and encoding of data from social media to fascilitate access and retreival of information. and Reddit are used here as independent social media sources. The NLP pipeline for automated extraction and encoding of information involves Data Collection and Preprocessing, Long Covid Filter (only for Twitter data), Entity Extraction, and Normalization components.


The Vector Institute’s Industry Innovation and AI Engineering teams worked with Roche Canada, Deloitte and TELUS to explore this question. Their project: applying natural language processing (NLP) techniques to social media posts made by people with long COVID to see if any patterns arise. This involved creating a machine learning pipeline and testing the capabilities of various NLP models on first-hand testimony from social media. The hope is that these patterns, if identified, could reveal clues about when and how frequently symptoms arise and where clusters of the condition occur. Any insights could be shared with clinicians to hone their research questions, identify trends early, or inform treatment strategies. 


# Quick Links

To learn more about this work please read our <a href="https://vectorinstitute.ai/2022/02/11/using-ai-to-help-solve-the-long-covid-puzzle/">blog post</a>. 
For details about our experimental setup, please read our paper titled: <a href="http://w3phiai2022.w3phi.com/index.html#">Towards Providing Clinical Insights on Long Covid from Twitter Data</a>. 

## Interactive Map
Here is a Kepler <a href="https://drive.google.com/file/d/1l8syhg5kb4SkGakxHXdrCfx51OAYD3sI/view?usp=sharing"> html </a>. 


## Data creation
We provide a suite of scripts to download and create twitter and reddit data along with a list of hashtags, etc. The process consists of three steps. 

1. Navigate to `SocialMediaData_creation/`.
2. There is an example config file used to highlight how to configure your run. Setup
    your config file with the various include, excludes, hashtags, etc.
3. Run `SocialMediaData_creation/main.py` and pass the necessary `key_file`, `credential_key`,
   `config_file`, and `output_file` parameters with `-k`, `-kc`, `-c`, and `-o`
   respectively, via the command line.

A sample command using the config under ./sample
```
python main.py -k sample/api_keys.yaml -ck v2_key -c sample/config_run.ini
```
## Information Extraction Pipeline
### Entity Extraction

Our UMLSBERT model which is finetuned on the [n2c2 (2010)](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) challenge for entity extraction is [available on huggingface](https://huggingface.co/RohanVB/umlsbert_ner).

### Enitity Normalization
The Nomralization steps are shown in the Colan Notebooks [here](https://github.com/VectorInstitute/ProjectLongCovid-NER/tree/main/Notebooks)
The Unique Concepts for the Entity mapping and nomralization can be found [here](https://docs.google.com/spreadsheets/d/1Y1Y4_uauW3c4Pxhjarz3puKkts2BXQh9K87qkUuaExA/edit#gid=1430806207)
The list of extracted entities before mapping can be found [here](https://docs.google.com/spreadsheets/d/1p_Ut-GlQghC8v_rhXURGp5lY8l-oXwHhVSBpr8m2S-w/edit#gid=1844583485)

### MetaMapLite 
MetaMapLite is used for obtaining CUI codes used to label entities, and can be installed [here](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/run-locally/MetaMapLite.html).


# Citation

Please consider citing our work if you found it useful in your research:

##### For any queries about this work, please contact: Rohan Bhambhoria at <r.bhambhoria@queensu.ca>

# Acknowledgements
