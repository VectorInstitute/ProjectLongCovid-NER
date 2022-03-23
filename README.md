# LongCOVID_Insights
## Towards Providing Clinical Insights on Long Covid from Twitter Data 

From the outset of the COVID-19 pandemic social media has
provided a platform for sharing and discussing experiences in
real time. This rich source of information may also prove useful to researchers for uncovering evolving insights into post-acute sequelae of SARS-CoV-2 (PACS), commonly referred to as Long COVID. In order to leverage social media data, we
propose using entity-extraction methods for providing clini cal insights prior to defining subsequent downstream tasks.
In this work, we address the gap between state-of-the-art entity recognition models, and the extraction of clinically relevant entities which may be useful to provide explanations for gaining relevant insights from Twitter data. We then propose
an approach to bridge the gap by utilizing existing configurable tools, and datasets to enhance the capabilities of these
models. 

# Quick Links

For details about our experimental setup, please read our paper titled: Towards Providing Clinical Insights on Long Covid from Twitter Data

### SETUP

### Data creation
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
### Entity Extraction

### Enitity Normalization

### MetaMapLite 


# Citation

Please consider citing our work if you found it useful in your research:

##### For any queries about this work, please contact: Rohan Bhambhoria at <r.bhambhoria@queensu.ca>

# Acknowledgements
