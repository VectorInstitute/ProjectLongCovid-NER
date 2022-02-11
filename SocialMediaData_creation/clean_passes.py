import hashlib
import preprocessor as tweet_proc
import string
import re


tweet_proc.set_options(tweet_proc.OPT.URL, tweet_proc.OPT.EMOJI,
                       tweet_proc.OPT.ESCAPE_CHAR)

MENTIONS = re.compile(r'@\w*')

def only_ids(entry):
    t_id = entry["id"]

    return {"tweet_id" : t_id}

def remove_unwanted_chars(entry):
    """
    Helper function to recursively encode entry in dict
    """

    if isinstance(entry, dict):
        return {key: self._clean(value) for key, value in entry.items()}
    elif isinstance(entry, list):
        return [self._clean(element) for element in entry]
    elif isinstance(entry, unicode):

        # Clean entry
        entry = tweet_proc.clean(entry)

        # Remove punctuation
        for char in string.punctuation:
            entry = entry.replace(char, " ")

        # Remove non-ascii
        entry = entry.encode("ascii", "ignore").decode()

        return entry
    else:
        return entry

def re_hash(matchobj):
    user = matchobj.group(0)
    assert user[0] == "@"
    user = user[1:]
    user_hash = hashlib.sha224(
                    user.encode()).hexdigest()
    return "@" + user_hash

def remove_unused_fields(entry):

    if "entities" in entry:
        if "mentions" in entry["entities"]:
            del entry["entities"]["mentions"]

    if "author_id" in entry:
        del entry["author_id"]

    if "id" in entry:
        del entry["id"]

    if "user" in entry:
        if "name" in entry["user"]:
            del entry["user"]["name"]
        if "id" in entry["user"]:
            del entry["user"]["id"]
        if "profile_image_url" in entry["user"]:
            del entry["user"]["profile_image_url"]

    return entry



def hash_ids(entry):

    if "user" in entry:
        if "username" in entry["user"]:
            entry["user"]["username"] = hashlib.sha224(
                entry["user"]["username"].encode()).hexdigest()

        if "description" in entry["user"]:
            new_text = re.sub(MENTIONS, re_hash, entry["user"]["description"])
            entry["user"]["description"] = new_text

    if "text" in entry:
        new_text = re.sub(MENTIONS, re_hash, entry["text"])
        entry["text"] = new_text


    return entry



roche_pharma = ['roche', 'alecensa', 'alectinib', 'activase', 'alteplase', 'cotellic',
               'cobimetinib', 'hemlibra', 'emicizumab', 'glucophage', 'metformin',
               'xolair', 'omalizumab', 'esbriet', 'pirfenidone', 'enspryng',
               'satralizumab', 'bactrim', 'sulfamethoxazole', 'quaifenesin', 'trimethoprim',
               'nutropin', 'somatropin', 'herceptin', 'trastuzumab', 'zelboraf',
               'vemurafenib', 'tecentriq', 'atezolizumab', 'xofluza', 'baloxavir marboxil',
               'madopar', 'benserazide', 'levodopa', 'avastin', 'bevacizumab',
               'lexotan', 'bromazepam', 'rocaltrol', 'calcitriol', 'xeloda',
               'capecitabine', 'dilatrend', 'carvedilol', 'rocephin',
               'ceftriaxone', 'inhibace', 'cilazapril', 'rivotril',
               'clonazepam', 'valium', 'diazepam', 'pulmozyme',
               'dornase', 'fuzeon', 'enfuvirtide', 'rozlytrek', 'entrectinib',
               'tarceva', 'erlotinib', 'neorecormon', 'erythropoietin',
               'anexate', 'flumazenil', 'cymevene', 'ganciclovir', 'kytril',
               'granisetron', 'bonviva', 'ibandronic', 'roferon', 'interferon',
               'ro0228181', 'roaccutane', 'isotretinoin', 'lariam', 'mefloquine',
               'mircera', 'methoxy polyethylene glycol-epoetin beta', 'epoetin',
               'dormicum', 'midazolam', 'cellcept', 'mycophenolate mofetil',
               'gazyva', 'obinutuzumab', 'ocrevus', 'ocrelizumab', 'xenical',
               'orlistat', 'tamiflu', 'oseltamivir', 'pegasys', 'peginterferon',
               'perjeta','pertuzumab', 'phesgo', 'pertuzumab', 'trastuzumab',
               'konakion', 'phytomenadione', 'polivy', 'polatuzumab', 'ranibizumab',
               'ranibizumab','copegus', 'ribavirin', 'evrysdi', 'risdiplam',
               'mabthera', 'rituximab', 'invirase', 'saquinavir', 'bactrim',
               'sulfamethoxazole', 'ro0062580','trimethoprim', 'tnkase',
               'tenecteplase', 'actemra', 'tocilizumab', 'kadcyla', 'trastuzumab',
               'valcyte', 'valganciclovir', 'erivedge', 'vismodegib']

roche_diagnostic = [
               'cobas', 'accu-chek', 'accutrend', 'coagucheck', 'reflotron',
               'urisys', 'sars-cov-2 rapid antibody test',
               'sars-cov-2 rapid antigen nasal test', 'sars-cov-2 rapid antigen test',
               'ventana', 'navify', 'mysugr', 'avenio', 'lightcycler', 'magna pure']

roche_prods = roche_pharma + roche_diagnostic
roche_prods = [s.lower() for s in roche_prods]
roche_prods = list(set(roche_prods))
roche_prods = [re.compile("\W{}\W".format(s)) for s in roche_prods]

def filter_roche_mentions(entry):
    text = entry["text"].lower()

    for prog in roche_prods:
        if prog.search(text) is not None:
            print("filtered: ", text)
            return None

    return entry


