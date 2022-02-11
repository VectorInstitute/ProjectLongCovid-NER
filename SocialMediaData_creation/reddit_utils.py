
import hashlib
import string
import re

MENTIONS = re.compile(r'/u/\w*')

def re_hash(matchobj):
    user = matchobj.group(0)
    user = user[3:]
    print(user)
    user_hash = hashlib.sha224(
                    user.encode()).hexdigest()
    return "/u/" + user_hash


def anonymize_text(field):
    def f(entry):

        if "author" in entry:
            entry["author"] = hashlib.sha224(
                entry["author"].encode()).hexdigest()

        if field in entry:
            new_text = re.sub(MENTIONS, re_hash, str(entry[field]))
            entry[field] = new_text
        return entry

    return f

# Words that cannot be included in the data per Roche's request
pharma = ['roche', 'alecensa', 'alectinib', 'activase', 'alteplase', 'cotellic',
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

diagnostic = [
               'cobas', 'accu-chek', 'accutrend', 'coagucheck', 'reflotron',
               'urisys', 'sars-cov-2 rapid antibody test',
               'sars-cov-2 rapid antigen nasal test', 'sars-cov-2 rapid antigen test',
               'ventana', 'navify', 'mysugr', 'avenio', 'lightcycler', 'magna pure']

prods = pharma + diagnostic
prods = [s.lower() for s in prods]
prods = list(set(prods))
prods = [re.compile("\W{}\W".format(s)) for s in prods]

def remove_keywords(row, col_name):
    text = row[col_name].lower()

    for prog in prods:
        if prog.search(text) is not None:
            print("filtered: ", text)
            return False

    return True


