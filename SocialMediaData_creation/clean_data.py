import pandas as pd
import contractions
import re
import string
import nltk
from nltk.corpus import stopwords
from typing import List

nltk.download("stopwords")
stop_words = stopwords.words("english")

"""
Default cleaning steps: 
    - expand_contractions
    - remove_special_characters
After self-report filter / optional steps
    - remove_urls
    - remove_punctuation
    - lowercase
""" 

def expand_contractions(texts: List[str]) -> List[str]:
    """Expands contractions for every string in texts.

    Args:
        texts (List[str]): input texts.

    Returns:
        List[str]: texts with contractions removed.
    """
    return [contractions.fix(x) for x in texts]
    
def lowercase(texts: List[str]) -> List[str]:
    """Lowercases every string in texts.

    Args:
        texts (List[str]): input texts.

    Returns:
        List[str]: lowercased texts.
    """
    return [x.lower() for x in texts]

def remove_urls(texts: List[str]) -> List[str]:
    """Removes URLs in every string in texts.

    Args:
        texts (List[str]): input texts.

    Returns:
        List[str]: texts with URLs removed.
    """
    return [re.sub(r"https?://\S+", "", x) for x in texts]

def remove_special_characters(texts: List[str]) -> List[str]:
    """Removes special characters (mentions, new line characters, "&amp;") from 
    every string in texts.

    Args:
        texts (List[str]): input texts.

    Returns:
        List[str]: texts with special characters removed.
    """
    def remove(text: str):
        text = re.sub("@\S*", "", text) # Remove mentions
        text = re.sub("\n", " ", text) # Remove new line characters
        text = re.sub("&amp;", "&", text) # Replace "&amp" with "&"
        text = text.strip() # Remove whitespace at either end
        return text
    
    return [remove(x) for x in texts]

def remove_punctuation(texts: List[str]) -> List[str]:
    """Remove punctuation.

    Args:
        texts (List[str]): input texts.

    Returns:
        List[str]: texts with punctuation removed.
    """
    return [x.translate(x.maketrans("", "", string.punctuation)) for x in texts]

def remove_stopwords(texts: List[str]) -> List[str]:
    """Removes NLTK stopwords from every string in texts.

    Args:
        texts (List[str]): input texts.

    Returns:
        List[str]: texts with NLTK stopwords removed.
    """
    return [" ".join([word for word in x.split() if word not in stop_words]) for x in texts]


if __name__ == "__main__":
    # Load data (below is just some examples)
    data = ["I'm tired #CovidLonghaul", 
            "See this study about COVID LH https://pubmed.ncbi.nlm.nih.gov/32644129/, \n their findings align with my experience",
            "  I have Long Covid &amp; I'm tired every single day. \nPls help @Doctor.    "]
    data = expand_contractions(data)
    data = remove_special_characters(data)
    
    # Save data
    print(data)