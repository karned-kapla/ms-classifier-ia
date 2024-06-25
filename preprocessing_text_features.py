import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import spacy
from langdetect import detect
import re

set_stop_words_french = set(stopwords.words('french'))
set_stop_words_english = set(stopwords.words('english'))
nlp_fr = spacy.load('fr_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')


def _stop_words_filtering(liste_mots, set_stop_words):
    tokens = [mot for mot in liste_mots if mot not in set_stop_words]
    return tokens


def _tokenisation_et_lemmatisation(mots, nlp_for_language):
    sortie = []
    phrase_nlp = nlp_for_language(mots)
    for mot in phrase_nlp:
        if mot.lemma_.startswith('zz'):  # pour garder la répétition des mots
            sortie.append(mot.lemma_)
        else:
            if (mot.is_stop == False):  # pour ne pas prendre les stop-words
                lemma = mot.lemma_  # on ne rajoute que les lemmes
                if (lemma not in sortie): sortie.append(lemma)
    return sortie


def token_lemmatization_and_remove_stop_words(text: str) -> str:
    """
    Args:
        text (str): the text string transform
            
    Returns:
        (str) the text without stop words
    """
    token_word_tokenize = lambda row: word_tokenize(row)
    token_remove_stop_words_en = lambda row: _stop_words_filtering(row, set_stop_words_english)
    token_remove_stop_words_fr = lambda row: _stop_words_filtering(row, set_stop_words_french)

    token_spacy_en = lambda row: _tokenisation_et_lemmatisation(row, nlp_en)
    token_spacy_fr = lambda row: _tokenisation_et_lemmatisation(row, nlp_fr)
    token_rebuild_text = lambda row: ' '.join(row)

    def _text_remove_stop_words(row):
        if detect(row) == 'en':
            return token_rebuild_text(token_spacy_en(row))
        elif detect(row) == 'fr':
            return token_rebuild_text(token_spacy_fr(row))
        else:
            return token_rebuild_text(token_remove_stop_words_fr(token_remove_stop_words_en(token_word_tokenize(row))))

    return _text_remove_stop_words(text)


def lemmatization_and_remove_stop_words_in_dataframe(df_with_text: pd.DataFrame) -> pd.DataFrame:
    """
        Args:
            df_filenames (pd.DataFrame): dataframe containing 
                the list of 'filename' as index 
                column 'text' the ocerised text
            OCR_file_loaded_from_or_saved_to : the file with previous extracted features
    
        Returns:
            pd.DataFrame: return the dataframe passed in argument completed with extracted OCR
        """
    df_with_text["text"] = df_with_text["text"].apply(lambda text: token_lemmatization_and_remove_stop_words(text))
    return df_with_text


def clean_transform_ocerised_text(text: str) -> str:
    """
    Args:
        text (str): the text string to clean and transform
            
    Returns:
        (str) the cleaned and transformed text
    """

    # transform the string in lower cases
    x = (lambda row: str(row).lower())(text)
    # replace specific keywords depending on classes
    x = (lambda row: re.sub(r"to:|to\s+:", " zztozz zztozz zztozz", row))(x)  # email ou letter : traite le "to:"
    x = (lambda row: re.sub(r"from:|from\s+:", " zzfromzz zzfromzz zzfromzz", row))(
        x)  # email ou letter : traite le "from:"
    x = (lambda row: re.sub(r"cc:|cc\s+:", " zzcczz zzcczz zzcczz ", row))(x)  # email ou letter : traite le "cc:"
    x = (lambda row: re.sub(r"subject:|subject\s+:", " zzsubjectzz zzsubjectzz zzsubjectzz zzsubjectzz zzsubjectzz ",
                            row))(x)  # email ou letter : traite le "subject:"
    x = (lambda row: re.sub(r"\?|question", " zzquestionzz ", row))(x)  # questionnaire : traite le "?"
    x = (lambda row: re.sub(r"\?|dear", " zzdearzz zzdearzz zzdearzz zzdearzz zzdearzz ", row))(
        x)  # email ou letter : mot ou formule de politesse
    x = (lambda row: re.sub(r"\?|respectfully", " zzrespectfullyzz zzrespectfullyzz zzrespectfullyzz ", row))(
        x)  # email ou letter : mot ou formule de politesse
    # split as tokens
    x = (lambda row: word_tokenize(row))(x)
    # remove word when too small
    minletters = 2
    x = (lambda row: [word for word in row if len(word) >= minletters])(x)
    # remove word when too few
    minwords = 2
    x = (lambda row: row if len(row) > minwords else [])(x)
    # collapse the token back in one string
    x = (lambda row: ' '.join(row))(x)
    x = (lambda text: np.nan if (len(text) < 5) else text)(x)
    fct_clean = x

    return fct_clean


def clean_transform_dataframe_with_ocerised_text(df_with_text: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df_with_text (pd.DataFrame): dataframe containing 
            the list of 'filename' as index 
            column 'text' the ocerised text
        OCR_file_loaded_from_or_saved_to : the file with previous extracted features

    Returns:
        pd.DataFrame: return the dataframe passed in argument completed with extracted OCR
    """
    df_with_text["text"] = df_with_text["text"].apply(lambda text: clean_transform_ocerised_text(text))
    df_with_text = df_with_text.dropna(subset = ['text'])
    return df_with_text


if __name__ == "__main__":
    pass
