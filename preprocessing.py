import pandas as pd
import re
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# Function to keep first occurrence of duplicates
def keep_first_occurrence(df, column_name):
    return df.drop_duplicates(subset=[column_name], keep='first')

# Function to detect languages in a text column with error handling
def detecter_langues(df, colonne_texte):
    langues_detectees = []

    # Itérer sur chaque phrase dans la colonne spécifiée
    for phrase in df[colonne_texte].dropna():
        try:
            langue = detect(phrase)  # Détecter la langue de la phrase
            langues_detectees.append(langue)
        except:
            langues_detectees.append('unknown')  # Si la détection échoue, ajouter 'unknown'

    # Ajouter la colonne des langues détectées au DataFrame
    df['langue_detectee'] = langues_detectees

    # Afficher les langues détectées et le nombre de lignes pour chaque langue
    langues_comptage = df['langue_detectee'].value_counts()
    print("Langues détectées et nombre de lignes par langue :")
    print(langues_comptage)

    return df

# Function to remove non-English rows
def supprimer_non_anglais(df):
    df = df[df['langue_detectee'] == 'en']
    return df.drop(columns=['langue_detectee'])

# Clean text function to remove unwanted characters
def clean_text(text):
    text = re.sub(r'http\\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text.lower()

# Remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

# Tokenize text function
def tokenize_text(text):
    return word_tokenize(text)

# Remove vowel/consonant sequences function
def remove_consonant_or_vowel_sequences_from_tokens(tokens):
    pattern = r'(?:[bcdfghjklmnpqrstvwxyz]{3,}|[aeiou]{3,})'
    filtered_tokens = [token for token in tokens if not re.search(pattern, token)]
    return filtered_tokens

# Lemmatize tokens function
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Replace empty cells with NaN
def replace_empty_with_nan(df, column_name):
    # Remplacer les chaînes vides par NaN sans utiliser inplace=True
    df[column_name] = df[column_name].replace('', np.nan)
    return df


def remove_nan_rows(df):
    df.dropna(inplace=True)
    return df

# Remove short words function
def remove_short_words(tokens, min_length=2):
    # Vérifier que 'tokens' est une liste de chaînes, sinon retourner une liste vide
    if not isinstance(tokens, list):
        return []
    return [token for token in tokens if isinstance(token, str) and len(token) >= min_length]



#df_clean= remove_short_words(df_clean,'text_without_stopwords')


# Remove duplicates based on a column
def remove_duplicates_by_column(df, column_name):
    return df.drop_duplicates(subset=[column_name])

# Remove terms with 3 identical characters in a row
def supprimer_termes_3_caracteres_identiques(tokens):
    return [token for token in tokens if not re.search(r'(.)\\1{2,}', token)]

# Main function for the cleaning process
def nettoyage_automatisé(df):
    # 1. Garder la première occurrence basée sur 'SentenceId'
    df = keep_first_occurrence(df, 'SentenceId')

    # 2. Détecter les langues dans la colonne 'Phrase'
    df = detecter_langues(df, 'Phrase')

    # 3. Supprimer les lignes non anglaises
    df = supprimer_non_anglais(df)

    # 4. Nettoyer le texte dans la colonne 'Phrase'
    df['cleaned_text'] = df['Phrase'].apply(clean_text)

    # 5. Retirer les stopwords pour la colonne 'cleaned_text'
    df['text_without_stopwords'] = df['cleaned_text'].apply(remove_stopwords)

    # 6. Tokeniser le texte dans 'text_without_stopwords'
    df['tokens'] = df['text_without_stopwords'].apply(tokenize_text)

    # 7. Retirer les séquences de consonnes ou voyelles dans les 'tokens'
    df['tokens'] = df['tokens'].apply(remove_consonant_or_vowel_sequences_from_tokens)

    # 8. Lemmatiser les 'tokens'
    df['lemmatized_tokens'] = df['tokens'].apply(lemmatize_tokens)

    # 9. Remplacer les cellules vides par des NaN
    df = replace_empty_with_nan(df, 'Phrase')

    #10. Supprimer les NaN
    df = remove_nan_rows(df)

    # 10. Retirer les mots courts dans la colonne 'tokens'
    df['text_without_short_words'] = df['tokens'].apply(lambda tokens: remove_short_words(tokens, min_length=2))

    # 11. Supprimer les doublons par 'PhraseId'
    df = remove_duplicates_by_column(df, 'PhraseId')

    # 12. Supprimer les termes avec 3 caractères identiques dans 'lemmatized_tokens'
    df['lemmatized_tokens'] = df['lemmatized_tokens'].apply(lambda tokens: supprimer_termes_3_caracteres_identiques(tokens))

    return df
