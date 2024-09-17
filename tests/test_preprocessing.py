import pytest
import pandas as pd
import re
from preprocessing import (
    keep_first_occurrence,
    detecter_langues,
    supprimer_non_anglais,
    clean_text,
    remove_stopwords,
    tokenize_text,
    remove_consonant_or_vowel_sequences_from_tokens,
    lemmatize_tokens,
    replace_empty_with_nan,
    remove_short_words,
    remove_duplicates_by_column,
    supprimer_termes_3_caracteres_identiques
)

@pytest.fixture
def sample_df():
    # Créer un DataFrame minimal pour les tests
    return pd.DataFrame({
        'sentenceID': [1, 2, 3, 4],
        'Phrase': ['Hello World!', 'Ceci est un test.', 'Bonjour le monde!', 'Hello World!'],
        'cleaned_text': ['Hello World!', 'Ceci est un test', 'Bonjour le monde', 'Hello World!'],
        'tokens': [['Hello', 'World'], ['Ceci', 'est', 'un', 'test'], ['Bonjour', 'le', 'monde'], ['Hello', 'World']]
    })

def test_keep_first_occurrence(sample_df):
    result = keep_first_occurrence(sample_df, 'sentenceID')
    assert len(result) == 4  # Pas de doublons dans 'sentenceID', donc la taille reste la même

def test_detecter_langues(sample_df):
    result = detecter_langues(sample_df, 'Phrase')
    assert 'langue_detectee' in result.columns  # Vérifie si la colonne 'langue_detectee' a été ajoutée
    assert len(result['langue_detectee']) == len(sample_df)  # Autant de langues détectées que de phrases

def test_supprimer_non_anglais(sample_df):
    sample_df['langue_detectee'] = ['en', 'fr', 'fr', 'en']
    result = supprimer_non_anglais(sample_df)
    assert len(result) == 2  # Les phrases non-anglaises doivent être supprimées

def test_clean_text(sample_df):
    sample_df['cleaned_text'] = sample_df['Phrase'].apply(clean_text)
    assert all('!' not in text for text in sample_df['cleaned_text'])  # Vérifie si les caractères spéciaux ont été supprimés

def test_remove_stopwords(sample_df):
    sample_df['text_without_stopwords'] = sample_df['cleaned_text'].apply(remove_stopwords)
    assert 'the' not in sample_df['text_without_stopwords'].iloc[0]  # Remplacez "le" par "the" pour l'anglais

def test_tokenize_text(sample_df):
    sample_df['tokens'] = sample_df['cleaned_text'].apply(tokenize_text)
    assert isinstance(sample_df['tokens'].iloc[0], list)  # Les tokens doivent être une liste

def test_remove_consonant_or_vowel_sequences_from_tokens(sample_df):
    sample_df['tokens'] = sample_df['tokens'].apply(remove_consonant_or_vowel_sequences_from_tokens)
    assert all(not re.search(r'(.)\1{2,}', token) for tokens in sample_df['tokens'] for token in tokens)  # Vérifie les séquences indésirables

def test_lemmatize_tokens(sample_df):
    # Ajout de plus de données pour que les tailles correspondent
    sample_df['tokens'] = [['running', 'cats'], ['walking', 'dogs'], ['playing', 'balls'], ['eating', 'fruits']]
    sample_df['lemmatized_tokens'] = sample_df['tokens'].apply(lemmatize_tokens)

    # Vérification que la lemmatisation change les tokens (test avec des mots affectés par la lemmatisation)
    assert sample_df['lemmatized_tokens'].iloc[0] != sample_df['tokens'].iloc[0]
def test_replace_empty_with_nan(sample_df):
    sample_df['cleaned_text'] = ['', 'test', '', 'Hello']
    result = replace_empty_with_nan(sample_df, 'cleaned_text')
    assert pd.isna(result['cleaned_text'].iloc[0])  # Les cellules vides doivent être remplacées par NaN

def test_remove_short_words(sample_df):
    sample_df['tokens'] = sample_df['tokens'].apply(lambda tokens: remove_short_words(tokens, min_length=3))
    assert all(len(token) >= 3 for tokens in sample_df['tokens'] for token in tokens)  # Les mots courts doivent être supprimés

def test_remove_duplicates_by_column(sample_df):
    new_row = pd.DataFrame({'sentenceID': [1], 'Phrase': ['Duplicate test']})
    sample_df = pd.concat([sample_df, new_row], ignore_index=True)
    result = remove_duplicates_by_column(sample_df, 'sentenceID')
    assert len(result) == 4  # Les doublons doivent être supprimés


def test_supprimer_termes_3_caracteres_identiques():
   tokens = ['hello', 'world', 'good']
   result = supprimer_termes_3_caracteres_identiques(tokens)
   assert result == ['hello', 'world', 'good'], "Échec du test pour les tokens sans caractères identiques"
