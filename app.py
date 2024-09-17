import streamlit as st
import requests
import pandas as pd
from io import StringIO
import nltk

# Téléchargement des ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# URL de l'API FastAPI
api_url = "http://127.0.0.1:8000"  # Assurez-vous que c'est l'URL correcte

# Configuration de la page Streamlit
st.set_page_config(page_title="✨ Prédiction de Sentiment ✨", layout="wide")

# Initialiser la variable session_state pour suivre l'onglet actif
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 'Fichier CSV/TSV'

# Initialiser l'historique des prédictions pour les textes manuels dans session_state
if 'manual_text_history' not in st.session_state:
    st.session_state['manual_text_history'] = []

# Fonction pour changer l'onglet actif
def switch_tab(tab_name):
    st.session_state['active_tab'] = tab_name



# Créer des onglets pour les différentes options (fichier ou texte)
tab1, tab2, tab3 = st.tabs(["Fichier CSV/TSV", "Commentaire texte", "Historique"])

# Onglet pour l'upload de fichier
with tab1:
    st.header("📩 Uploader un fichier TSV")
    file = st.file_uploader("Choisir un fichier TSV", type=["tsv"])

    if file is not None:
        # Lecture du fichier uploadé
        df = pd.read_csv(file, sep='\t')
        st.write("Aperçu du fichier avant nettoyage :")
        st.dataframe(df)

        # Bouton pour effectuer le nettoyage via l'API
        if st.button("Nettoyer et analyser les sentiments", on_click=switch_tab, args=('Fichier CSV/TSV',)):
            # Envoi du fichier à l'API pour nettoyage et prédiction
            response = requests.post(f"{api_url}/predict-sentiment/", files={"file": file.getvalue()})

            if response.status_code == 200:
                # Lire la réponse de l'API et afficher le fichier modifié
                updated_tsv = StringIO(response.text)
                df_updated = pd.read_csv(updated_tsv, sep='\t')
                st.write("Fichier nettoyé et prédictions ajoutées :")
                st.dataframe(df_updated)

                st.write(f"Taille du fichier après nettoyage et prédictions : {df_updated.shape}")

                # Bouton pour télécharger le fichier modifié
                st.download_button(
                    label="Télécharger le fichier modifié",
                    data=response.text,
                    file_name="fichier_avec_sentiment.tsv",
                    mime="text/tsv"
                )
            else:
                st.error(f"Erreur lors de l'analyse des sentiments: {response.text}")



# Onglet pour entrer un commentaire texte
with tab2:
    st.header("⌨️ Entrer un commentaire en anglais")

    # Champ texte pour le commentaire
    text = st.text_area("Entrez un commentaire (min. 50 caractères)", max_chars=500)

    if st.button("Prédire à partir du texte", on_click=switch_tab, args=('Commentaire texte',)):
        if len(text) < 50:
            st.warning("Le commentaire doit contenir au moins 50 caractères.")
        else:
            # Requête à l'API pour prédire le sentiment à partir du texte
            response = requests.post(f"{api_url}/predict-text/", json={"text": text})
            if response.status_code == 200:
                prediction = response.json()

                # Vérifiez si la réponse est une liste (nettoyée)
                if isinstance(prediction, list):
                    # Afficher la phrase nettoyée et sa prédiction
                    st.write(f"Prédiction : {prediction[0]['sentiment_prediction']}")
                    st.dataframe(pd.DataFrame(prediction))  # Affichage du DataFrame nettoyé

                    # Ajouter à l'historique des textes manuels
                    st.session_state['manual_text_history'].append(prediction[0])  # Enregistrer la phrase nettoyée et prédiction
                else:
                    st.error(f"Erreur lors de la prédiction : {prediction['error']}")
            else:
                st.error(f"Erreur lors de la prédiction: {response.text}")



# Onglet pour afficher l'historique des prédictions de textes manuels
with tab3:
    st.header("Historique des prédictions (textes manuels uniquement)")

    if st.session_state['manual_text_history']:
        # Créer un DataFrame de l'historique des textes manuels
        df_history = pd.DataFrame(st.session_state['manual_text_history'])

        # Afficher le DataFrame
        st.dataframe(df_history)

        # Bouton pour télécharger l'historique sous forme de CSV
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="Télécharger l'historique en CSV",
            data=csv,
            file_name="historique_predictions_textes_manuels.csv",
            mime="text/csv"
        )
    else:
        st.write("Aucune prédiction enregistrée pour le moment.")
