from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from preprocessing import nettoyage_automatisé  # Assurez-vous que c'est la bonne fonction
import io
from langdetect import detect, LangDetectException
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

# Charger le modèle
model = joblib.load("model/bernoulli_model.joblib")

# Définir un modèle Pydantic pour l'entrée de texte
class TextInput(BaseModel):
    text: str

# Fonction pour détecter si le texte est en anglais
def is_english(text: str) -> bool:
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return False

# Endpoint pour le nettoyage et la prédiction de sentiments à partir d'un fichier CSV/TSV
@app.post("/predict-sentiment/")
async def predict_sentiment(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # Lire le fichier TSV original
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep='\t')
        print(f"Taille du fichier original : {df.shape}")

        # Appliquer le prétraitement
        df_cleaned = nettoyage_automatisé(df)
        print(f"Taille après nettoyage : {df_cleaned.shape}")

        # Vérifier que la colonne textuelle 'cleaned_text' est présente
        if 'cleaned_text' not in df_cleaned.columns:
            raise HTTPException(status_code=400, detail="Colonne textuelle 'cleaned_text' manquante après le nettoyage.")

        # Prédire le sentiment
        predictions = model.predict(df_cleaned['cleaned_text'])
        print(f"Nombre de prédictions générées : {len(predictions)}")

        # Assurez-vous que les tailles correspondent
        if len(predictions) != len(df_cleaned):
            raise HTTPException(status_code=500, detail="Le nombre de prédictions ne correspond pas au nombre de lignes après nettoyage.")

        # Mapper les prédictions (0, 1 -> Négatif) et (3, 4 -> Positif)
        sentiment_labels = {0: 'Négatif', 1: 'Négatif', 3: 'Positif', 4: 'Positif'}
        df_cleaned['sentiment'] = [sentiment_labels.get(pred, 'Neutre') for pred in predictions]

        # Convertir le DataFrame nettoyé avec les prédictions en TSV
        output = df_cleaned.to_csv(sep='\t', index=False)

        # Renvoyer le fichier TSV nettoyé avec les prédictions
        return Response(content=output, media_type="text/tsv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

# Endpoint pour le nettoyage seul à partir d'un fichier CSV/TSV
@app.post("/clean-csv/")
async def clean_csv(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep='\t')
        df_cleaned = nettoyage_automatisé(df)
        return df_cleaned.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du nettoyage : {str(e)}")

# Modèle pour recevoir les données nettoyées
class CleanedDataModel(BaseModel):
    cleaned_data: list

# Endpoint pour faire la prédiction à partir d'un fichier déjà nettoyé
@app.post("/predict-from-cleaned/")
async def predict_from_cleaned(data: CleanedDataModel):
    try:
        # Convertir la liste des données nettoyées en DataFrame
        df_cleaned = pd.DataFrame(data.cleaned_data)

        # Vérifiez que les données ont les colonnes nécessaires pour la prédiction
        if df_cleaned.empty or 'cleaned_text' not in df_cleaned.columns:
            raise HTTPException(status_code=400, detail="Les données nettoyées sont invalides ou manquent de colonnes nécessaires.")

        # Faire la prédiction
        predictions = model.predict(df_cleaned['cleaned_text'])

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

# Endpoint pour prédire directement à partir d'une phrase entrée à la main
@app.post("/predict-text/")
async def predict_text(input: TextInput):
    text = input.text

    # Vérification de la langue
    if not is_english(text):
        return {"error": "Le commentaire n'est pas en anglais. Veuillez entrer un commentaire en anglais."}

    # Créer un DataFrame à partir de la phrase saisie manuellement
    df = pd.DataFrame([{"Phrase": text, "SentenceId": 1, "PhraseId": 1}])

    # Appliquer le preprocessing (nettoyage automatisé)
    df_cleaned = nettoyage_automatisé(df)
    print(f"Phrase après nettoyage : {df_cleaned}")

    # Vérifier que la colonne textuelle 'cleaned_text' est présente après nettoyage
    if 'cleaned_text' not in df_cleaned.columns:
        raise HTTPException(status_code=400, detail="Erreur lors du nettoyage : la colonne 'cleaned_text' est manquante.")

    # Faire la prédiction
    predictions = model.predict(df_cleaned['cleaned_text'])
    print(f"Prédiction : {predictions}")

    # Mapper les prédictions (0, 1 -> Négatif) et (3, 4 -> Positif)
    sentiment_labels = {0: 'Négatif', 1: 'Négatif', 3: 'Positif', 4: 'Positif'}
    df_cleaned['sentiment_prediction'] = sentiment_labels.get(predictions[0], "Inconnu")

    # Retourner le DataFrame nettoyé avec la prédiction
    return df_cleaned.to_dict(orient='records')
