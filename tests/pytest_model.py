import pytest
import joblib
from sklearn.naive_bayes import BernoulliNB

# Charger le modèle sauvegardé avec joblib
@pytest.fixture(scope="module")
def load_model():
    """Fixture pour charger le modèle une seule fois."""
    model = joblib.load('bernoulli_model.joblib')
    return model

# Fonction de mappage des prédictions en sentiments
def convert_prediction_to_sentiment(pred):
    """Convertit une prédiction numérique en un sentiment ('Négatif', 'Positif', ou 'Inconnu')."""
    if pred in [0, 1]:
        return "Négatif"
    elif pred in [3, 4]:
        return "Positif"
    else:
        return "Inconnu"

# Test des prédictions du modèle
def test_model_prediction(load_model):
    """Test pour vérifier que la prédiction du modèle est bien 'Négatif' ou 'Positif'."""
    model = load_model
    text = ["It was a magnificent and very moving film."]  # Assurez-vous que c'est une liste ou un tableau
    prediction = model.predict(text)[0]  # Prédire la classe

    # Convertir la prédiction numérique en texte (Négatif ou Positif)
    sentiment = convert_prediction_to_sentiment(prediction)

    # Vérification que la prédiction est bien une chaîne de caractères (Négatif ou Positif)
    assert sentiment in ["Négatif", "Positif"], f"Prédiction inattendue: {sentiment}"

# Test de la fonction de mappage des prédictions
def test_convert_prediction_to_sentiment():
    """Test pour vérifier que les prédictions numériques sont correctement mappées aux sentiments."""
    assert convert_prediction_to_sentiment(0) == "Négatif"
    assert convert_prediction_to_sentiment(1) == "Négatif"
    assert convert_prediction_to_sentiment(3) == "Positif"
    assert convert_prediction_to_sentiment(4) == "Positif"

# Test d'une prédiction complète avec mappage des sentiments
def test_full_prediction(load_model):
    """Test pour vérifier que le modèle produit une prédiction correcte sous forme de texte et que le sentiment est mappé correctement."""
    model = load_model
    text = ["I didn't like this film."]  # Utilisez toujours une liste pour le modèle
    prediction = model.predict(text)[0]  # Prédire la classe
    sentiment = convert_prediction_to_sentiment(prediction)

    # Vérification que la prédiction est bien un sentiment "Négatif" ou "Positif"
    assert sentiment in ["Négatif", "Positif"], f"Sentiment inattendu: {sentiment}"
