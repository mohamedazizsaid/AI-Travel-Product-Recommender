from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import mysql.connector
import os
import uuid
import numpy as np  # ✅ needed for log/exp
import matplotlib
matplotlib.use('Agg')  # 🔥 No GUI
import matplotlib.pyplot as plt
import pickle
from flask import jsonify  # for returning JSON responses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import joblib
model_pipeline = joblib.load('sentiment_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
df_elec = pd.read_csv("data/fullelectronique (2).csv")

model_flight = None
features_encoded_flight = None

model_hotel = joblib.load("hotel_price_model.pkl")
features_hotel = model_hotel.feature_names_in_
app = Flask(__name__)
os.makedirs("static", exist_ok=True)

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="product_catalogue"
    )

def load_products():
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT product_full FROM all_products", conn)
    conn.close()
    return df['product_full'].dropna().tolist()

PRODUCT_LIST = load_products()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/products', methods=['GET', 'POST'])
def products():
    history_plot = None
    forecast_plot = None
    selected_product = None

    if request.method == 'POST':
        selected_product = request.form['product_name']

        conn = get_connection()
        df = pd.read_sql("SELECT * FROM all_products", conn)
        conn.close()

        price_cols = [col for col in df.columns if col.startswith("prix_")]
        df_long = df.melt(
            id_vars=["product_full", "marque", "category", "available", "store"],
            value_vars=price_cols,
            var_name="periode",
            value_name="prix_saisonnier"
        )
        df_long[['year', 'season']] = df_long['periode'].str.extract(r"prix_(\d{4})_(\w+)?")
        df_long['year'] = df_long['year'].fillna('2025').astype(int)
        df_long['season'] = df_long['season'].fillna('winter')
        season_map = {'winter': 1, 'spring': 4, 'summer': 7, 'fall': 10}
        df_long['month'] = df_long['season'].map(season_map)
        df_long['date'] = pd.to_datetime(df_long[['year', 'month']].assign(day=1))

        subset = df_long[df_long['product_full'] == selected_product][['date', 'prix_saisonnier']].dropna()

        if not subset.empty:
            unique_id = str(uuid.uuid4())[:8]

            plt.figure(figsize=(10, 5))
            plt.plot(subset['date'], subset['prix_saisonnier'], marker='o')
            plt.title(f"Price History: {selected_product}")
            plt.xlabel("Date")
            plt.ylabel("Price (DT)")
            plt.grid(True)
            plt.tight_layout()
            history_plot = f"static/history_{unique_id}.png"
            plt.savefig(history_plot)
            plt.close()

            ts = subset.rename(columns={'date': 'ds', 'prix_saisonnier': 'y'})
            model = Prophet(yearly_seasonality=True)
            model.fit(ts)
            future = model.make_future_dataframe(periods=4, freq='Q')
            forecast = model.predict(future)

            plt.figure(figsize=(10, 5))
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.4)
            plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
            plt.scatter(ts['ds'], ts['y'], color='black', label='Actual')
            plt.title(f"Price Forecast: {selected_product}")
            plt.xlabel("Date")
            plt.ylabel("Forecasted Price (DT)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            forecast_plot = f"static/forecast_{unique_id}.png"
            plt.savefig(forecast_plot)
            plt.close()

    return render_template('products.html',
                           product_list=PRODUCT_LIST,
                           selected_product=selected_product,
                           history_plot=history_plot,
                           forecast_plot=forecast_plot)

@app.route('/hotels/select_city', methods=['GET', 'POST'])
def select_city():
    conn = get_connection()
    cities = pd.read_sql("SELECT DISTINCT city FROM hotels WHERE city IS NOT NULL", conn)['city'].tolist()
    conn.close()

    if request.method == 'POST':
        selected_city = request.form['city']
        return redirect(url_for('select_hotel', city=selected_city))

    return render_template('select_city.html', cities=cities)

@app.route('/hotels/<city>', methods=['GET', 'POST'])
def select_hotel(city):
    forecast_plot = None
    selected_hotel = None

    conn = get_connection()
    hotels = pd.read_sql("SELECT DISTINCT name FROM hotels WHERE city = %s", conn, params=(city,))
    hotel_list = hotels['name'].dropna().tolist()
    conn.close()

    if request.method == 'POST':
        selected_hotel = request.form['hotel_name']

        conn = get_connection()
        df = pd.read_sql("SELECT * FROM hotels WHERE name = %s", conn, params=(selected_hotel,))
        conn.close()

        price_cols = [col for col in df.columns if col.startswith("prix_")]
        df_long = pd.melt(
            df,
            id_vars=["name", "city", "formule"],
            value_vars=price_cols,
            var_name="periode",
            value_name="prix_saisonnier"
        )
        df_long[['year', 'season']] = df_long['periode'].str.extract(r"prix_(\d{4})_(\w+)?")
        df_long['season'] = df_long['season'].fillna('winter')
        season_map = {'winter': 1, 'spring': 2, 'summer': 3, 'fall': 4}
        df_long['season_num'] = df_long['season'].map(season_map)
        df_long['date'] = pd.to_datetime(df_long['year'] + '-' + (df_long['season_num'] * 3).astype(str) + '-01')

        subset = df_long[['date', 'prix_saisonnier']].dropna().rename(columns={'date': 'ds', 'prix_saisonnier': 'y'})

        if not subset.empty:
            # Set floor and cap for logistic growth
            subset['floor'] = 10
            subset['cap'] = 150

            unique_id = str(uuid.uuid4())[:8]

            model = Prophet(growth='logistic', yearly_seasonality=True)
            model.fit(subset)

            future = model.make_future_dataframe(periods=4, freq='Q')
            future['floor'] = 10
            future['cap'] = 150

            forecast = model.predict(future)

            # Clip just in case, extra safety
            forecast['yhat'] = forecast['yhat'].clip(lower=10, upper=150)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=10, upper=150)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=10, upper=150)

            plt.figure(figsize=(10, 5))
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.4)
            plt.plot(forecast['ds'], forecast['yhat'], label='Prévision')
            plt.scatter(subset['ds'], subset['y'], color='black', label='Historique')
            plt.title(f"Prévision des prix – {selected_hotel}")
            plt.xlabel("Date")
            plt.ylabel("Prix prévu (DT)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            forecast_plot = f"static/hotel_forecast_{unique_id}.png"
            plt.savefig(forecast_plot)
            plt.close()

    return render_template("select_hotel.html",
                           city=city,
                           hotel_list=hotel_list,
                           selected_hotel=selected_hotel,
                           forecast_plot=forecast_plot)

@app.route('/hotels')
def hotels_redirect():
    return redirect(url_for('select_city'))

@app.route('/recommend', methods=['GET', 'POST'])
def recommendation_form():
    import torch
    import pandas as pd
    from torch import nn

    # Load necessary files
    df = pd.read_csv("client_product_enhanced_final.csv")
    all_products = pd.read_sql("SELECT id, product_full, marque, category, prix_2025_winter FROM all_products", get_connection())
    
    user_index = pd.read_csv("user_mapping.csv", index_col=0, header=None).squeeze()
    item_index = pd.read_csv("item_mapping.csv", index_col=0, header=None).squeeze()

    product_list = [{'id': row['id'], 'name': row['product_full']} for _, row in all_products.iterrows()]
    recommendations = []

    class RecommenderNN(nn.Module):
        def __init__(self, n_users, n_items, n_brands, n_categories, emb_size=50):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, emb_size)
            self.item_emb = nn.Embedding(n_items, emb_size)
            self.brand_emb = nn.Embedding(n_brands, emb_size // 2)
            self.cat_emb = nn.Embedding(n_categories, emb_size // 2)
            self.fc = nn.Sequential(
                nn.Linear(emb_size * 2 + emb_size + 1, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, user, item, brand, category, price):
            u = self.user_emb(user)
            i = self.item_emb(item)
            b = self.brand_emb(brand)
            c = self.cat_emb(category)
            x = torch.cat([u, i, b, c, price.unsqueeze(1)], dim=1)
            return self.fc(x).squeeze(1)

    # Load model
    df['user_enc'], _ = pd.factorize(df['client_id'])
    df['item_enc'], _ = pd.factorize(df['product_id'])
    n_users = df['user_enc'].nunique()
    n_items = df['item_enc'].nunique()
    n_brands = df['brand'].nunique()
    n_categories = df['categorie_enc'].nunique()

    model = RecommenderNN(n_users, n_items, n_brands, n_categories)
    model.load_state_dict(torch.load("recommender_model.pth", map_location=torch.device('cpu')))
    model.eval()

    if request.method == 'POST':
        selected_ids = request.form.getlist('selected_products')

        if len(selected_ids) >= 5:
            selected_ids = list(map(int, selected_ids))

            # Filter out selected products
            candidate_df = df[~df['product_id'].isin(selected_ids)].drop_duplicates('product_id')

            # Get info about selected products
            selected_df = df[df['product_id'].isin(selected_ids)]

            # Build pseudo-user profile (most frequent brand/category, average price)
            brand = torch.tensor(selected_df['brand'].astype('category').cat.codes.mode()[0], dtype=torch.long).repeat(len(candidate_df))
            category = torch.tensor(selected_df['categorie_enc'].mode()[0], dtype=torch.long).repeat(len(candidate_df))
            price = torch.tensor(candidate_df['price'].values, dtype=torch.float32)

            # Dummy user and item encodings
            fake_user = torch.zeros(len(candidate_df), dtype=torch.long)
            item = torch.tensor(candidate_df['item_enc'].values, dtype=torch.long)

            # Run prediction
            with torch.no_grad():
                scores = model(fake_user, item, brand, category, price)

            candidate_df = candidate_df.copy()
            candidate_df['score'] = scores.numpy()
            top_5 = candidate_df.sort_values('score', ascending=False).head(5)

            recommended_ids = top_5['product_id'].tolist()
            recommendations = all_products[all_products['id'].isin(recommended_ids)]['product_full'].tolist()
        else:
            recommendations = ["⚠️ Please select at least 5 products."]

    return render_template('recommend.html',
                           product_list=product_list,
                           recommendations=recommendations)

# Load model and scaler at startup
with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
@app.route('/flight_cancel', methods=['GET', 'POST'])
def flight_cancel_predict():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Still collect compagnie for display or future use
            compagnie = request.form['Compagnie']  # Not used for prediction

            distance_vol = float(request.form['Distance_Vol_KM'])
            nombre_escales = int(request.form['Nombre_Escales'])
            saison_touristique = int(request.form['Saison_Touristique'])

            # Scale the 3 features used in training
            features = [[distance_vol, nombre_escales, saison_touristique]]
            X_scaled = scaler.transform(features)

            prediction_val = model.predict(X_scaled)[0]
            prediction = "✈️ Annulé" if prediction_val == 1 else "🛫 Non annulé"

        except Exception as e:
            error = f"Une erreur est survenue : {str(e)}"

    return render_template("flight_cancel_predict.html", prediction=prediction, error=error)



@app.route('/hotel1', methods=['GET', 'POST'])
def hotel1_home():
    global model_hotel, features_hotel

    if model_hotel is None:
        return "❌ Modèle hôtel non chargé."

    if request.method == 'POST':
        try:
            input_data = {
                'nb_etoiles': int(request.form['nb_etoiles']),
                'Mois': int(request.form['Mois']),
                'city': request.form['city'],
                'formule': request.form['formule'],
                'name': request.form['name']
            }

            df_input = pd.DataFrame([input_data])
            df_encoded = pd.get_dummies(df_input)

            for col in features_hotel:
                if col not in df_encoded:
                    df_encoded[col] = 0
            df_encoded = df_encoded[features_hotel]

            prediction = model_hotel.predict(df_encoded)[0]
            return render_template("hotels1.html", prediction=round(prediction, 2))

        except Exception as e:
            return render_template("hotels1.html", error=f"Erreur: {str(e)}")

    # On GET request, just render the form
    return render_template("hotels1.html")

    
    
@app.route('/predict_flight', methods=['POST'])
def predict_flight_price():
    global model_flight, features_encoded_flight

    if model_flight is None:
        return "❌ Modèle de vol non chargé."

    try:
        input_data = {
            'Nombre_Escales': int(request.form['Nombre_Escales']),
            'Taxe_Price': float(request.form['Taxe_Price']),
            'AirlineName': request.form['AirlineName'],
            'Region': request.form['Region'],
            'Mois': int(request.form['Mois']),
            'Jour': int(request.form['Jour'])
        }

        df_input = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df_input)

        for col in features_encoded_flight:
            if col not in df_encoded:
                df_encoded[col] = 0
        df_encoded = df_encoded[features_encoded_flight]

        prediction = model_flight.predict(df_encoded)[0]
        return render_template('vol.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template('vol.html', error=f"❌ Erreur : {e}")


@app.route('/flight_price', methods=['GET'])
def flight_price_form():
    return render_template('vol.html')

def load_flight_model():
    global model_flight, features_encoded_flight
    try:
        model_flight = joblib.load('model_flight.pkl')
        features_encoded_flight = model_flight.feature_names_in_
        print("✅ Modèle de vol chargé.")
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle de vol : {e}")


@app.route('/sentiment', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        try:
            commentaire = request.form['commentaire']
            if len(commentaire.strip()) < 5:
                prediction_label = "inconnu"
                message = "Texte trop court ou vide 😶. Veuillez entrer un commentaire valide."
            else:
                input_data = pd.DataFrame([{'commentaire': commentaire}])
                prediction_encoded = model_pipeline.predict(input_data)[0]
                prediction_label = label_encoder.inverse_transform([int(prediction_encoded)])[0]

                if prediction_label == 'positif':
                    message = "Client satisfait 😊🎉!"
                elif prediction_label == 'négatif':
                    message = "Client non satisfait 😞 "
                else:
                    message = "Merci pour votre retour neutre 🙂"

            return render_template('sentiment.html', prediction=prediction_label, message=message)

        except Exception as e:
            return render_template('sentiment.html', error=f"Erreur de prédiction : {str(e)}")
    
    return render_template('sentiment.html')

@app.route('/chatbot')
def chatbot_home():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def chatbot_ask():
    user_input = request.json.get("message")
    corpus = df_elec['product_full'].astype(str).tolist()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus)
    query_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    index = similarity.argmax()

    if similarity[index] < 0.2:
        return jsonify({"response": "❌ Désolé, je n'ai pas trouvé ce produit."})

    produit = df_elec.iloc[index]
    response = f"""
💻 {produit['product_full']}
🏷️ Marque: {produit['marque']}
🏬 Vendeur: {produit['Source']}
💵 Prix: {produit['prix']} TND
"""
    return jsonify({"response": response})


if __name__ == '__main__':
    load_flight_model()
    app.run(debug=True)
