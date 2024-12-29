import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import pandas as pd

# Titre de l'application
st.title("Application d'Analyse de Marché Financier")

# Entrée utilisateur : Symbole de l'action
symbol = st.text_input("Entrez le symbole de l'action (ex. AAPL, TSLA):", "AAPL")

# Récupération des données
if symbol:
    st.write(f"Données historiques pour {symbol}")
    
    # Télécharger les données via Yahoo Finance
    try:
        data = yf.download(symbol, start="2020-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
        if data.empty:
            st.error("Aucune donnée trouvée pour ce symbole.")
        else:
            st.write(data.tail())
            
            # Afficher les prix historiques
            st.subheader("Graphique des prix historiques")
            plt.figure(figsize=(10, 5))
            plt.plot(data['Close'], label='Prix de clôture')
            plt.title(f"Prix de clôture de {symbol}")
            plt.xlabel("Date")
            plt.ylabel("Prix")
            plt.legend()
            st.pyplot(plt)
            
            # Modèle de prédiction avec ARIMA
            st.subheader("Prévision des tendances")
            
            # Préparer les données pour ARIMA
            training_data = data['Close']
            model = auto_arima(training_data, seasonal=False, trace=True, suppress_warnings=True)
            
            # Faire des prévisions pour les 7 prochains jours
            forecast = model.predict(n_periods=7)
            future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=7)
            forecast_df = pd.DataFrame({"Date": future_dates, "Prévision": forecast}).set_index("Date")

            # Afficher les prévisions
            st.write(forecast_df)

            # Graphique des prévisions
            plt.figure(figsize=(10, 5))
            plt.plot(data['Close'], label='Prix historique')
            plt.plot(forecast_df['Prévision'], label='Prévision', linestyle='--', color='orange')
            plt.title(f"Prévisions des 7 prochains jours pour {symbol}")
            plt.xlabel("Date")
            plt.ylabel("Prix")
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Erreur : {e}")
