import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("📈 Previsione Prezzo Casa")

# 1. Caricamento file
file = st.file_uploader("Carica un file CSV con colonne 'Superficie_mq' e 'Prezzo_migliaia'", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📊 Dati caricati:")
    st.write(df)

    # 2. Selezione X e y
    try:
        X = df[['Superficie_mq']]
        y = df['Prezzo_migliaia']

        # 3. Divisione train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Modello
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 5. Risultati
        st.subheader("📍 Confronto Valori Reali vs Predetti:")
        result = pd.DataFrame({'Reale': y_test.values, 'Predetto': y_pred.round(2)})
        st.write(result)

        # 6. Grafico
        st.subheader("📉 Grafico Regressione:")
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Dati reali')
        ax.plot(X, model.predict(X), color='red', label='Modello')
        ax.set_xlabel('Superficie (mq)')
        ax.set_ylabel('Prezzo (in migliaia)')
        ax.set_title('Previsione Prezzo Casa')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except KeyError:
        st.error("❌ Il file deve contenere le colonne 'Superficie_mq' e 'Prezzo_migliaia'")
else:
    st.info("📥 Carica un file CSV per iniziare.")