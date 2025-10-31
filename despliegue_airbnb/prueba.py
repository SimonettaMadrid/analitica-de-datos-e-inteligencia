# Creamos el archivo de la APP en el interprete principal (Python)
###se corre con streamlit run prueba.py
#####################################################
# Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
######################################################

# Definimos la instancia
@st.cache_resource
def load_data():
    # Lectura del archivo csv
    dfb = pd.read_csv("barcelona_super_limpio.csv")
    dfc = pd.read_csv("cambridgelimpio.csv")
    dfbo = pd.read_csv("limpiosB.csv")  # Boston
    dfh = pd.read_csv("hawaii_limpio.csv")
    dfbu = pd.read_csv("Budapest_Limpio.csv")
    return dfb, dfc, dfbo, dfh, dfbu


# limpieza de datos
def clean_data(dfb, dfc, dfbo, dfh, dfbu):
    # Barcelona
    if "price" in dfb.columns:
        dfb["price"] = (
            dfb["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    # Cambridge
    if "price" in dfc.columns:
        dfc["price"] = (
            dfc["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    # Boston
    if "price" in dfbo.columns:
        dfbo["price"] = (
            dfbo["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    # Hawái
    if "price" in dfh.columns:
        dfh["price"] = (
            dfh["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    # Budapest
    if "price" in dfbu.columns:
        dfbu["price"] = (
            dfbu["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    return dfb, dfc, dfbo, dfh, dfbu


###############################################################################
# CREACIÓN DEL DASHBOARD
###############################################################################
# Sidebar
st.sidebar.title("Análisis de Datos Airbnb")

# Widget 1: Selectbox (vista)
View = st.sidebar.selectbox(
    label="Tipo de Análisis",
    options=["Extracción de Características", "Tablas comparativas", "Regresión Lineal"],
)

# Widget 2: Checkbox (ver datos)
show_data = st.sidebar.checkbox(label="Mostrar Datos")

# Cargamos datos
dfb, dfc, dfbo, dfh, dfbu = load_data()
dfb, dfc, dfbo, dfh, dfbu = clean_data(dfb, dfc, dfbo, dfh, dfbu)

# Mostrar datos
if show_data:
    st.subheader("Datos de Airbnb en Barcelona")
    st.dataframe(dfb.head(10))
    st.subheader("Datos de Airbnb en Cambridge")
    st.dataframe(dfc.head(10))
    st.subheader("Datos de Airbnb en Boston")
    st.dataframe(dfbo.head(10))
    st.subheader("Datos de Airbnb en Hawái")
    st.dataframe(dfh.head(10))
    st.subheader("Datos de Airbnb en Budapest")
    st.dataframe(dfbu.head(10))

# Multiselect para extracción (puede elegir varias)
ciudades_multiselect = st.sidebar.multiselect(
    label="Selecciona las Ciudades (Extracción)",
    options=["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"],
    default=["Barcelona", "Cambridge"],
)

# Radio para regresión (solo una)
ciudad_regresion = st.sidebar.radio(
    label="Ciudad para regresión",
    options=["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"],
    index=0,
)

###############################################################################
# 1) EXTRACCIÓN DE CARACTERÍSTICAS
###############################################################################
if View == "Extracción de Características":
    st.title("Extracción de Características")
    st.write("Análisis de características clave en los datos de Airbnb.")

    if not ciudades_multiselect:
        st.warning("Selecciona al menos una ciudad en la barra lateral 👈")
    else:
        # vamos ciudad por ciudad
        for c in ciudades_multiselect:
            # elegir df según ciudad
            if c == "Barcelona":
                df_actual = dfb
            elif c == "Cambridge":
                df_actual = dfc
            elif c == "Boston":
                df_actual = dfbo
            elif c == "Hawái":
                df_actual = dfh
            else:  # Budapest
                df_actual = dfbu

            st.subheader(f"Análisis de {c}")

            # gráfico de barrios si existe
            if "neighbourhood_cleansed" in df_actual.columns:
                st.write("Número de alojamientos por barrio:")
                barrio_counts = df_actual["neighbourhood_cleansed"].value_counts()
                st.bar_chart(barrio_counts)

            # columnas numéricas
            num_cols = df_actual.select_dtypes(include=["int64", "float64"]).columns.tolist()

            if not num_cols:
                st.info("Esta ciudad no tiene columnas numéricas para graficar.")
                continue

            # multiselect de variables numéricas para esta ciudad
            vars_sel = st.multiselect(
                f"Selecciona variables numéricas para graficar ({c})",
                options=num_cols,
                # si hay price lo dejamos fuera del default, si no, primera col
                default=[col for col in num_cols if col != "price"][:2] or [num_cols[0]],
                key=f"vars_{c}",
            )

            if not vars_sel:
                st.info(f"Selecciona al menos una variable para {c}.")
                continue

            tiene_price = "price" in df_actual.columns

            # graficamos cada variable
            for v in vars_sel:
                st.markdown(f"**Variable: `{v}`**")

                # 1. Histograma
                fig_hist = px.histogram(
                    df_actual,
                    x=v,
                    nbins=30,
                    title=f"Distribución de {v} en {c}",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # 2. Scatter vs price (si hay price y no es la misma)
                if tiene_price and v != "price":
                    tmp = df_actual[[v, "price"]].dropna()
                    if not tmp.empty:
                        fig_scatter = px.scatter(
                            tmp,
                            x=v,
                            y="price",
                            trendline="ols",
                            title=f"{v} vs Price en {c}",
                            labels={v: v, "price": "Price"},
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                # 3. Boxplot de price (una sola vez tendría sentido, pero lo dejamos por var)
                if tiene_price and v != "price":
                    fig_box = px.box(
                        df_actual,
                        y="price",
                        title=f"Distribución de Price en {c}",
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

###############################################################################
# 2) TABLAS COMPARATIVAS
###############################################################################
elif View == "Tablas comparativas":
    st.title("Tablas Comparativas")
    st.write("Comparación de estadísticas clave entre ciudades de Airbnb.")

    # multiselect de ciudades para comparar
    ciudades_sel = st.multiselect(
        "Selecciona las ciudades a comparar",
        options=["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"],
        default=["Barcelona", "Cambridge"],
    )

    dfs_ciudades = {
        "Barcelona": dfb,
        "Cambridge": dfc,
        "Boston": dfbo,
        "Hawái": dfh,
        "Budapest": dfbu,
    }

    tablas = []

    for ciudad in ciudades_sel:
        df_ciudad = dfs_ciudades[ciudad]
        stats = df_ciudad.describe().T[["mean", "50%", "std"]]
        prefijo = ciudad
        stats.columns = [
            f"Mean_{prefijo}",
            f"Median_{prefijo}",
            f"Std_{prefijo}",
        ]
        tablas.append(stats)

    if len(tablas) == 0:
        st.warning("Selecciona al menos una ciudad para comparar.")
    else:
        comparison_df = pd.concat(tablas, axis=1)
        st.dataframe(comparison_df)

###############################################################################
# 3) REGRESIÓN LINEAL
###############################################################################
elif View == "Regresión Lineal":
    st.title("Regresión Lineal")
    st.write("Análisis de regresión lineal para predecir precios.")

    # Elegimos el dataframe según la ciudad del radio
    if ciudad_regresion == "Barcelona":
        st.subheader("Regresión Lineal - Barcelona")
        df_reg = dfb
    elif ciudad_regresion == "Cambridge":
        st.subheader("Regresión Lineal - Cambridge")
        df_reg = dfc
    elif ciudad_regresion == "Boston":
        st.subheader("Regresión Lineal - Boston")
        df_reg = dfbo
    elif ciudad_regresion == "Hawái":
        st.subheader("Regresión Lineal - Hawái")
        df_reg = dfh
    else:  # Budapest
        st.subheader("Regresión Lineal - Budapest")
        df_reg = dfbu

    # Validamos columnas
    if "accommodates" not in df_reg.columns or "price" not in df_reg.columns:
        st.error("La ciudad seleccionada no tiene las columnas necesarias ('accommodates' y 'price').")
    else:
        X = df_reg[["accommodates"]]
        y = df_reg["price"].astype(float)

        # quitar filas con NaN
        valid = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid]
        y = y[valid]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        st.write(model.summary())

        y_pred = model.predict(sm.add_constant(X_test))
        r2 = r2_score(y_test, y_pred)
        st.write(f"R² Score: {r2}")

        fig = px.scatter(
            x=X_test["accommodates"],
            y=y_test,
            labels={"x": "Accommodates", "y": "Price"},
            title=f"Precio vs Accommodates - {ciudad_regresion}",
        )
        fig.add_trace(
            go.Scatter(
                x=X_test["accommodates"],
                y=y_pred,
                mode="lines",
                name="Predicted Price",
                line=dict(color="red"),
            )
        )
        st.plotly_chart(fig)
