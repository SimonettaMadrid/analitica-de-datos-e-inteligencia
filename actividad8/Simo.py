### simo.py
import pandas as pd
import numpy as np
import os

# ============================================================
# a) Carga de archivos (csv, xlsx, html y json) a DataFrames
# ============================================================
def cargar_dataset(ruta_archivo):
    """
    Carga un archivo en un DataFrame de pandas, detectando el tipo por su extensión.
    Soporta csv, xlsx, json y html.

    Parámetros:
    ruta_archivo (str): ruta y nombre del archivo.

    Retorna:
    pd.DataFrame
    """
    extension = os.path.splitext(ruta_archivo)[1].lower()

    if extension == ".csv":
        return pd.read_csv(ruta_archivo)
    elif extension in [".xlsx", ".xls"]:
        return pd.read_excel(ruta_archivo)
    elif extension == ".json":
        return pd.read_json(ruta_archivo)
    elif extension == ".html":
        tablas = pd.read_html(ruta_archivo)
        return tablas[0]  # Devuelve la primera tabla encontrada
    else:
        raise ValueError("Formato no soportado. Usa csv, xlsx, json o html.")
    # ============================================================
# b) Sustitución de valores nulos con ffill
# ============================================================
def nulos_ffill(df):
    """
    Rellena valores nulos hacia adelante (ffill).
    """
    return df.fillna(method="ffill")

# ============================================================
# c) Sustitución de valores nulos con bfill
# ============================================================
def nulos_bfill(df):
    """
    Rellena valores nulos hacia atrás (bfill).
    """
    return df.fillna(method="bfill")

# ============================================================
# d) Sustitución de nulos en variables string por un valor fijo
# ============================================================
def nulos_string(df, valor="Desconocido"):
    """
    Rellena nulos en columnas de tipo string con un valor específico.
    """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include="object").columns:
        df_copy[col] = df_copy[col].fillna(valor)
    return df_copy

# ============================================================
# e) Sustitución de nulos por promedio
# ============================================================
def nulos_prom_df(df):
    """
    Rellena nulos en columnas numéricas usando el promedio.
    """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include="number").columns:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    return df_copy

# ============================================================
# f) Sustitución de nulos por mediana
# ============================================================
def nulos_median_df(df):
    """
    Rellena nulos en columnas numéricas usando la mediana.
    """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include="number").columns:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    return df_copy

# ============================================================
# g) Sustitución de nulos por constante
# ============================================================
def nulos_constante_df(df, constante=0):
    """
    Rellena nulos en columnas numéricas con un valor constante.
    """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include="number").columns:
        df_copy[col] = df_copy[col].fillna(constante)
    return df_copy

# ============================================================
# i) Identificación de valores nulos
# ============================================================
def identificar_nulos(df):
    """
    Devuelve un DataFrame con el número de nulos por columna y el total.
    """
    nulos_por_columna = df.isnull().sum()
    total_nulos = df.isnull().sum().sum()
    return pd.DataFrame({
        "Nulos_por_columna": nulos_por_columna,
        "Total_nulos": [total_nulos] + [""] * (len(nulos_por_columna) - 1)
    })
# ============================================================


    