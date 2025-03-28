import pandas as pd

def cargar_datos(ruta_csv):
    return pd.read_csv(ruta_csv, parse_dates=['Mes'])

def crear_features(df):
    df['ventas_1oz_lag1'] = df['1 oz'].shift(1)
    df['ventas_2oz_rolling3'] = df['2 oz'].rolling(window=3).mean()
    df['trimestre'] = df['Mes'].dt.quarter
    return df.dropna()

if __name__ == "__main__":
    df = crear_features(cargar_datos("data/Proyecto_final_reducido.csv"))
    print(df.head())



