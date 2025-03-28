import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from model import entrenar_modelo_arima
from preprocessing import crear_features


# Cargar y procesar datos
df = pd.read_csv('Proyecto_final.csv', parse_dates=['Mes'])
df = crear_features(df)
df.set_index('Mes', inplace=True)  # Establecer 'Mes' como índice
df.index.freq = 'M'  # Establecer frecuencia explícita

print(df.columns)

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server  # Necesario para despliegue en servidores

app.layout = html.Div([
    html.Div([
        html.H1("Análisis de Ventas - Modelo Predictivo", className="header-title"),
        html.P("Dashboard interactivo con análisis estadístico y pronósticos", className="header-description"),
    ], className="header"),
    
    html.Div([
        html.Div([
            html.Label("Seleccionar Producto:"),
            dcc.Dropdown(
                id='producto-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns[1:4]],
                value=df.columns[1]  # Valor predeterminado
            ),
        ], className="six columns"),
        
        html.Div([
            html.Label("Ajustar Parámetro α (Ridge):"),
            dcc.Slider(id='alpha-slider', min=0, max=10, step=0.5, value=1.0),
        ], className="six columns"),
    ], className="row"),
    
    html.Div([
        dcc.Graph(id='time-series-plot'),
        dcc.Graph(id='histogram-plot'),
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("Métricas Clave"),
            dash_table.DataTable(
                id='metrics-table',
                columns=[{"name": i, "id": i} for i in ['Métrica', 'Valor']],
                style_cell={'textAlign': 'center'},
            ),
        ], className="six columns"),
        
        html.Div([
            html.H3("Intervalos de Confianza (95%)"),
            dcc.Graph(id='confidence-intervals'),
        ], className="six columns"),
    ], className="row"),
    
    html.Div([
        html.H3("Pronóstico ARIMA (3 meses)"),
        dcc.Graph(id='forecast-plot'),
    ]),
])

@app.callback(
    [Output('time-series-plot', 'figure'),
     Output('histogram-plot', 'figure'),
     Output('confidence-intervals', 'figure'),
     Output('metrics-table', 'data'),
     Output('forecast-plot', 'figure')],
    [Input('producto-dropdown', 'value'),
     Input('alpha-slider', 'value')]
)
def update_dashboard(producto, alpha):
    if producto is None:
        return {}, {}, {}, [], {}  # Manejo de errores si no hay producto seleccionado

    # Gráfico de serie temporal
    ts_fig = px.line(df, x=df.index, y=producto, title=f'Tendencia de {producto}')
    
    # Histograma con curva de densidad
    hist_fig = px.histogram(df, x=producto, marginal="rug", title=f'Distribución de {producto}')
    
    # Intervalos de confianza
    conf_fig = px.box(df, y=producto, title=f'Variabilidad de {producto}')
    
    # Métricas estadísticas
    stats_data = {
        'Métrica': ['Media', 'Error Estándar', 'Coef. Variación'],
        'Valor': [f"{df[producto].mean():.2f}", 
                  f"{df[producto].sem():.2f}", 
                  f"{(df[producto].std()/df[producto].mean())*100:.2f}%"]
    }
    
    # Pronóstico ARIMA
    model = entrenar_modelo_arima(df[producto])
    forecast = model.get_forecast(steps=3)
    pred_df = pd.DataFrame({
        'Mes': pd.date_range(df.index[-1], periods=4, freq='ME')[1:],  # Cambiar 'M' por 'ME'
        'Pronóstico': forecast.predicted_mean,
        'Intervalo Inferior': forecast.conf_int().iloc[:, 0],
        'Intervalo Superior': forecast.conf_int().iloc[:, 1]
    })
    forecast_fig = px.line(pred_df, x='Mes', y='Pronóstico', 
                          title='Pronóstico con Intervalos de Confianza')
    forecast_fig.add_scatter(x=pred_df['Mes'], y=pred_df['Intervalo Inferior'], 
                            mode='lines', line=dict(color='gray'), name='Límite Inferior')
    forecast_fig.add_scatter(x=pred_df['Mes'], y=pred_df['Intervalo Superior'], 
                            mode='lines', line=dict(color='gray'), name='Límite Superior')
    
    return ts_fig, hist_fig, conf_fig, stats_data, forecast_fig

if __name__ == '__main__':
    app.run(debug=False)

# Removed unnecessary JSON-like structure at the end of the file.
