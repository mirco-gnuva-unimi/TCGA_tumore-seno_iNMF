import plotly.express as px
import pandas as pd

data = pd.read_csv('C:/Users/mirco/Desktop/Tesi/Shared/Cloud/Results/Methods_Summary - confronto_integrazioni.csv')
data = data.rename(columns={'Distanza massima': 'Differenza di ROC AUC', 'Integrazione': 'Processo di integrazione'})


fig = px.box(data, x="Processo di integrazione", y="Differenza di ROC AUC", color='Processo di integrazione', boxmode="overlay")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='LightGrey')
fig.update_layout(font_size=30)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.85
))
fig.show()
