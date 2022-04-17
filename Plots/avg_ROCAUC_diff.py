import pandas as pd
import plotly.express as px

data = pd.read_csv('C:/Users/mirco/Desktop/Tesi/Shared/Cloud/Results/Methods_Summary - confronto_avg_processi.csv')
data = data.rename(columns={'Distanza massima': 'Differenza di ROC AUC'})


fig = px.bar(data[data['Metodo'] != 'No'], x='Processo', y='Differenza di ROC AUC', color='Metodo', barmode="group")
fig.add_shape(type='line',
              y0=data['Differenza di ROC AUC'].iloc[0], x0=-0.5, y1=data['Differenza di ROC AUC'].iloc[0], x1=len(data), line=dict(color='Grey', ), xref='x',
              yref='y')
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.90,
    bgcolor='white'
))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_layout(font_size=30)



fig.show()
exit()

# fig = px.bar(melted, color='Pesatura', x='Test', y='Distanza', barmode="group")
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

melted = None
fig = px.bar(melted, color='Pesatura', y='Processi ricampionamento', x='Distanza', barmode="group", orientation='h',
             height=2000, width=1500)
fig.add_shape(type='line',
              x0=melted['Distanza'].mean(), y0=0, x1=melted['Distanza'].mean(), y1=len(data), line=dict(color='Grey', ),
              xref='x', yref='y', layer='below')
fig.update_yaxes(showticklabels=False)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.90
))

fig.update_layout(font_size=30)
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()

bal_better = data[data['Pesato'] > data['Non pesato']]
print(f'Bal > Non Bal: {len(bal_better)} ({len(bal_better) * 100 / len(data)}%)')

non_bal_better = data[data['Pesato'] < data['Non pesato']]
print(f'Non Bal > Bal: {len(non_bal_better)} ({len(non_bal_better) * 100 / len(data)}%)')

equal = data[data['Pesato'] == data['Non pesato']]
print(f'Non Bal = Bal: {len(equal)} ({len(equal) * 100 / len(data)}%)')

avg = melted['Distanza'].mean()
print(f'Media distanza: {avg}')
over_avg = melted[melted['Distanza'] > avg]
print(f'Distanze sopra la media: {len(over_avg)}')

bal_over_avg = over_avg[over_avg['Pesatura'] == 'Pesato']
print(f'Pesati sopra la media: {len(bal_over_avg)}')

non_bal_over_avg = over_avg[over_avg['Pesatura'] == 'Non pesato']
print(f'Non pesati sopra la media: {len(non_bal_over_avg)}')
