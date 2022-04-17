import pandas as pd
import plotly.express as px

data = pd.read_csv('C:/Users/mirco/Desktop/Tesi/Shared/Cloud/Results/Methods_Summary_-_Balanced_vs_Unbalanced.csv')
data['Processi ricampionamento'] = [str(i) for i in range(len(data))]
data['Pesato'] = data['Pesato'].apply(lambda x: x.replace(',', '.'))
data['Pesato'] = pd.to_numeric(data['Pesato'])
data['Non pesato'] = data['Non pesato'].apply(lambda x: x.replace(',', '.'))
data['Non pesato'] = pd.to_numeric(data['Non pesato'])
data = data.rename(columns={'Processi ricampionamento': 'Processi di integrazione e ricampionamento'})

melted = pd.melt(data, value_vars=['Pesato', 'Non pesato'], var_name='Pesatura', value_name='Differenza di ROC AUC', id_vars=['Processi di integrazione e ricampionamento'])
#melted['Differenza di ROC AUC'] = melted['Differenza di ROC AUC'].apply(lambda x: x.replace(',', '.'))
#melted['Differenza di ROC AUC'] = pd.to_numeric(melted['Differenza di ROC AUC'])

# fig = px.bar(melted, color='Pesatura', x='Test', y='Differenza di ROC AUC', barmode="group")
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig = px.bar(melted, color='Pesatura', y='Processi di integrazione e ricampionamento', x='Differenza di ROC AUC', barmode="group", orientation='h', height=2000, width=1500)
fig.add_shape(type='line',
                x0=melted['Differenza di ROC AUC'].mean(), y0=0, x1=melted['Differenza di ROC AUC'].mean(), y1=len(data), line=dict(color='Grey',), xref='x', yref='y', layer='below')
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
print(f'Bal > Non Bal: {len(bal_better)} ({len(bal_better) * 100/len(data)}%)')

non_bal_better = data[data['Pesato'] < data['Non pesato']]
print(f'Non Bal > Bal: {len(non_bal_better)} ({len(non_bal_better) * 100/len(data)}%)')

equal = data[data['Pesato'] == data['Non pesato']]
print(f'Non Bal = Bal: {len(equal)} ({len(equal) * 100/len(data)}%)')

avg = melted['Differenza di ROC AUC'].mean()
print(f'Media Differenza di ROC AUC: {avg}')
over_avg = melted[melted['Differenza di ROC AUC'] > avg]
print(f'Distanze sopra la media: {len(over_avg)}')

bal_over_avg = over_avg[over_avg['Pesatura'] == 'Pesato']
print(f'Pesati sopra la media: {len(bal_over_avg)}')

non_bal_over_avg = over_avg[over_avg['Pesatura'] == 'Non pesato']
print(f'Non pesati sopra la media: {len(non_bal_over_avg)}')
