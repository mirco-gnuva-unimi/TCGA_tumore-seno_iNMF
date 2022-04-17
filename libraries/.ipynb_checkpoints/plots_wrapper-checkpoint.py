import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pandas import DataFrame
import os
from datetime import datetime


def merge_labels(df, labels):
    df = df.merge(labels, how='inner', left_index=True, right_index=True)
    df['PFI'] = df['PFI'].astype(str)
    df['DFI'] = df['DFI'].astype(str)

    return df


def plot_title(pipeline: str, silhouette: float, homogeneity: float, rf_report: dict, datasets: dict) -> str:
    rf_report['f_score'] = [f'{cls}: {value:.3f}' for cls, value in rf_report['f_score'].items()]
    report_str = ' | '.join([f'{metric.capitalize()} {value:.3f}' if isinstance(value, float) else f'{metric} {value}' for metric, value in rf_report.items()])
    datasets_string = ','.join(datasets.keys())
    title = f'{pipeline}<br><sup>Silhouette {silhouette:.3f} | Homogeneity {homogeneity:.3f} | {report_str}</sup><br><sup>Datasets: {datasets_string}</sup>'

    return title


def get_iplot(integrated_df: DataFrame, labels: DataFrame = None, title: str = None, hover_data: list = None):
    assert (isinstance(integrated_df, DataFrame))
    assert (isinstance(labels, DataFrame) or labels is None)
    assert (isinstance(title, str) or title is None)

    n_components = len(integrated_df.columns)
    integrated_df = merge_labels(integrated_df, labels)

    fig_class = px.scatter if n_components == 2 else px.scatter_3d

    fig = fig_class(integrated_df, x=0, y=1, z=2, color='PFI', width=1000, height=1000, title=title, hover_data=hover_data)
    fig.update_traces(marker_size=3)

    return fig


def create_dir(root, info, timestring):
    dir_name = f'{info}_{timestring}'

    path = os.path.join(root, dir_name)
    if os.path.isdir(path):
        alt = 0
        while os.path.isdir(os.path.join(root, f'{dir_name}_{alt}')):
            alt += 1

        path = os.path.join(root, f'{dir_name}_{alt}')

    os.mkdir(path)

    return path


def save_fig(fig, root, info, subfolder: bool = False):
    now = datetime.now()
    timestring = now.strftime("%d-%m-%Y_%H-%M-%S")

    fig_name = f'{info}_{timestring}.png'

    if subfolder:
        result_dir = create_dir(root, info, timestring)
        fig_path = os.path.join(root, result_dir, fig_name)
    else:
        fig_path = os.path.join(root, fig_name)

    fig.write_image(fig_path)


def plot_error_vs_silhoutte(results, parameter_name, title):
    parameter_header = f'{parameter_name.capitalize()}'
    df = DataFrame(results, columns=[parameter_header, 'Error', 'Silhoulette'])
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Error", "Silhoulette"))
    fig.append_trace(go.Scatter(x=df[parameter_header], y=df['Error']), row=1, col=1)
    fig.append_trace(go.Scatter(x=df[parameter_header], y=df['Silhoulette']), row=2, col=1)
    fig.update_layout(height=800, showlegend=False, title_text=title)
    fig.show()
