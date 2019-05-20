import pathlib
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Union

import pandas as pd


def load_data(folders: Iterable[str] = ('rec2', 'rec3', 'rec4', 'rec5', 'rec6', 'rec7', 'rec8', 'rec9', 'rec10')) -> pd.DataFrame:
    path = pathlib.Path('./recordings/')

    all_frames = []
    col_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    for folder in folders:
        file_name = path / folder / 'driving_log.csv'
        file_data = pd.read_csv(file_name, header=None, names=col_names)
        file_data['source'] = folder
        all_frames.append(file_data)

    return pd.concat(all_frames)


def plot_and_save(data: pd.DataFrame, name: str, fig_folder: Optional[pathlib.Path] = None) -> None:
    """Plot an histogram of data.
    If `fig_folder` is `None`, display it on the screen, otherwise save it in `fig_folder`.
    """
    plt.figure()
    plt.hist(data.steering)
    plt.grid(True)
    plt.title(f'Data for recording {name}')
    if fig_folder is None:
        plt.show()
    else:
        plt.savefig(fig_folder / f'{name}_hist.png')


def analyse_data(data: pd.DataFrame, fig_folder: Optional[Union[str, pathlib.Path]] = None) -> Iterable[float]:
    """Generate some histograms about the data and save it to a fig_folder.
    If `fig_folder` is `None`, no file is saved."""
    if fig_folder is not None:
        fig_folder = pathlib.Path(fig_folder)

    for rec in set(data.source):
        subset = data[data.source == rec]
        plot_and_save(subset, rec, fig_folder)
    plot_and_save(data, 'all', fig_folder)


