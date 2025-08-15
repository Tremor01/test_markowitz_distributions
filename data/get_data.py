from pandas import read_excel
from pathlib import Path


def get_prices(filename='prices_binance.xlsx'):
    file_path = Path(__file__).parent / filename
    df = read_excel(file_path)
    df = df.set_index('Unnamed: 0')
    df.index.name = 'Date'
    return df


def get_volumes(filename='volumes_binance.xlsx'):
    file_path = Path(__file__).parent / filename
    df = read_excel(file_path)
    df = df.set_index('Unnamed: 0')
    df.index.name = 'Date'
    return df

