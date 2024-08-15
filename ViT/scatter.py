import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
from tqdm import tqdm
from collections import defaultdict

def process_space_data(path):
    space_weather_data = defaultdict(list)
    dataset= []
    date = []
    for file_name in os.listdir(path):
        year = file_name.split('.')[0]
        data = pd.read_csv(os.path.join(path, file_name), skiprows= 5, usecols = [4, 5, 6, 7, 8])
        data1 = pd.read_csv(os.path.join(path, file_name), skiprows= 5, usecols = [1,2,3])
        data = data.dropna(axis=0, how='any')
        data1 = data1.dropna(axis=0, how='any')
        data = data.values
        data1 = data1.values
        dataset.append(data)
        date.append(data1)
    tec_data = dataset[0]
    date_data = date[0]
    for i in range(1, len(dataset)):
        tec_data = np.vstack((tec_data,dataset[i]))
        #date_data = np.vstack((date_data,date[i]))
        
    space_weather_data['Kp'] = np.array(tec_data[:, 0]).flatten()
    space_weather_data['R'] = np.array(tec_data[:, 1]).flatten()
    space_weather_data['Dst'] = np.array(tec_data[:, 2]).flatten()
    space_weather_data['Ap'] = np.array(tec_data[:, 3]).flatten()
    space_weather_data['f10.7'] = np.array(tec_data[:, 4]).flatten()
    return year, space_weather_data , date

def history_line(history_rmse, space_data, space_weather, year):
    fig, ax1 = plt.subplots()
    plt.title(f'2020-2021 RMSE')
    plt.xlabel('hour')
    ax2 = ax1.twinx()

    ax1.set_ylabel('RMSE', color='red')
    ax1.plot(range(len(history_rmse)), history_rmse, color='red', alpha=0.75)
    ax1.tick_params(axis='y', labelcolor='red')

    ax2.set_ylabel(space_weather, color='skyblue')
    ax2.plot(range(len(space_data[263:])), space_data[263:], color='skyblue', alpha=1)
    ax2.tick_params(axis='y', labelcolor='skyblue')
    fig.tight_layout()
    plt.savefig(f'pictures/2020-2021_RMSE_{space_weather}_acc_v2.jpg', dpi=1000)
    # plt.show()

def scatter_plot(history_rmse, space_data, space_weather, year):
    fig = plt.subplots()
    plt.scatter(history_rmse, space_data[263:])
    plt.title('2020-2021 RMSE')
    plt.ylabel(space_weather, color='skyblue')
    plt.xlabel('RMSE', color='red')
    plt.savefig(f'pictures/2020-2021_RMSE_{space_weather}_sactter_acc.jpg', dpi=1000)

def correlation(history_rmse, space_data, space_weather, year):
    df = pd.DataFrame({'x':history_rmse, 'y':space_data[263:]})
    correlation = df['x'].corr(df['y'])
    print(f"Pearson correlation coefficient of {space_weather} :", correlation)

def mkcsv(history_rmse, space_data, space_weather, date_data):
    df = pd.DataFrame({'Year':date_data[263:,0],'day':date_data[263:,1],'hour':date_data[263:,2],'RMSE':history_rmse, space_weather:space_data[263:]})
    df.to_csv(f'./2020-2021_RMSE_{space_weather}.csv', index=False)

def main(args):
    dataset = pd.read_csv(f'{args.file}.csv')
    history_rmse = dataset.iloc[:,1]
    #print(history_rmse)
    year, space_weather_data, date_data = process_space_data('../data/test')
    print(date_data)
    #print(len(space_weather_data['R'][263:]))

    for space_weather, space_data in space_weather_data.items():
        #history_line(history_rmse, space_data, space_weather, year)
        #scatter_plot(history_rmse, space_data, space_weather, year)
        #correlation(history_rmse, space_data, space_weather, year)
        mkcsv(history_rmse, space_data, space_weather, date_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='predict')
    parser.add_argument('-r', '--record', type=str, default='./pictures/')
    parser.add_argument('-ca', '--cal_all', type=bool, default=False)

    args = parser.parse_args()
    if not os.path.isdir(args.record):
        os.mkdir(args.record)
    main(args)