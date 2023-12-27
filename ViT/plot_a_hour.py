import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
from collections import defaultdict

def plot_heatmap_on_earth_car(truth_np, pred_np, RECORDPATH, epoch, info):  # plot castleline with cartopy
        
    extent = (-180, 175, -87.5, 87.5)
    # 設置畫布大小
    fig, axes = plt.subplots(
        ncols=3,
        sharex=True,
        # sharey=True,
        figsize=(15, 4),
        # gridspec_kw=dict(width_ratios=[4,4,4,0.4,0.4]),
        subplot_kw={'projection': ccrs.PlateCarree()},
        )
    cbar_ax1 = fig.add_axes([.9, .3, .02, .4])
    cbar_ax2 = fig.add_axes([.95, .3, .02, .4])
    # 繪製heatmap
    for idx, (ax, data) in enumerate(zip(axes[:3], [truth_np, pred_np, abs(truth_np-pred_np)])):
        data = data.reshape((71, 72))
        lat = np.linspace(extent[3],extent[2],data.shape[0])
        lon = np.linspace(extent[0],extent[1],data.shape[1])
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.3, color='black')
        gl = ax.gridlines(draw_labels=True, color = "None", crs=ccrs.PlateCarree(),)
        gl.xlabel_style = dict(fontsize=6)
        gl.ylabel_style = dict(fontsize=6)
        gl.top_labels = False
        gl.right_labels = False
        gl.right_labels = True if idx == 0 else False

        Lat,Lon = np.meshgrid(lat,lon)
        # if idx == 0:
        #     print(Lon, Lat)

        cmap = plt.get_cmap('jet')
        cmap.set_under('white')
        cmap2 = plt.get_cmap('Greens')
        cmap2.set_under('white')
        ax.pcolormesh(Lon,Lat,
                      np.transpose(data),
                      vmin=0,
                      vmax=40 if idx != 2 else 5,
                      cmap=cmap if idx != 2 else cmap2,
                    #   cbar_ax = cbar_ax1 if idx == 0 else cbar_ax2 if idx == 2 else None,
                    #   cbar=idx in (0, 2),
                     )
        
        # ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 10)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
        for index, xlabel in enumerate(ax.get_xticklabels()):
            k = 10
            vis = index % k == 0
            xlabel.set_visible(vis)
        for index, ylabel in enumerate(ax.get_yticklabels()):
            k = 10
            vis = index % k == 0
            ylabel.set_visible(vis)
        
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('latitude')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel('longtitude')
        print('lens:',len(ax.get_xticklabels()), len(ax.get_yticklabels()))

    mae_loss = cal_mae_loss(truth_np, pred_np)
    rmse_loss = cal_rmse_loss(truth_np, pred_np)
    # fig.subplots_adjust(bottom=0, right=0.9, top=1)
    axes[0].set(title=f'Truth({int(info[0])}, {int(info[1])}, {int(info[2])})')
    axes[1].set(title='Prediction')
    axes[2].set(title=f'Difference  MAE:{mae_loss}, RMSE:{rmse_loss}')
    # fig.canvas.draw()
    fig.tight_layout(rect=[0, 0, .9, 1])
    # fig.tight_layout()
    fig.colorbar(axes[0].collections[0], cax=cbar_ax1)
    fig.colorbar(axes[2].collections[0], cax=cbar_ax2)
    # 儲存圖片
    
    fig.suptitle('GTEC MAP', fontsize=16)
    plt.savefig(RECORDPATH + f'prediction_truth_diff_{epoch}.jpg', dpi=1000)

def plot_accumulation_loss(data, count):
    extent = (-180, 175, -87.5, 87.5)
    # 設置畫布大小
    fig, axes = plt.subplots(
        ncols=1,
        sharex=True,
        # sharey=True,
        figsize=(15, 4),
        # gridspec_kw=dict(width_ratios=[4,4,4,0.4,0.4]),
        subplot_kw={'projection': ccrs.PlateCarree()},
        )
    cbar_ax2 = fig.add_axes([.95, .3, .02, .4])
    # 繪製heatmap
    data = data.reshape((71, 72))
    lat = np.linspace(extent[3],extent[2],data.shape[0])
    lon = np.linspace(extent[0],extent[1],data.shape[1])
    axes.set_extent(extent, crs=ccrs.PlateCarree())
    axes.coastlines(linewidth=0.3, color='black')
    gl = axes.gridlines(draw_labels=True, color = "None", crs=ccrs.PlateCarree(),)
    gl.xlabel_style = dict(fontsize=6)
    gl.ylabel_style = dict(fontsize=6)
    gl.top_labels = False
    gl.right_labels = False
    gl.right_labels = False

    Lat,Lon = np.meshgrid(lat,lon)
    print(count)
    cmap2 = plt.get_cmap('Greens')
    cmap2.set_under('white')
    axes.pcolormesh(Lon,Lat,
                    np.transpose(data),
                    vmin=0,
                    vmax=count*2,
                    cmap=cmap2,
                    )
    
    for index, xlabel in enumerate(axes.get_xticklabels()):
        k = 10
        vis = index % k == 0
        xlabel.set_visible(vis)
    for index, ylabel in enumerate(axes.get_yticklabels()):
        k = 10
        vis = index % k == 0
        ylabel.set_visible(vis)
    
    if axes.get_subplotspec().is_last_row():
        axes.set_xlabel('latitude')
    if axes.get_subplotspec().is_first_col():
        axes.set_ylabel('longtitude')
    print('lens:',len(axes.get_xticklabels()), len(axes.get_yticklabels()))

    axes.set(title=f'accumulation_loss')
    fig.tight_layout(rect=[0, 0, .9, 1])
    fig.colorbar(axes.collections[0], cax=cbar_ax2)
    # 儲存圖片
    
    # fig.suptitle('GTEC MAP', fontsize=16)
    plt.savefig('pictures/accumulation_loss.jpg', dpi=1000)

def cal_mae_loss(truth_np, pred_np):
    loss, count = 0, 0
    for t, p in zip(truth_np, pred_np):
        if t != -1 and p != -1:
            loss += abs(t-p)
            count += 1
    return round(loss /count, 2)

def cal_rmse_loss(truth_np, pred_np):
    diff = np.subtract(truth_np, pred_np)
    square = np.square(diff)
    mse = square.mean()
    rmse = np.sqrt(mse)
    return round(rmse, 2)

def process_data(data, pretrained, cal_all):
    info = data[:3]
    patch_size = int(data[3])
    patch_count = 72*72//(patch_size*patch_size)
    target_world = [[]for _ in range(patch_count)]

    if pretrained:
        temp = data[4][1:-1]
        if temp == '':
            mask = []
        else:
            mask = list(map(int, temp.split(',')))
        tec_data = data[5:]
    else:
        mask = [i for i in range(patch_count)]
        tec_data = data[4:]

    for longitude in range(71):
        for lat in range(72):
            patch_idx = (longitude//patch_size)*(72//patch_size) + lat//patch_size
            if cal_all:
                target_world[patch_idx].append(tec_data[longitude*72 + lat])
            else:
                if patch_idx in mask:
                    target_world[patch_idx].append(tec_data[longitude*72 + lat])
                else:
                    target_world[patch_idx].append(-1)

    tec_map = []
    for patch in range(0, len(target_world), 72//patch_size):
        for lat_idx in range(len(target_world[patch])//patch_size):
            for lon_idx in range(72//patch_size):
                tec_map += target_world[patch + lon_idx][lat_idx*patch_size:(lat_idx+1)*patch_size]
    return info, tec_map

def history_line(history_rmse, space_data, space_weather, year):
    fig, ax1 = plt.subplots()
    plt.title(f'{year} RMSE')
    plt.xlabel('hour')
    ax2 = ax1.twinx()

    ax1.set_ylabel('RMSE', color='red')
    ax1.plot(range(len(history_rmse)), history_rmse, color='red', alpha=0.75)
    ax1.tick_params(axis='y', labelcolor='red')

    ax2.set_ylabel(space_weather, color='skyblue')
    ax2.plot(range(len(space_data[12:])), space_data[12:], color='skyblue', alpha=1)
    ax2.tick_params(axis='y', labelcolor='skyblue')
    fig.tight_layout()
    plt.savefig(f'pictures/{year}_RMSE_{space_weather}_acc.jpg', dpi=1000)
    # plt.show()

def process_space_data(path):
    space_weather_data = defaultdict(list)
    for file_name in os.listdir(path):
        year = file_name.split('.')[0]
        data = pd.read_csv(os.path.join(path, file_name), skiprows= 5, usecols = [4, 5, 6, 7, 8])
        data = data.dropna(axis=0, how='any')
        data = data.values
    space_weather_data['Kp'] = np.array(data[:, 0]).flatten()
    space_weather_data['R'] = np.array(data[:, 1]).flatten()
    space_weather_data['Dst'] = np.array(data[:, 2]).flatten()
    space_weather_data['Ap'] = np.array(data[:, 3]).flatten()
    space_weather_data['f10.7'] = np.array(data[:, 4]).flatten()
    return year, space_weather_data

def main(args):
    dataset = pd.read_csv(f'{args.file}.csv', header=list(range(2))).reset_index(drop=True)
    if dataset.columns[4][0] == 'mask':
        pretrained = True
    else:
        pretrained = False
    
    year, space_weather_data = process_space_data('../data/test')
    
    accumulation_loss = [0 for _ in range(5112)]
    history_rmse, history_day = [], []
    flag = False
    count = 0
    for i in tqdm(range(0, len(dataset), 2)):
        p_info, pred_sr = process_data(dataset.values[i], pretrained, args.cal_all)
        t_info, truth_sr = process_data(dataset.values[i+1], pretrained, args.cal_all)
        # plot_heatmap_on_earth_car(np.array(truth_sr), np.array(pred_sr), args.record, 0, p_info)
        rmse_loss = cal_rmse_loss(np.array(truth_sr), np.array(pred_sr))
        history_rmse.append(rmse_loss)
        history_day.append(i//2)
        # accumulation_loss += abs(np.array(truth_sr)-np.array(pred_sr))
        # count += 1
        if i == 20:
            break
        # input()
        try:
            if int(year) != int(p_info[0]):
                raise ValueError(f'test data({year}年) 的檔案與 predict({int(p_info[0])}年) 的檔案年份不一致')
            flag = True
        except ValueError as ve:
            print(f'異常: {ve}')
            break
    if flag and not pretrained:
        for space_weather, space_data in space_weather_data.items():
            history_line(history_rmse, space_data, space_weather, year)
    # plot_accumulation_loss(accumulation_loss, count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='predict')
    parser.add_argument('-r', '--record', type=str, default='./pictures/')
    parser.add_argument('-ca', '--cal_all', type=bool, default=False)

    args = parser.parse_args()
    if not os.path.isdir(args.record):
        os.mkdir(args.record)
    main(args)