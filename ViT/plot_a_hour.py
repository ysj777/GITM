import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs

def plot_heatmap_on_earth_car(truth_np, pred_np, RECORDPATH, epoch):  # plot castleline with cartopy
        
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
        if idx == 0:
            print(Lon, Lat)

        cmap = cm.get_cmap('jet').copy()
        cmap.set_under('white')
        cmap2 = cm.get_cmap('Greens').copy()
        cmap2.set_under('white')
        ax.pcolormesh(Lon,Lat,
                      np.transpose(data),
                      vmin=0,
                      vmax=30 if idx != 2 else 5,
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

    diff_loss = cal_loss(truth_np, pred_np)
    # fig.subplots_adjust(bottom=0, right=0.9, top=1)
    axes[0].set(title='Truth')
    axes[1].set(title='Prediction')
    axes[2].set(title=f'Difference  {diff_loss}')
    # fig.canvas.draw()
    fig.tight_layout(rect=[0, 0, .9, 1])
    # fig.tight_layout()
    fig.colorbar(axes[0].collections[0], cax=cbar_ax1)
    fig.colorbar(axes[2].collections[0], cax=cbar_ax2)
    # 儲存圖片
    
    fig.suptitle('GTEC MAP', fontsize=16)
    plt.savefig(RECORDPATH + f'prediction_truth_diff_{epoch}.jpg', dpi=1000)

def cal_loss(truth_np, pred_np):
    loss, count = 0, 0
    for t, p in zip(truth_np, pred_np):
        if t != -1 and p != -1:
            loss += abs(t-p)
            count += 1
    return round(loss /count, 2)

def process_data(data):
    patch_size = data[0]
    temp = data[1][1:-1]
    mask = list(map(int, temp.split(',')))
    tec_data = data[2:]
    patch_count = 72*72//(patch_size*patch_size)
    target_world = [[]for _ in range(patch_count)]

    for longitude in range(71):
        for lat in range(72):
            patch_idx = (longitude//patch_size)*(72//patch_size) + lat//patch_size
            if patch_idx in mask:
                target_world[patch_idx].append(tec_data[longitude*72 + lat])
            else:
                target_world[patch_idx].append(-1)

    tec_map = []
    for patch in range(0, len(target_world), 72//patch_size):
        for lat_idx in range(len(target_world[patch])//patch_size):
            for lon_idx in range(72//patch_size):
                tec_map += target_world[patch + lon_idx][lat_idx*patch_size:(lat_idx+1)*patch_size]
    return tec_map

def main(args):
    
    RECORDPATH = Path(args.record)
    
    dataset = pd.read_csv(f'{args.file}.csv', header=list(range(2))).reset_index(drop=True)
    
    for i in range(0, len(dataset), 2):
        pred_sr = process_data(dataset.values[i])
        truth_sr = process_data(dataset.values[i+1])
        plot_heatmap_on_earth_car(np.array(truth_sr), np.array(pred_sr), RECORDPATH, 0)
        input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='predict')
    parser.add_argument('-r', '--record', type=str, default='./')
    main(parser.parse_args())