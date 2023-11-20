# ViT

## 檔案功能
### 訓練指令
Fine-tune 設定 -ma ，代表模型要 load 哪一種 pretrained model 的 weight
```
python main.py [-e epoch] [-b batch_size] [-p patch_size] [-t target_hour] [-i input_history] [-m mode] [-pt pretrained] [-tm test_mode] [-ma mask_ratio] [-mt mask_type]

optional arguments:
-e 訓練次數
-b 批次數
-p 子圖片的大小
-t 目標時間
-i 輸入時間長度
-m 資料處理要使用何種方式(minmax/zscores/None)
-pt 此模型是否是pretrained
-ma 是否為測試階段
-tm 遮住圖片的比例(load 哪種 pretrained model 的 weight)
-mt 遮住圖片的方式為何(random : 隨機, column : 以經度的方式, block : 以區塊的方式)
```

### 復現圖片
```
python plot_a_hourain.py -f 讀取預測結果的檔案 -r 儲存位置 -ca 是否計算整張圖片
``` 

### 重複測試同一張圖片
```
python test_loss.py -f 當前目錄儲存結果的檔案 -t 要哪一個時間段 -e 重複測試次數 -r 圖片儲存位置 -p 子圖片大小 -pt 此模型是否是pretrained (-ma 遮住圖片的比例)
```
