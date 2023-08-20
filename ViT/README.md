# ViT

## 檔案功能
### 訓練指令 (括弧內為 pretrained 專用)
` python main.py -e 訓練幾次 -b 批次數 -p 子圖片的大小 -t 目標時間 -i 輸入時間長度 -m minmax/zscores/None -pt 此模型是否是pretrained -tm 是否為測試階段 (-ma 遮住圖片的比例)`

### 復現圖片
` python plot_a_hourain.py -f 當前目錄儲存結果的檔案 -r 儲存位置`

### 重複測試同一張圖片
` python test_loss.py -f 當前目錄儲存結果的檔案 -t 要哪一個時間段 -e 重複測試次數 -r 圖片儲存位置 -p 子圖片大小 -pt 此模型是否是pretrained (-ma 遮住圖片的比例)  `
