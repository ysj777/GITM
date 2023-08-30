# GITM

## 安裝套件 (requirements)
```
cd GITM
pip install -r requirements.txt 
```

## 下載資料 
```
cd GITM
gdown 1zYsnxAEThTpBszQ3yKt4_LvQXnKzs-cw
unzip data.zip
```

## 下載 ViT 預訓練好的權重檔 
```
cd GITM/ViT/save_model
gdown 1th8IrF5k1O3yDPh8-4WiNbURG27xWpAJ
unzip save_model.zip
```

## 訓練
### 指令註釋
``` 
python main.py -e 訓練幾次 -b 批次數 -p 子圖片的大小 -t 目標時間 -i 輸入時間長度 -m minmax/zscores/None -pt 此模型是否是pretrain
```

## File Structure
```
|--- data
|   |---pretrained
|       |--- train
|           |--- 2018.csv
|           |--- 2019.csv
|           |--- 2020.csv
|           |--- 2021.csv
|       |--- valid
|       |--- test
|   |--- train
|       |--- 2018.csv
|       |--- 2019.csv
|   |--- valid
|       |--- 2020.csv
|   |--- test
|       |--- 2021.csv
|
|--- ViT
|   |--- dataloader.py
|   |--- inference.py
|   |--- main.py
|   |--- model.py
|   |--- train_model.py
|   |--- plot_a_hour.py
|   |--- save_model
|       |--- patch_4_mask_ratio_1
|           |--- pretrained_model.pth
|       |--- patch_4_mask_ratio_3
|           |--- pretrained_model.pth
|       |--- patch_4_mask_ratio_5
|           |--- pretrained_model.pth
|       |--- patch_4_mask_ratio_7
|           |--- pretrained_model.pth
|
|--- SwinT
|   |--- dataloader.py
|   |--- inference.py
|   |--- main.py
|   |--- model.py
|   |--- train_model.py
|   |--- save_model
|--- requirements.txt
|--- README.md
```
