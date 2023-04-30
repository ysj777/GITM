# GITM
## File Directory
+ 需要額外下載data裡面的資料:https://drive.google.com/file/d/1yMob-Mp87ORs56fp61rmt5KfEFz9zLNd/view?usp=share_link
```
|--- data
|   |--- train
|      |--- 2018.csv
|      |--- 2019.csv
|   |--- valid
|      |--- 2020.csv
|   |--- test
|      |--- 2021.csv
|
|--- ViT
|   |--- dataloader.py
|   |--- inference.py
|   |--- main.py
|   |--- model.py
|   |--- train_model.py
|   |--- save_model
|
|--- SwinT
|   |--- dataloader.py
|   |--- inference.py
|   |--- main.py
|   |--- model.py
|   |--- train_model.py
|   |--- save_model
```

## 訓練
### 指令註釋
` python main.py -e 訓練幾次 -b 批次數 -p 子圖片的大小 -t 目標時間 -i 輸入時間長度 -m minmax/zscores/None -pt 此模型是否是pretrain`