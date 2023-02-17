# GITM
## File Directory
+ 需要額外下載data裡面的資料:https://drive.google.com/file/d/1yMob-Mp87ORs56fp61rmt5KfEFz9zLNd/view?usp=share_link
```
|--- data
|   |--- 2018.csv
|   |--- 2019.csv
|   |--- 2020.csv
|   |--- 2021.csv
|
|--- ViT
|   |--- dataloader.py
|   |--- inference.py
|   |--- main.py
|   |--- model.py
|   |--- train_model.py
|   |--- save_model
|
|--- Swim Transformer
|   |--- dataloader.py
|   |--- inference.py
|   |--- main.py
|   |--- model.py
|   |--- train_model.py
|   |--- save_model
```

## 訓練
### 指令註釋
` python ViT/main.py -e 訓練幾次 -b 批次數 -P 看子圖片的大小 `

` python "Swim Transformer"/main.py -e 訓練幾次 -b 批次數 -P 看子圖片的大小 `
