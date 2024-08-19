# GITM
## Fine-tuned 結果

![](./ViT/prediction_truth_diff_0.jpg)

## 安裝套件 (requirements)
```
cd GITM
pip install -r requirements.txt 
```

## 下載資料 
```
cd GITM
gdown https://drive.google.com/uc?id=1YHLb72Y8UdSAyclbujPF3UjLF4IYZ39d
unzip data.zip
```

## ~~下載 ViT 預訓練好的權重檔~~
~~要是參數量不一致，可能會導致模型無法正確 load 進去，這時就需要重新訓練模型~~
這步跳過
```
cd GITM/ViT/save_model
gdown 1th8IrF5k1O3yDPh8-4WiNbURG27xWpAJ
unzip save_model.zip
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
