# A-lightweight-Unsupervised-Feature-Extractor
## 簡介（）
本模型透過ROI技術以當下的偵測框擷取YOLO骨幹網路中的深層特徵，其中以骨幹網路最後一層的SPPCSPC作為唯一的特徵來源，並將其以邊界框切分形成數個不同的片段，最終將各個不同的片段重塑為(10,10)的形狀方便後續的訓練與推論。

