# A-lightweight-Unsupervised-Feature-Extractor
## 簡介（）
本模型透過ROI技術以當下的偵測框切分YOLO骨幹網路中的深層特徵訓練而成，以骨幹網路最後一層的 `SPP-CSPC` 作為唯一的特徵來源，並將其以邊界框切分後形成數個不同的片段，最終將各個不同的片段重塑為(10,10)的形狀方便後續的訓練與推論。

作為一個 Encoder 透過接收形狀為 `[1, 512, Hf, Wf]` 的張量進行訓練與推論,透過 ROI Align 將對應區域的特徵裁剪與重採樣為固定大小之 ROI 特徵 並依據當下偵測框的數量將特徵形狀重塑為 `[N, 512, 10, 10]`，其中 
𝑁 為該影像中偵測框的數量。

| 階段           | 張量形狀                       | 說明                  |
| ------------ | -------------------------- | ------------------- |
| Backbone 輸出  | `[1, 512, Hf, Wf]`         | YOLOv7 SPP-CSPC 特徵圖 |
| 偵測框          | `[N, 4]`                   | 每個物件的 bounding box  |
| ROI Align 輸入 | `[1, 512, Hf, Wf] + [N,5]` | 加上 batch index |
| ROI Align 輸出 | `[N, 512, 10, 10]`         | 固定大小 ROI 特徵         |

> [!NOTE]
>
>  ROI Align 時為單圖擷取因此該階端的batch index 皆為0，直至搜集完全資料集中圖像之 ROI 特徵才透過DataLoader將 ROI Dataset 編入正式的batch index  

為驗證本模型於不同畫面下的相同物件之辨識能力，利用不同幀下的標記圖像評估模型的辨識能力，該驗證資料集透過透過成對的圖像與標記檔模擬不同幀下的圖像輸入，其中為確保指標計算得以正常進行標記檔中至少存在一個跨越不同幀的物件。


| 指標名稱 | 定義 | 值域 / 方向 | note |
|---|---|---|---|
| **Top-1 Accuracy** | 正確匹配的候選物件是否排在第 1 名：<br>$$\text{Top1}=\frac{1}{Q_v}\sum_{i=1}^{Q_v}\mathbf{1}\big[\arg\max_j P_{ij}=gt_i\big]$$ | $$[0,1],\ \text{越大越好}$$ | 最直觀的辨識能力指標，代表模型「第一個選擇」是否正確 |
| **Mean Rank** | 正確目標在排序中的平均名次：<br>$$\text{MeanRank}=\frac{1}{Q_v}\sum_{i=1}^{Q_v}\text{rank}_i$$ | $$[1,N],\ \text{越小越好}$$ | 模型通常把正確目標排在第幾名，越靠前代表區分能力越好 |
| **MRR (Mean Reciprocal Rank)** | 名次倒數的平均值：<br>$$\text{MRR}=\frac{1}{Q_v}\sum_{i=1}^{Q_v}\frac{1}{\text{rank}_i}$$ | $$(0,1],\ \text{越大越好}$$ | 強調前幾名的重要性，rank=1 影響最大 |
| **Recall@K** | 正確目標是否落在前 \(K\) 名：<br>$$\text{Recall@}K=\frac{1}{Q_v}\sum_{i=1}^{Q_v}\mathbf{1}[\text{rank}_i\le K]$$ | $$[0,1],\ \text{越大越好}$$ | 模型在 Top-\(K\) 內是否能「找得到正解」 |


