# DFAN-HAD
論文Deep Feature Aggregation Network for Hyperspectral Anomaly Detection的程式碼練習

## Usage
### Requirements
* python==3.7
* torch==1.13.1
* torchvision==0.14.1
* scikit-learn==1.02
* numpy==1.21.5
* matplotlib==3.5.0
* scipy==1.7.3

## Abstract
Hyperspectral anomaly detection (HAD) is a challenging task since it identifies the anomaly targets without prior knowledge. In recent years, deep learning methods have emerged as one of the most popular algorithms in the HAD. These methods operate on the assumption that the background is well reconstructed while anomalies cannot, and the degree of anomaly for each pixel is represented by reconstruction errors. However, most approaches treat all background pixels of a hyperspectral image (HSI) as one type of ground object. This assumption does not always hold in practical scenes, making it difficult to distinguish between backgrounds and anomalies effectively. To address this issue, a novel deep feature aggregation network (DFAN) is proposed in this paper, and it develops a new paradigm for HAD to represent multiple patterns of backgrounds. The DFAN adopts an adaptive aggregation model, which combines the orthogonal spectral attention module with the backgroundanomaly category statistics module. This allows effective utilization of spectral and spatial information to capture the distribution of the background and anomaly. To optimize the proposed DFAN better, a novel multiple aggregation separation loss is designed, and it is based on the intra-similarity and interdifference from the background and anomaly. The constraint
function reduces the potential anomaly representation and strengthens the potential background representation. Additionally, the extensive experiments on the six real hyperspectral datasets demonstrate that the proposed DFAN achieves superior performance for HAD. The code is available at https://github.com/ChengXi-1217/DFAN-HAD.

### Directory
The directory should be like this:

````
-- main.py
-- train.py        (two-stage training)
-- test.py         (anomaly detection)
-- model.py        (construction of DFAN )
-- evaluate.py     (AUCs of (Pd,Pf) and (Pf, τ))
-- transforms.py   (GramSchmidt Transform)
-- utils.py        (adaptive aggregation model ,AAM)
-- requirements.txt
-- Result (.mat)
-- Data (train dataset and test dataset)
   |-- HYDICE.mat
   |-- Salinas.mat
   |-- abu-airport-4.mat 
   |-- abu-airport-4.mat
   |-- abu-beach-4.mat
   |-- abu-urban-1.mat
   |-- abu-urban-1.mat
   
````
### Training and tesing of the DFAN

running main.py
