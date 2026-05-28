#!/bin/bash

# 下載訓練好的模型
echo "Downloading models from Google Drive..."

# 下載段落選擇模型
gdown --folder https://drive.google.com/drive/folders/1gmJMwgH6O8TWzke4UFcPKoVsfyyhu1WM -O ./paragraph_selection_model_full

# 下載QA模型  
gdown --folder https://drive.google.com/drive/folders/1gmJMwgH6O8TWzke4UFcPKoVsfyyhu1WM -O ./qa_model_full

echo "Download completed!"