#!/bin/bash

#========================
# ISIC 数据集下载脚本
#========================

# 设置下载目录
OUTDIR="/mnt/c/Users/guyiq/Desktop/ISIC_Archive"
mkdir -p "$OUTDIR"

# 定义函数：下载指定类别
download_class() {
    local search_filter="$1"
    local limit="$2"
    local out_csv="$3"

    echo "开始下载: $out_csv ..."

    # 执行下载
    isic image download --search "$search_filter" --limit $limit "$OUTDIR"

    # 检查 metadata.csv 是否生成
    if [ -f "$OUTDIR/metadata.csv" ]; then
        mv "$OUTDIR/metadata.csv" "$OUTDIR/$out_csv"
        echo "保存 metadata: $out_csv"
    else
        echo "警告: metadata.csv 未生成，跳过 mv"
    fi
}

#========================
# 按 diagnosis 下载各类别
#========================

download_class 'diagnosis_2:"Benign melanocytic proliferations"' 12875 NV.csv
download_class 'diagnosis_2:"Malignant melanocytic proliferations (Melanoma)"' 4522 MEL.csv
download_class 'diagnosis_3:"Basal cell carcinoma"' 3393 BCC.csv
download_class 'diagnosis_3:"Seborrheic keratosis"' 1464 SK.csv
download_class 'diagnosis_3:"Solar or actinic keratosis"' 869 AK.csv
download_class 'diagnosis_3:"Squamous cell carcinoma in situ"' 656 SCC.csv
download_class 'diagnosis_3:"Pigmented benign keratosis"' 384 BKL.csv
download_class 'diagnosis_3:"Solar lentigo"' 270 SL.csv
download_class 'diagnosis_2:"Benign soft tissue proliferations - Vascular"' 253 VASC.csv //
download_class 'diagnosis_3:"Dermatofibroma"' 246 DF.csv
download_class 'diagnosis_4:"Actinic keratosis, Lichenoid"' 16 LK.csv
download_class 'diagnosis_3:"Lentigo simplex"' 27 LS.csv
download_class 'diagnosis_3:"Hemangioma"' 15 AN.csv

#========================
# 可选：调用 merge.py 合并 CSV
#========================
if [ -f "./merge.py" ]; then
    python3 ./merge.py
else
    echo "merge.py 未找到，跳过合并"
fi

echo "下载完成"