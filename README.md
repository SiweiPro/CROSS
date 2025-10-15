# NeurIPS 2025 CROSS

This is the official implementation for NeurIPS Research Track Paper: **Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with Large Language Models**

We have provided the related resources here: [[📃 Paper](https://arxiv.org/abs/2503.14411)]  [[🎬 Video]()] 


> Feel free to cite this work if you find it useful to you! 😄
```
@inproceedings{zhang2025unifying,
  title={Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with Large Language Models}, 
  author={Siwei Zhang and Yun Xiong and Yateng Tang and Jiarong Xu and Xi Chen and Zehao Gu and Xuezheng Hao and Zian Jia and Jiawei Zhang},
  booktitle={Proceedings of Neural Information Processing Systems},
  year={2025}
}

```

## Paper Intro


## How to run CROSS

### Step 1: Prepare the dataset.

We provide the example of the Enron dataset, and the other datasets follow a similar setup.

First, download the original dataset from [here](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk) and save them into the ```./DyLink_Datasets/Enron/``` directory. Then, download the LLM-generated texts from [here](https://drive.google.com/drive/folders/1ppHXycl702xq3gzfOs54O9bm2p8vi9U6) and replace the corresponding files of the Enron dataset into the directory. 

The final directory structure should be:
```{bash}
DyLink_Datasets/
└── Enron/
    ├── entity_text.csv
    ├── edge_list.csv
    ├── relation_text.csv
    ├── LLM_temporal_chain_data.csv
    └── chain_results.json
└── GDELT/
    ├── entity_text.csv
    ├── edge_list.csv
    ├── relation_text.csv
    ├── LLM_temporal_chain_data.csv
    └── chain_results.json
└── ICESW1819/
    ├── entity_text.csv
    ├── edge_list.csv
    ├── relation_text.csv
    ├── LLM_temporal_chain_data.csv
    └── chain_results.json
└── Googlemap_CT/
    ├── entity_text.csv
    ├── edge_list.csv
    ├── relation_text.csv
    ├── LLM_temporal_chain_data.csv
    └── chain_results.json

```
### Step 2: Prepare the text embeddings.

Run the following commands to generate the text embeddings using MiniLM:

**For raw texts:**
```{bash}
cd ./CROSS
CUDA_VISIBLE_DEVICES=0 python get_pretrained_embeddings.py
```
**For LLM-generated texts:**
```{bash}
CUDA_VISIBLE_DEVICES=0 python get_temporal_chain_embeddings.py
```

### Step 3: Run temporal link prediction.
To train CROSS using DyGFormer for temporal link prediction on the Enron dataset, run:
```{bash}
CUDA_VISIBLE_DEVICES=0 python train_link_prediction.py --dataset_name Enron --model_name DyGFormer

CUDA_VISIBLE_DEVICES=0 python train_link_prediction.py --dataset_name GDELT --model_name DyGFormer

CUDA_VISIBLE_DEVICES=0 python train_link_prediction.py --dataset_name ICESW1819 --model_name DyGFormer

CUDA_VISIBLE_DEVICES=0 python train_link_prediction.py --dataset_name Googlemap_CT --model_name DyGFormer
```

## How to get the LLM-generated texts

You should add your own authorization from your API account and change the url if using other llms. Please make sure your network connection is stable. In addition, our code supports multi-threaded workers, which you can adjust according to your needs.

Run:
```{bash}
CUDA_VISIBLE_DEVICES=0 python temporal_chain_LLMs.py --model_name deepseek --dataset_name Enron --num_workers 80

CUDA_VISIBLE_DEVICES=0 python temporal_chain_LLMs.py --model_name deepseek --dataset_name GDELT --num_workers 80

CUDA_VISIBLE_DEVICES=0 python temporal_chain_LLMs.py --model_name deepseek --dataset_name ICESW1819 --num_workers 80

CUDA_VISIBLE_DEVICES=0 python temporal_chain_LLMs.py --model_name deepseek --dataset_name Googlemap_CT --num_workers 80
```

## Training logs

To facilitate a quick verification of the reproducibility of our results, we provide the logs of training CROSS in ```./logs/```, which contains the main experimental results and other details.

## Acknowledge

Codes and model implementations are referred to [DTGB](https://github.com/zjs123/DTGB) and [DyGLib](https://github.com/yule-BUAA/DyGLib). Thanks for their great contributions!
