# Codes for Submission 5864 "Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with Large Language Models"

We sincerely thank the reviewers for their time and contributions during the review process.

## Quick reproducibility check

To facilitate a quick verification of the reproducibility of our results during the review process, we provide the logs of training CROSS in ```./CROSS/logs/```, which contains the main experimental results and other details. Please feel free to review them.

## How to run

We provide example steps to run our code on the Enron dataset.

Due to the file size restriction, the dataset must be prepared before running the experiments.

---

### Step 1: Prepare the dataset.

First, unzip the LLM-generated textual data for Enron:
```{bash}
cd ./DyLink_Datasets
unzip Enron.zip
```

Then, download the original Enron dataset from [here](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk), and place the following files into the ```./DyLink_Datasets/Enron/``` directory. The final directory structure should be:
```{bash}
DyLink_Datasets/
└── Enron/
    ├── entity_text.csv
    ├── edge_list.csv
    ├── relation_text.csv
    └── chain_results_8.json
```
### Step 2: Prepare the pretrained text embeddings.

Run the following commands to generate the pretrained text embeddings using MiniLM:

**For raw texts in TTAGs:**
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
```

## Acknowledge

Codes and model implementations are referred to [DyGLib](https://github.com/yule-BUAA/DyGLib) and [DTGB](https://github.com/zjs123/DTGB) project. Thanks for their great contributions!
