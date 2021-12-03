# Build Instructions

1. `pip install -r requirements.txt`
2. `pip install overrides==3.1.0`

# Baselines

## Random Baseline
Change `dataset_label_filter` to the source label in random_attack_sst.py and run the the file.

## Nearest Neighbor Baseline
Change `dataset_label_filter` to the source label in nearest_neigbor_attack_sst.py and run the the file.

## Hardcoded Baseline

## Top Frequent Word Baseline
### Count Top Frequent Word and Visualize with wordcloud.
```python
python top_frequent_wordcloud_sst.py
```
### Run baseline (Top Frequent Word)
Change `dataset_label_filter` to the source label in sst_freq_baseline.py and run the the file.
```python
python sst_freq_baseline.py
```

## Universal Adversarial Attack Baseline
Change `dataset_label_filter` to the source label  in universal_adversarial_attack_sst.py and run the the file.
