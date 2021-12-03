# Build Instructions

1. !pip install -r requirements.txt
2. !pip install overrides==3.1.0

# Baselines

## Random Baseline
Change `dataset_label_filter` to the source label and `target_label` to the target label in random_attack_snli.py and run the the file.

## Nearest Neighbor Baseline
Change `dataset_label_filter` to the source label and `target_label` to the target label in nearest_neigbor_attack_snli.py and run the the file.

## Hardcoded Baseline
## Top Frequent Words Baseline
## Universal Adversarial Attack Baseline
Change `dataset_label_filter` to the source label and `target_label` to the target label in universal_adversarial_attack_snli.py and run the the file.
