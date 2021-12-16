## Usage

### Fine-tuning (Training) Defense Model
```bash
python sst_defemse.py ${RATIO} ${NUM_EPOCHS}
```

- ${RATIO}: The ratio of augmented adversarial samples for training
- ${NUM_EPOCHS}: The number of epochs to train for each attack-defense iteration

### Error Analysis
The ```error_analysis.py``` load a model each time and print out all errors made by the model (Attack Direction: Negative -> Positive)
```bash
python error_analysis
```

### Helper Tool: log_parsing
This tool is used to find common errors made by both models (the original model and the defense model) and
the errors got corrected in the defense model.