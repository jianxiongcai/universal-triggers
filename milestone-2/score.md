Usage: python score.py [MODEL] -t [TRIGGER] [TRIGGER] ... -s [SUBSET]

Displays accuracy on chosen subset of dev dataset with chosen model, without then with given trigger sequence.

MODEL:str - Model to evaluate. 'sst' for Sentiment Analysis, 'snli' for Natural Language Inference

TRIGGERS:list(str) - Trigger sequence to evaluate. Input as multiple arguments e.g. -t zoning tapping fiennes

SUBSET:str - Subset of dataset to evaluae. 'positive','negative', or 'all' for Sentiment Analysis, 
'entailment', 'contradiction', 'neutral', or 'all' for Natural Language Inference. If not all, then the 
dataset is filtered to contain only examples with the chosen label.
