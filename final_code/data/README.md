## CIS 530 Project - Data Description
### Alexander Feng, Benedict Florance Arockiaraj, Jianxiong Cai, Xiaoyu Cheng

#### Tasks:


1. **Sentiment Analysis**
- **SST (Stanford Sentiment Treebank)** [Dataset Link](https://nlp.stanford.edu/sentiment/code.html)
     - Published in 2013, the SST dataset contains fine grained sentiment labels for 215,154 phrases from 11,855 movie review sentences cite{socher2013recursive}. Modeled as a classification problem, each phrase is labeled as either 'very negative', 'negative', 'neutral', 'positive' or 'very positive'. This dataset also provides a standard train-dev-test split for benchmark evaluation.

     - The dataset stores all 215,154 phases in dictionary.txt file, and all sentiment scores (float numbers range from 0 to 1) in the sentiment_labels.txts. To model it as a classification task, the author recommend to divided into 5 classes by using the following cut-off: [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]. 

     - Example Input:  '' exceeds expectations . Good fun , good action , good acting , good dialogue , good pace
     - Example Output (Sentiment Score): 0.95833 (Very Positive)
        
- **IMDB Movie Reviews** [Dataset Link](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) 
    - The IMDB Movie Reviews is a large binary sentiment classification dataset containing 50,000 movie reviews (25,000 for training, and 25,000 for testing). The dataset contains movie reviews (with <=30 reviews per movie), and each sentence is labeled as either ’positive’ or ’negative’. Additionally, it provides 50,000 unlabeled sentences, which is intended for unsupervised training. As a standard benchmark dataset for sentiment analysis, the overall distribution is balanced in terms of number of positive vs. negative samples. Only highly polarizing reviews are considered, with negative sentences having <= 4/10 score, and positive sentences having >=7/10 score.

    - There are two top-level directories [train/, test/] corresponding to the training and test sets. Each contains [pos/, neg/] directories for the reviews with binary labels positive and negative. Within these directories, reviews are stored in text files named following the convention [[id]-[rating].txt] where [id] is a unique id and [rating] is the star rating for that review on a 1-10 scale. The [train/unsup/] directory has 0 for all ratings because the ratings are omitted for this portion of the dataset.
                
    - Example Review of rating 1: _"Robert DeNiro plays the most unbelievably intelligent illiterate of all time. This movie is so wasteful of talent, it is truly disgusting. The script is unbelievable. The dialog is unbelievable. Jane Fonda's character is a caricature of herself, and not a funny one. The movie moves at a snail's pace, is photographed in an ill-advised manner, and is insufferably preachy. It also plugs in every cliche in the book. Swoozie Kurtz is excellent in a supporting role, but so what? Equally annoying is this new IMDB rule of requiring ten lines for every review. When a movie is this worthless, it doesn't require ten lines of text to let other readers know that it is a waste of time and tape. Avoid this movie."_
        
2. **Natural Language Inference**
- **SNLI** [Dataset Link](https://nlp.stanford.edu/projects/snli/)
     - The Stanford Natural Language Inference (SNLI) Corpus: SNLI is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels ‘entailment’, ‘contradiction’, and ‘neutral’. The corpus is distributed in both JSON lines and tab separated files. The dataset is split into train, validation, and test, with 550152, 10000, and 10000 instances respectively. 
Each instance is a JSON line contains three fields: premise (a string for determining the truthfulness of hypothesis), hypothesis (a string with truth condition compared to the premise, takes value of true, false, or unknown), and label (an integer takes value of 0, 1, or 2, indicating the entailment / neutral / contradiction relationship between premise and hypothesis; instances without golden label has value -1). The mean token count for premises and hypotheses are 14.1 and 8.3 respectively.  

    - Example Instance: 
```    
{'premise': 'Two women are embracing while holding to go packages.'
 'hypothesis': 'The sisters are hugging goodbye while holding to go packages after just eating lunch.'
 'label': 1}
```

- **MNLI** [Dataset Link](https://cims.nyu.edu/~sbowman/multinli/)
    - The MultiNLI dataset is based off of the SNLI dataset and follows the same format. It differs from SNLI in that it contains different genres, and that it supports evaluation across genres. The dataset, consisting of 433,000 premise-hypothesis pairs, is split into a train set and two dev sets, one of which containing examples from the same sources as the train set and one containing examples from mismatched sources. 
    - Each JSON line contains the same fields as SNLI, as well as fields for parsed versions of the sentences label information from the annotator and validators of the dataset. 
                
    - Example:
```
{"annotator_labels": ["neutral"], 
"genre": "government", 
"gold_label": "neutral", 
"pairID": "31193n", 
"promptID": "31193", 
"sentence1": "Conceptually cream skimming has two basic dimensions - product and geography.", 
"sentence1_binary_parse": "( ( Conceptually ( cream skimming ) ) ( ( has ( ( ( two ( basic dimensions ) ) - ) ( ( product and ) geography ) ) ) . ) )", 
"sentence1_parse": "(ROOT (S (NP (JJ Conceptually) (NN cream) (NN skimming)) (VP (VBZ has) (NP (NP (CD two) (JJ basic) (NNS dimensions)) (: -) (NP (NN product) (CC and) (NN geography)))) (. .)))", 
"sentence2": "Product and geography are what make cream skimming work. ", 
"sentence2_binary_parse": "( ( ( Product and ) geography ) ( ( are ( what ( make ( cream ( skimming work ) ) ) ) ) . ) )", 
"sentence2_parse": "(ROOT (S (NP (NN Product) (CC and) (NN geography)) (VP (VBP are) (SBAR (WHNP (WP what)) (S (VP (VBP make) (NP (NP (NN cream)) (VP (VBG skimming) (NP (NN work)))))))) (. .)))"}
```                
