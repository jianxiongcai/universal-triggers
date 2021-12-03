import argparse
from sentiment-analysis import sst, utils
from natural-language-inference import snli

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Evaluate on Sentiment Analysis ('sst') or Natural Language Inference ('snli') model")
parser.add_argument("triggers", type=list, help="List of triggers to evaluate")
parser.add_argument("-s", "--subset", type=str, help="Subset of data to evaluate on. \
('positive','negative','all' for Sentiment Analysis, 'entailment', 'contradiction', 'neutral', 'all' for Natural Language Inference)", default='all')

args = parser.parse_args()

def main():
  if(args.model == 'sst'):
    # load the binary SST dataset.
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer},
                                                    use_subtrees=True)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    # test_dataset = reader.read('data/sst/test.txt')

    vocab = Vocabulary.from_instances(train_data)

    # Randomly initialize vectors
    if EMBEDDING_TYPE == "None":
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)
        word_embedding_dim = 300

    # Load word2vec vectors
    elif EMBEDDING_TYPE == "w2v":
        embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        weight = _read_pretrained_embeddings_file(embedding_path,
                                                  embedding_dim=300,
                                                  vocab=vocab,
                                                  namespace="tokens")
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=300,
                                    weight=weight,
                                    trainable=False)
        word_embedding_dim = 300

    # Initialize model, cuda(), and optimizer
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    model = sst.LstmClassifier(word_embeddings, encoder, vocab)
    model.cuda()

    # where to save the model
    model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model.th"
    vocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab"
    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = sst.LstmClassifier(word_embeddings, encoder, vocab)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    # otherwise train model from scratch and save its weights
    else:
        iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
        iterator.index_with(vocab)
        optimizer = optim.Adam(model.parameters())
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_data,
                          validation_dataset=dev_data,
                          num_epochs=5,
                          patience=1,
                          cuda_device=0)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
    model.train().cuda() # rnn cannot do backwards in train mode
    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    if args.subset=='positive':
      dataset_label_filter = "1"
      targeted_dev_data = []
      for instance in dev_data:
          if instance['label'].label == dataset_label_filter:
              targeted_dev_data.append(instance)
    elif args.subset=='negative':
      dataset_label_filter = "0"
      targeted_dev_data = []
      for instance in dev_data:
          if instance['label'].label == dataset_label_filter:
              targeted_dev_data.append(instance)
    elif args.subset == 'all:
      targeted_dev_data = dev_data
    else:
      raise RuntimeError(f"Error: {args.subset} is not a valid subset for sst")
    try:
      trigger_token_ids = [vocab.get_token_index(trigger) for trigger in args.triggers]
    except KeyError:
      raise
    for trigger, token in zip(args.triggers, trigger_token_ids):
      if token == vocab._oov_token: print(f'Warning: {token} out of vocabulary')
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids=None)
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids)
    
  elif(args.model == 'snli'):
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    tokenizer = WordTokenizer(end_tokens=["@@NULL@@"]) # add @@NULL@@ to the end of sentences
    reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)
    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')
    # Load model and vocab
    model = load_archive('https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz').model
    model.eval().cuda()
    vocab = model.vocab

    # add hooks for embeddings so we can compute gradients w.r.t. to the input tokens
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # save the word embedding matrix

    # Batches of examples to construct triggers
    universal_perturb_batch_size = 32
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    #check argument is valid
    if args.subset not in ['entailment','contradiction','neutral','all']:
      raise RuntimeError(f'Error: {args.subset} is not a valid subset for snli')
    elif args.subset == 'all':
      subset_dev_dataset = dev_dataset
    else:
    # Subsample the dataset to one class to do a universal attack on that class
      subset_dev_dataset = []
      for instance in dev_dataset:
          if instance['label'].label == args.subset:
              subset_dev_dataset.append(instance)



    # Get original accuracy before adding universal triggers
    utils.get_accuracy(model, subset_dev_dataset, vocab, trigger_token_ids=None, snli=True)
    model.train() # rnn cannot do backwards in train mode
    try:
      trigger_token_ids = [vocab.get_token_index(trigger) for trigger in args.triggers]
    except KeyError:
      raise
    for trigger, token in zip(args.triggers, trigger_token_ids):
      if token == vocab._oov_token: print(f'Warning: {token} out of vocabulary')
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids, snli=True)
  else:
    print("ERROR: model must be 'sst' or 'snli'")
if __name__=='__main__':
  main()
