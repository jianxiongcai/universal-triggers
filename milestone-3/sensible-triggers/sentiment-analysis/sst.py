import sys
import os.path
from sklearn.neighbors import KDTree
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
sys.path.append('..')
import utils
import attacks
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--attack', type=str, help='The type of attack by which to generate trigger candidates',default = 'random')
parser.add_argument('-l', '--lamda', type=float, help='Proportion of loss function taken by perplexity', default=0)
parser.add_argument('-b', '--beam', type=int, help='Beam size to use in getting best candidates. 1 if not using beam search', default=1)
parser.add_argument('-s', '--sentiment', type=int, help='Sentiment to filter on. 1 to flip positive to negative; 0 to flip negative to positive', default=1)
args = parser.parse_args()

# Simple LSTM classifier that uses the final hidden state to classify Sentiment. Based on AllenNLP
class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

EMBEDDING_TYPE = "w2v" # what type of word embeddings to use

def main():
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
    model = LstmClassifier(word_embeddings, encoder, vocab)
    model.cuda()

    # where to save the model
    if not os.path.exists('tmp/'):
         os.mkdir('tmp/')
    model_path = "tmp/" + EMBEDDING_TYPE + "_" + "model.th"
    vocab_path = "tmp/" + EMBEDDING_TYPE + "_" + "vocab"
    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = LstmClassifier(word_embeddings, encoder, vocab)
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

    # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
    # We use the gradient later in the attack.
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # also save the word embedding matrix

    # Use batches of size universal_perturb_batch_size for the attacks.
    universal_perturb_batch_size = 128
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # Build k-d Tree if you are using gradient + nearest neighbor attack
    # tree = KDTree(embedding_weight.numpy())

    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    dataset_label_filter = args.sentiment
    if dataset_label_filter:
      print("This experiment is for flipping positive to negative sentiment")
    else:
      print("This experiment is for flipping negative to positive sentiment")
    targeted_dev_data = []
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)

    # get accuracy before adding triggers
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids=None)
    model.train() # rnn cannot do backwards in train mode

    # initialize triggers which are concatenated to the input
    num_trigger_tokens = 3
    print("The trigger token length for this experiment is {}".format(num_trigger_tokens))
    trigger_token_ids = [vocab.get_token_index("the")] * num_trigger_tokens
    
    pos_pattern_pool = [['ADV','ADJ','NOUN'], ["PRON", "VERB", "PRON"], ["ADV", "VERB", "PRON"], ["NOUN", "VERB", "ADJ"], ["VERB", "PRON", "VERB"], ["VERB", "PRON", "ADJ"], ["VERB", "PRON", "NOUN"]] 
    pos_pattern = random.choice(pos_pattern_pool)
    if args.attack == 'hotflip_with_pos' or args.attack == 'random_pos': print("POS pattern for this experiment is {}".format(pos_pattern))

    # sample batches, update the triggers, and repeat
    for batch in lazy_groups_of(iterator(targeted_dev_data, num_epochs=5, shuffle=True), group_size=1):
        # get accuracy with current triggers
        utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids)
        model.train() # rnn cannot do backwards in train mode

        # get gradient w.r.t. trigger embeddings for current batch
        averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids)

        # pass the gradients to a particular attack to generate token candidates for each token.
        if args.attack == 'hotflip_with_pos':
            while True:
              cand_trigger_token_ids = attacks.hotflip_with_pos_attack(averaged_grad,
                                                              embedding_weight,
                                                              trigger_token_ids,
                                                              vocab,
                                                              pos_pattern,
                                                              num_candidates=100,
                                                              increase_loss=True)
              valid_sequence = True
              for i in range(num_trigger_tokens):
                if not len(cand_trigger_token_ids[i]):
                  valid_sequence = False
                  break
              if valid_sequence:
                break
              pos_pattern = random.choice(pos_pattern_pool)
              print("Updated POS pattern for this experiment is {}".format(pos_pattern))

            print("Length of eligbible candidates - {}, {}, {}".format(len(cand_trigger_token_ids[0]), len(cand_trigger_token_ids[1]), len(cand_trigger_token_ids[2])))
        elif args.attack == 'hotflip':
            cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                            embedding_weight,
                                                            trigger_token_ids,
                                                            num_candidates=40,
                                                            increase_loss=True)
        elif args.attack == 'random':
            cand_trigger_token_ids = attacks.random_attack(embedding_weight,
                                                            trigger_token_ids,
                                                            num_candidates=40)
        elif args.attack == 'random_pos':                                                    
            cand_trigger_token_ids = attacks.random_pos_attack(embedding_weight,
                                                            trigger_token_ids, vocab=vocab,
                                                            num_candidates=40)
        elif args.attack == 'nearest_neighbor':                                                
            cand_trigger_token_ids = attacks.nearest_neighbor_grad(averaged_grad,
                                                                    embedding_weight,
                                                                    trigger_token_ids,
                                                                    tree,
                                                                    100,
                                                                    num_candidates=40,
                                                                    increase_loss=True)
        else:
            raise RuntimeError("Error: invalid attack")
        # Tries all of the candidates and returns the trigger sequence with highest loss.
        trigger_token_ids = utils.get_best_candidates(model,
                                                      batch,
                                                      trigger_token_ids,
                                                      cand_trigger_token_ids, vocab, beam_size = args.beam, lamda = args.lamda)

    # print accuracy after adding triggers
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids)

if __name__ == '__main__':
    main()
