import random
import sys
import os.path

import allennlp.data.fields
import numpy.random
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
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
sys.path.append('..')
import utils
import attacks
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train the model on the new data',default=5)
parser.add_argument('-r', '--ratio', type=float, help='the ratio of original training data vs. augmented adversarial sample (can be bigger than 1)', default=0.6)
args = parser.parse_args()

torch.manual_seed(52)
random.seed(15)
numpy.random.seed(43)

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


def train_model(model, train_data, dev_data, vocab, model_path, vocab_path, num_epochs):
    """
    Train the model with train_data and dev_data, then save to disk
    @return: the updated model
    """
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    optimizer = optim.Adam(model.parameters())
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_data,
                      validation_dataset=dev_data,
                      num_epochs=num_epochs,
                      patience=1,
                      cuda_device=0)
    trainer.train()
    with open(model_path, 'wb') as f:
        print("[INFO] Model saved to " + model_path)
        torch.save(model.state_dict(), f)
    vocab.save_to_files(vocab_path)
    return model


def generate_triggers(model, vocab, dev_data):
    # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
    # We use the gradient later in the attack.
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model)  # also save the word embedding matrix

    # Use batches of size universal_perturb_batch_size for the attacks.
    universal_perturb_batch_size = 128
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # Build k-d Tree if you are using gradient + nearest neighbor attack
    # tree = KDTree(embedding_weight.numpy())

    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    dataset_label_filter = "0"
    targeted_dev_data = []
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)

    # get accuracy before adding triggers
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids=None)
    model.train()  # rnn cannot do backwards in train mode

    # initialize triggers which are concatenated to the input
    num_trigger_tokens = 3
    trigger_token_ids = [vocab.get_token_index("the")] * num_trigger_tokens

    # sample batches, update the triggers, and repeat
    for batch in lazy_groups_of(iterator(targeted_dev_data, num_epochs=5, shuffle=True), group_size=1):
        # get accuracy with current triggers
        utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids)
        model.train()  # rnn cannot do backwards in train mode

        # get gradient w.r.t. trigger embeddings for current batch
        averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids)

        # pass the gradients to a particular attack to generate token candidates for each token.
        cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                        embedding_weight,
                                                        trigger_token_ids,
                                                        num_candidates=40,
                                                        increase_loss=True)
        # cand_trigger_token_ids = attacks.random_attack(embedding_weight,
        #                                                trigger_token_ids,
        #                                                num_candidates=40)
        # cand_trigger_token_ids = attacks.nearest_neighbor_grad(averaged_grad,
        #                                                        embedding_weight,
        #                                                        trigger_token_ids,
        #                                                        tree,
        #                                                        100,
        #                                                        num_candidates=40,
        #                                                        increase_loss=True)

        # Tries all of the candidates and returns the trigger sequence with highest loss.
        trigger_token_ids = utils.get_best_candidates(model,
                                                      batch,
                                                      trigger_token_ids,
                                                      cand_trigger_token_ids)

    trigger_tokens = []
    for token_id in trigger_token_ids:
        trigger_str = vocab.get_token_from_index(token_id)
        trigger_tokens.append(Token(trigger_str))
    return trigger_tokens

# ============================ Helper Functions for Defense ===============================
# def prepend_triggers(data, triggers, ratio, single_id_indexer):
#     """
#     Append the generated trigger to part of the training data
#     @param train_data: List of allennlp.Instance
#     @param triggers: the generated universal adversarial triggers
#                     (e.g. trigger = [Token("good"), Token("good"), Token("good")])
#     @param ratio: (float) 0-1: the portion of training data to prepend the trigger
#     """
#     N_samples = int(len(data) * ratio)
#     data_raw = list(data)
#     data_shuffled = list(data)
#     random.shuffle(data_shuffled)
#     data_sampled = data_shuffled[0:N_samples]
#
#     # adversarial training samples
#     data_adversarial = []
#
#     for instance_old in data_sampled:
#         # get token and label info
#         tokens_old = instance_old.fields['tokens']
#         label_old = instance_old.fields["label"]
#
#         # generate new training instance and label
#         tokens_new = allennlp.data.fields.TextField(trigger + tokens_old.tokens, {"tokens": single_id_indexer})
#         label_new = label_old
#         instance_new = Instance({
#             'tokens': tokens_new,
#             'label': label_old
#         })
#         data_adversarial.append(instance_new)
#     return data_adversarial


def augment_training_data(data, triggers, ratio, single_id_indexer):
    """
    Argument the original dataset with the generated triggers
    @param train_data: the original training dataset
    @param triggers: List of generated triggers
    @param ratio: the ratio of samples to argument
    """
    assert isinstance(triggers, list)
    print("[INFO] Preparing augmented training data with adversarial samples. Ratio: " + str(ratio) + " N_triggers: " + str(len(triggers)))
    # Compute number of samples to augment
    N_samples = int(len(data) * ratio)
    data_raw = list(data)

    # select N samples from the data to augment (copy then augment)
    data_shuffled = list(data)
    random.shuffle(data_shuffled)
    data_sampled = data_shuffled[0:N_samples]

    data_adversarial = []
    for instance_old in data_sampled:
        # get token and label info
        tokens_old = instance_old.fields['tokens']
        label_old = instance_old.fields["label"]

        # select the trigger to prepend
        idx = random.randint(0, len(triggers) - 1)
        tri = triggers[idx]

        # generate new training instance and label
        tokens_new = allennlp.data.fields.TextField(tri + tokens_old.tokens, {"tokens": single_id_indexer})
        label_new = label_old
        instance_new = Instance({
            'tokens': tokens_new,
            'label': label_new
        })
        data_adversarial.append(instance_new)

    return data_adversarial

def get_model_path(iteration, EMBEDDING_TYPE):
    # model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "models.th"
    model_dir = os.path.join("/tmp", EMBEDDING_TYPE + "_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, str(iteration) + ".pth")
    return model_path


# ===================================== Parameters ===================================
ratio = args.ratio                   # the ratio of original training data vs. augmented adversarial sample (can be bigger than 1)
num_epochs = args.epochs             # Number of epoches to train for each iteration

# ======================================== MAIN ======================================
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
    vocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab"

    print("[INFO] Training Stage 0")
    # train the initial model
    model_path_0 = get_model_path(0, EMBEDDING_TYPE)
    model = train_model(model, train_data, dev_data, vocab, model_path_0, vocab_path, num_epochs=num_epochs)
    model.train().cuda()  # rnn cannot do backwards in train mode

    # generate initial triggers
    trigger_curr = generate_triggers(model, vocab, dev_data)
    utils.reset_hooks(model)

    # prepending to training data
    # train_data_adv = []
    triggers = [trigger_curr]

    for stage_id in range(1, 100):
        train_data_adv = augment_training_data(train_data, triggers, ratio, single_id_indexer)
        # train_data_adv += data_extended
        train_data_combined = train_data + train_data_adv

        # retrain the model with dataset including adv samples
        # reset the model to remove gradient hooks
        # model = None
        # encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
        #                                              hidden_size=512,
        #                                              num_layers=2,
        #                                              batch_first=True))
        # model = LstmClassifier(word_embeddings, encoder, vocab)
        # model.cuda()
        # model.load_state_dict(torch.load(model_path))
        # print("[INFO] Model Loaded from " + model_path)

        # retain the model with the combined dataset
        print("[INFO] Training Stage {}".format(stage_id))
        model_path = get_model_path(stage_id, EMBEDDING_TYPE)
        model = train_model(model, train_data_combined, dev_data, vocab, model_path, vocab_path, num_epochs=num_epochs)
        model.train().cuda()  # rnn cannot do backwards in train mode

        # generate triggers
        trigger_curr = generate_triggers(model, vocab, dev_data)
        triggers.append(trigger_curr)
        utils.reset_hooks(model)

    # meta = {
    #     'triggers': triggers,
    # }
    # save the trigger generated.
    with open("/tmp/trigger_generated.plk", "wb") as f:
        pickle.dump(triggers, f)

if __name__ == '__main__':
    main()
