import sys
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
sys.path.append('../..')

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# == main ===
# load the binary SST dataset.
single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
# use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
# Note: For actual train reader in baseline, use_subtrees should be set to True!
reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                token_indexers={"tokens": single_id_indexer})
train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                token_indexers={"tokens": single_id_indexer})
dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')

vocab = Vocabulary.from_instances(train_data)

freq_poositive = dict()
freq_negative = dict()
text_positive = ""
text_negative = ""

# stop_words = set(stopwords.words('english'))
stop_words = set(STOPWORDS)
stop_words.add(".")
stop_words.add(",")
stop_words.add("'")
stop_words.add("''")
stop_words.add("`")
stop_words.add("``")
stop_words.add("...")
stop_words.add("'s")
stop_words.add("--")
stop_words.add("-rrb-")
stop_words.add("-lrb-")
stop_words.add("film")
stop_words.add("movie")
stop_words.add("one")

# count frequency
for data in train_data:
    tokens = data.fields["tokens"]
    label_str = data.fields["label"].label
    for x in tokens:
        word = str(x).lower()
        if word in stop_words:
            continue;

        if (label_str == "1"):          # positive
            if word not in freq_poositive:
                freq_poositive[word] = 0
            freq_poositive[word] += 1
            text_positive += word + " "
        elif (label_str == "0"):        # negative
            if word not in freq_negative:
                freq_negative[word] = 0
            freq_negative[word] += 1
            text_negative += word + " "
        else:                           # error (unknown label)
            print("[ERROR] Label: " + label_str)

# ===== Generate Word Cloud based on Frequency ==========
wc1 = WordCloud(stopwords=stop_words, background_color="white")
wc1.generate_from_frequencies(freq_poositive)
plt.imshow(wc1, interpolation='bilinear')
plt.axis("off")
plt.savefig("wordcloud_positive.png")
plt.show()

wc2 = WordCloud(stopwords=stop_words, background_color="white")
wc2.generate_from_frequencies(freq_negative)
plt.imshow(wc2, interpolation='bilinear')
plt.axis("off")
plt.savefig("wordcloud_negative.png")
plt.show()

sorted_positve = sorted(freq_poositive.items(), key=(lambda x: -1 * x[1]))
sorted_negative = sorted(freq_negative.items(), key=(lambda x: -1 * x[1]))

# save the first x numbers
with open("word_positive_sorted.txt", "w") as f:
    for x in sorted_positve:
        f.write(str(x) + "\n")

with open("word_negative_sorted.txt", "w") as f:
    for x in sorted_negative:
        f.write(str(x) + "\n")

