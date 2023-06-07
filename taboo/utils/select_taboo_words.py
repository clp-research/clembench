"""Script that generates frequency lists from unigram counts.

    This script works in steps, some of which take some time, so intermediate
    results are written to files and then read in the next step.
    It is recommended to only run one of the 4 steps in main() at a time:

    1) preprocess_unigrams(UNIGRAMS)
        assigns a POS tag and lemma to each token and removes function words
            as listed below.
        produces tagged_unigrams.json
    2) preprocess_unigrams_from_json("tagged_unigrams.json")
        creates a taboo dataframe with the unique lemmas found in the unigram
            list.
        In steps of 100 words, combines the token frequencies for a lemma
        This step takes long and therefore creates single json files for every
            100 words.
        Use the index i to start off at a higher index after aborting the
            script at a previous run.
    3) combine_counts()
        adds the lemma counts to the taboo dataframe
        produces taboo_words_and_counts.json
        the single counts jsons can now be deleted.
    4) create_taboo_lists()
        sorts the table by frequency
        removes words with a frequency of less than 5 per 1 million
        divides the dataframe into 3 equally sized parts
        creates text files with words based on a random selection of 100 words
            from each part


Remove tokens that are tagged as:

IN – conjunction, subordinating or preposition
UH – interjection
WRB – wh-adverb
DT – determiner
PRP – pronoun, personal
CD – cardinal number
FW – foreign word
. – punctuation mark, sentence closer
WP$ – wh-pronoun, possessive
CC – conjunction, coordinating
WDT – wh-determiner
WP – wh-pronoun, personal
TO – infinitival "to"
PRP$ – pronoun, possessive
LS – list item marker
ADD – email
EX – existential there
XX – unknown
: – punctuation mark, colon or ellipsis
NFP – superfluous punctuation
`` – opening quotation mark
, – punctuation mark, comma
PDT – predeterminer
"""

import random
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")  # en-core-web-sm-3.5.0

DATA = ("taboo_high.json", "taboo_medium.json", "taboo_low.json")
UNIGRAMS = "unigram_freq.csv"

REMOVE = (
    "IN", "UH", "WRB", "DT", "PRP", "CD", "FW", ".", "WP$", "CC", "WDT", "WP",
    "TO", "LS", "ADD", "EX", "XX", ":", "NFP", "``", ",", "PDT", "PRP$"
)


def tag_it(word):
    token = nlp(word)
    return pd.Series([token[0].tag_, token[0].lemma_])


def is_function_word(tag):
    return tag in REMOVE


def preprocess_unigrams(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df[["POS", "lemma"]] = df["word"].apply(tag_it)
    df["exclude"] = df["POS"].map(is_function_word)
    df = df.where(df["exclude"] == False).dropna()
    df.to_json("tagged_unigrams.json")


def preprocess_unigrams_from_json(filename):
    df = pd.read_json("tagged_unigrams.json")
    get_taboo_words(df)


def get_taboo_words(df):
    # select the ones that are not function words
    labels = df["POS"].unique()
    for label in labels:
        print(f"{label} – {spacy.explain(label)}")

    taboo = pd.DataFrame({"word": df["lemma"].unique()})
    print(taboo.shape)
    taboo.to_json("taboo_words.json")

    print(df.shape)

    def sum_counts(lemma):
        return df["count"].where(df["lemma"] == lemma).sum()

    for i in range(143700, len(taboo), 100):
        print(i)
        count = taboo["word"][i:min(i + 100, len(taboo))].apply(sum_counts)
        count.to_json(f"counts_{i}_to_{i + 100}.json")

    taboo.to_json("taboo_words.json")
    print(taboo)


def create_taboo_lists(filename):
    taboo = pd.read_json(filename)
    taboo["frequency per million"] = taboo["count"] / 1000000
    taboo["freq_score"] = round(taboo["frequency per million"], 2)
    taboo = taboo.sort_values(by=["freq_score"], ascending=False)

    # taboo = taboo[101:].where(taboo["frequency per million"]>=5).dropna()
    taboo = taboo.where(taboo["frequency per million"] >= 5).dropna()

    high_freq = int(len(taboo) / 3)
    med_freq = len(taboo) - high_freq

    print(high_freq, med_freq)

    # choose 100 from the index
    data_high = taboo[:high_freq]
    high_freq_100 = random.choices(data_high.index, k=100)

    data_med = taboo[high_freq:med_freq]
    med_freq_100 = random.choices(data_med.index, k=100)

    data_low = taboo[med_freq:]
    low_freq_100 = random.choices(data_low.index, k=100)

    print(taboo.loc[low_freq_100, "word"][:10])
    print(taboo.loc[high_freq_100, "word"][:10])
    print(taboo.loc[med_freq_100, "word"][:10])

    with open("taboo_low_freq_100.txt", "w") as o:
        for w in taboo.loc[low_freq_100, "word"]:
            o.write(f"{w}\n")

    with open("taboo_medium_freq_100.txt", "w") as o:
        for w in taboo.loc[med_freq_100, "word"]:
            o.write(f"{w}\n")

    with open("taboo_high_freq_100.txt", "w") as o:
        for w in list(taboo.loc[high_freq_100, "word"]):
            o.write(f"{w}\n")


def combine_counts():
    # put all the files together
    taboo = pd.read_json("taboo_words.json")
    taboo = taboo.reindex(index=(i for i in range(len(taboo))))
    taboo["count"] = -1
    for i in range(0, len(taboo), 100):
        counts = pd.read_json(f"counts_{i}_to_{i + 100}.json", orient="index")
        taboo.loc[i:i + 100, "count"] = counts[0]

    taboo.to_json("taboo_words_and_counts.json")


if __name__ == "__main__":
    # preprocess_unigrams(UNIGRAMS)
    # preprocess_unigrams_from_json("tagged_unigrams.json")
    # combine_counts()
    create_taboo_lists("taboo_words_and_counts.json")
