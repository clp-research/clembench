# WinoDict new word generation
## 1. Download WinoDict code
Due to the unwieldy structure of the GitHub repository that contains the WinoDict code, the best solution to get the 
code utilized for generating new-words instances is to download the 
https://github.com/google-research/language/tree/master/language/wino_dict subdirectory from GitHub. This can be done 
using https://download-directory.github.io/, downloading the files one-by-one or similar. Make sure the files from the 
repository are in the `/resources/new_word_generation/wino_dict` subdirectory.
## 2. Install WinoDict requirements
Run `pip install -r requirements.txt` in `/resources/new_word_generation/wino_dict` using the python environment you're 
using for AdventureGame instance generation.  
Then download the required datasets with
```
python -m nltk.downloader omw-1.4
python -m nltk.downloader wordnet
python -m spacy download en_core_web_md-3.0.0a1
```
(Downloading datasets might not be needed, just like much of the code files.)
## 3. Generate new word files
To generate a words file, with `/resources/new_word_generation/wino_dict` as working directory, run
```
python create_new_words.py --output_path=../new_words.tsv 
```
To generate unique new words for new instances, a RNG seed is passed:
```
python create_new_words.py --output_path=../new_words.tsv --seed=42
```
The default seed is `1`, the seed used for the first set of new-word experiment instances is `42`.