# Text Classification: DBpedia Ontology (14 classes)

A small NLP project that compares a classic baseline (**TF-IDF + Multinomial Naive Bayes**) with a **simple neural network** (Keras embedding + average pooling) on the [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) dataset. Each Wikipedia abstract is assigned to one of 14 ontology classes.

It was written as a graded assignment for a university NLP course (Summer 2025). A summary of the task is in [`instructions/todo.md`](instructions/todo.md).

## Results

The test set is 21,000 held-out samples, 1,500 per class.

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Random guess (sanity check) | 0.075 | – |
| TF-IDF (5k features) + Multinomial NB | **0.944** | 0.94 |
| Embedding + GlobalAvgPool + Dense NN | **0.960** | 0.96 |

The neural network is better on most classes. The biggest gains are on `Company` and `Artist`, where Naive Bayes recall drops to 0.83. Naive Bayes trains in seconds and is still a strong baseline.

## Dataset

- **Source:** `fancyzhx/dbpedia_14` on Hugging Face (630,000 samples: 560k train + 70k test).
- **Classes:** Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.
- **Subset used:** to keep training fast, the original splits are merged, then **10,000 samples per class** (140k total) are drawn and split with stratification into:
  - train: 98,000 (70%)
  - validation: 21,000 (15%)
  - test: 21,000 (15%)
- Only the `content` field (the abstract) is used as model input. `title` is not used.

## Pipeline

All the code is in a single notebook, [`Agnihotri, Kavyanjali - NLP.ipynb`](Agnihotri,%20Kavyanjali%20-%20NLP.ipynb):

1. **Load and explore:** class balance, word-count statistics, top unigrams and bigrams per split.
2. **Preprocess** with NLTK: lowercase → remove non-letters → `word_tokenize` → remove English stopwords → WordNet lemmatization. The average length drops from about 46 to about 27 words.
3. **Model 1: Naive Bayes.** `TfidfVectorizer(max_features=5000)` + `MultinomialNB()`.
4. **Model 2: Neural network.** Keras `TextVectorization` (10k vocab, sequence length 200) → `Embedding(64)` → `GlobalAveragePooling1D` → `Dense(64, relu)` → `Dense(14, softmax)`, trained with Adam for 100 epochs.
5. **Evaluate:** accuracy, per-class precision/recall/F1, confusion matrices, and a random baseline.
6. **Discuss:** strengths and weaknesses of each model and when to pick which.

The full write-up is in [`Agnihotri, Kavyanjali -NLP_report.pdf`](Agnihotri,%20Kavyanjali%20-NLP_report.pdf).

## Running it

Requires Python 3.10+ (developed on 3.12).

```bash
pip install datasets nltk tensorflow scikit-learn pandas numpy matplotlib seaborn
jupyter notebook "Agnihotri, Kavyanjali - NLP.ipynb"
```

The notebook was written with Google Colab in mind, and a few cells need attention outside Colab:

- **HF token cell:** the cell that imports `google.colab.userdata` only works in Colab, and it contains a stray line (`Tokens`) that raises an error. The dataset is public, so **you can skip this cell** when running elsewhere.
- **Confusion matrix cell:** the `nb_cm = ...` and `nn_cm = ...` lines are commented out. Uncomment them (or run them first), or the heatmap plot raises a `NameError`.
- **Runtime:** preprocessing takes under a minute. The 100-epoch NN training takes a few minutes on CPU.
- **Reproducibility:** sampling and splits use `random_state=42`, but the NN has no fixed seed, so its scores will vary slightly between runs.

## Repository layout

```
.
├── Agnihotri, Kavyanjali - NLP.ipynb        # all code + outputs
├── Agnihotri, Kavyanjali -NLP_report.pdf    # written report
└── instructions/
    └── todo.md                              # summary of the assignment task
```

## Possible improvements

- Use early stopping for the NN. Validation loss is lowest at epoch 17 (0.135) and climbs to 0.24 by epoch 100, while training accuracy reaches about 99.8%, which means the model overfits.
- Try the `title` field as an extra feature. It is often very informative, for example "(novel)" or "(album)".
- Compare against a fine-tuned transformer such as DistilBERT.

## Author

Kavyanjali Agnihotri
