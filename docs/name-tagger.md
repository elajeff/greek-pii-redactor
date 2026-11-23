# Name tagger

`train/train_tagger.py` fine-tunes [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) (MIT) to tag person names. It trains on the WikiANN `el` train split plus 2000 synthetic sentences and keeps the epoch with the best F1 on the WikiANN validation split, stopping after 2 epochs without a gain (15 at most). `--seed`, `--without-synthetic` and `--train-fraction` change the seed, drop the synthetic sentences and train on a seeded share of the records; each epoch's numbers go to `epochs.json` in the output folder.

## Training

It needs a CUDA GPU. The default PyPI torch wheel on Windows is CPU-only, so install the CUDA build first:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.\.venv\Scripts\python.exe -m pip install -e ".[train,eval]"
.\.venv\Scripts\python.exe eval/download_ner.py
.\.venv\Scripts\python.exe eval/make_synthetic.py --seed 11 --split train --count 2000 --out data/synthetic_train.jsonl
.\.venv\Scripts\python.exe train/train_tagger.py --out C:\Users\you\weights\xlmr-person
```

For example, `.\.venv\Scripts\python.exe train/train_tagger.py --out C:\Users\you\weights\wikiann-only --without-synthetic --train-fraction 0.5 --seed 1` trains on half of WikiANN with seed 1.

The weights stay outside the repo and none are published. Pass the folder to `--model-dir` of the eval scripts, the CLI and the proxy.

## What the synthetic sentences add

`python eval/tagger_errors.py --model-dir <seed 1> <seed 2> <seed 3>`, once for taggers trained with `train/train_tagger.py --seed N` and once for `--without-synthetic --seed N`. Mean over the three seeds, standard deviation in brackets. The probes edit the Greek names in the synthetic test file: lowercase the name, keep only the surname of a two-word name, or drop the article before it.

| | WikiANN + synthetic | WikiANN only |
| --- | --- | --- |
| WikiANN F1 | 0.914 (0.005) | 0.915 (0.003) |
| Recall, lowercased name | 0.769 (0.043) | 0.327 (0.036) |
| Recall, surname alone | 0.956 (0.043) | 0.493 (0.105) |
| Recall, no article | 1.000 (0.001) | 0.824 (0.021) |
| greek_legal_ner test, PERSON words inside one predicted span | 0.769 (0.051) | 0.723 (0.079) |
| Same, all-caps words only | 0.728 (0.063) | 0.696 (0.110) |

The synthetic sentences don't change the WikiANN score and teach the tagger lowercase names and surnames on their own. The probes are edits of the generator's own sentences, so they likely favour the tagger trained on the generator's sentences.

`eval/tagger_errors.py` also prints the errors split into wrong boundary, missed and extra, with examples, and recall on names seen and not seen in training. `eval/label_flags.py` lists WikiANN labels a trained tagger disagrees with, for checking by hand.

`download_ner.py` drops WikiANN test sentences that also appear in its train or validation split. Many person names still appear in both, which is how WikiANN is built, so the WikiANN score is for names the model has partly seen.
