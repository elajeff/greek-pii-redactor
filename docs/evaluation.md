# Evaluation

Exact-span precision and recall: a span counts only if start, end and kind all match. Every number here comes from the script named with it, run on my machine (RTX 5070). The tagger is the one trained with `train/train_tagger.py --seed 1` on WikiANN plus the synthetic train file (`eval/make_synthetic.py --seed 11 --split train --count 2000`).

## Detectors

`eval/evaluate.py` runs four detectors over the same sentences:

- rules only: `find_ids` from `rules.py`, which finds IDs and no names
- Presidio with the Greek spaCy model `el_core_news_lg`. Its phone recognizer is set to the GR region, because with its default regions it reads ΑΜΚΑ and ΑΦΜ numbers as phone numbers.
- the XLM-R tagger for names plus the rules for IDs (`--model-dir`)
- qwen3:4b-instruct through Ollama for names plus the rules for IDs (`--ollama-model`). It asks the model to list the names in each sentence (seed 7, temperature 0) and counts every place a listed name appears in the text.

```powershell
.\.venv\Scripts\python.exe -m spacy download el_core_news_lg
.\.venv\Scripts\python.exe eval/download_ner.py
.\.venv\Scripts\python.exe eval/make_synthetic.py --seed 7 --split test --count 2000 --out data/synthetic_test.jsonl
.\.venv\Scripts\python.exe eval/evaluate.py --model-dir C:\Users\you\weights\xlmr-person --ollama-model qwen3:4b-instruct
```

`el_core_news_lg` is CC BY-NC-SA 3.0, so it is only used here for comparison and isn't part of the package.

## Greek sentences of the synthetic test file

1515 of the 2000 sentences are Greek. Precision / recall:

| Kind | Rules only | Presidio + spaCy el | qwen3:4b-instruct + rules | XLM-R tagger + rules |
| --- | --- | --- | --- | --- |
| Names (1515) | finds none | 0.986 / 0.698 | 0.981 / 0.869 | 1.000 / 0.987 |
| ΑΦΜ (232) | 1.000 / 1.000 | finds none | 1.000 / 1.000 | 1.000 / 1.000 |
| ΑΜΚΑ (270) | 1.000 / 1.000 | finds none | 1.000 / 1.000 | 1.000 / 1.000 |
| IBAN (483) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Phone (769) | 1.000 / 1.000 | 1.000 / 0.750 | 1.000 / 1.000 | 1.000 / 1.000 |
| Email (263) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

The sentences come from my own templates, and the tagger was trained on sentences from the same generator (other names and other templates), so its score here is a best case. The test IDs are written in the formats I generate, which are the formats the rules look for, so the rules' perfect score shows they match those formats, not how they do on IDs typed by people.

The script prints the same table for the 485 English sentences. Presidio runs there with the Greek spaCy model too, so its English row says little about Presidio.

## WikiANN el

2000 sentences of the WikiANN `el` test split, sampled with seed 7, with 894 person names. Precision / recall:

| Rules only | Presidio + spaCy el | qwen3:4b-instruct | XLM-R tagger |
| --- | --- | --- | --- |
| finds none | 0.639 / 0.649 | 0.647 / 0.628 | 0.909 / 0.932 |

`download_ner.py` drops WikiANN test sentences that also appear in its train or validation split. Many person names still appear in both, which is how WikiANN is built, so the WikiANN score is for names the tagger has partly seen.

## Speed

Wall time for the 2000 WikiANN sentences, average of three runs of `eval/evaluate.py --model-dir <tagger>`. The CPU row is the same command with `CUDA_VISIBLE_DEVICES=-1`. The LLM's time is the one run with `--ollama-model` (4260.2 s); I didn't repeat it.

| | Rules only | Presidio + spaCy el | qwen3:4b-instruct | XLM-R tagger |
| --- | --- | --- | --- | --- |
| GPU | 0.0 s | 7.9 s | 4260.2 s | 13.5 s |
| CPU | 0.0 s | 9.0 s | not run | 74.0 s |

The single runs behind the averages: Presidio 8.2, 7.7, 7.7 s on the GPU and 8.2, 8.5, 10.3 s on the CPU; the tagger 10.1, 20.7, 9.7 s on the GPU and 75.6, 77.0, 69.5 s on the CPU. Presidio and spaCy run on the CPU in both rows, so the GPU and CPU rows differ for it only by run-to-run noise. One tagger run on the GPU took twice as long as the other two and I haven't found why, so take the times as rough.

## Repeatability

The tagger gave the same counts on the GPU and on the CPU, and so did the rules and Presidio. Ollama's replies aren't exactly repeatable even at temperature 0 with a fixed seed: over three runs the LLM's true positives on the whole synthetic file went from 1796 to 1801, and on WikiANN from 553 to 561.

## Test data and licences

`eval/make_synthetic.py` writes invented Greek and English sentences with labelled names and IDs, about three quarters of them Greek. The same seed gives the same file, and train and test use different names and sentence templates. `eval/download_ner.py` downloads real Greek text with person labels into `data/`, and `eval/download_names.py` the Wiktionary case tables of Greek names. Nothing from these datasets is stored in the repo, and the weights are on Hugging Face, not in the repo.

| Data or model | Used for | Licence on its own page |
| --- | --- | --- |
| [unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann) (`el`) | training and testing | unknown; labels are machine-generated from Wikipedia |
| Greek Wiktionary, through the [kaikki.org](https://kaikki.org/elwiktionary/rawdata.html) extract | the name forms in `--names-dir`, and checking the inflection rules | Wiktionary text is CC BY-SA 4.0; the kaikki.org page names no other licence |
| [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | base model of the tagger | MIT |
| [elajeff/greek-pii-redactor-ner](https://huggingface.co/elajeff/greek-pii-redactor-ner) | the published tagger | MIT, set by me; its WikiANN training data has no known licence |
| [el_core_news_lg](https://spacy.io/models/el#el_core_news_lg) | the Presidio baseline | CC BY-NC-SA 3.0 |
| [qwen3:4b-instruct](https://ollama.com/library/qwen3) | the LLM baseline and placeholder survival | Apache 2.0 |

## Placeholder survival

An LLM can rewrite a placeholder (drop the brackets, translate it) and then restore has nothing to match. `eval/survival.py` sends redacted test sentences to a local Ollama model, asks for a polite rewrite, and counts how many placeholders come back unchanged. Temperature is 0 and the seed is set, but Ollama's replies can still differ slightly from run to run.

```powershell
ollama pull qwen3:4b-instruct
.\.venv\Scripts\python.exe eval/survival.py --model-dir C:\Users\you\weights\xlmr-person
```

Without `--model-dir` only IDs are redacted. On one sentence template the 4B model then wrapped the whole reply in brackets and dropped the placeholder's own brackets, so the ID-only numbers and the with-names numbers are not comparable.
