# Task 4 Listening Study

This directory contains the static listening-study page and its submission
endpoint. The study asks listeners to rate whether retrieved audio matches a
query caption on a scale from 1 to 5.

## Build The Study Page

The page is generated from the committed retrieval examples:

```bash
python src/human_eval.py build \
  --examples results/retrieval_examples/task4_examples_gnnlr.json \
  --endpoint https://gnn-bert-listening-study.vercel.app/api/submit \
  --out-dir infra/listening-study
```

Do not edit `index.html` manually.

## Score Responses

Participant responses are intentionally not stored in this public repository.
If authorized response files are available locally, place them under
`results/human_eval/ratings/` and run:

```bash
python src/fetch_ratings.py
python src/human_eval.py score \
  --examples results/retrieval_examples/task4_examples_gnnlr.json
```

The aggregate result used by the project is recorded in
`results/metrics_task4_human.json`: 174 ratings from six listeners, with a mean
score of `1.79 +/- 1.36`.
