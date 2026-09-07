# Task 4 listening study

The site that collects the human evaluation the specification requires (§6:
at least five listeners rating each retrieved clip 1–5 for whether it matches
its query caption).

`index.html` is generated — do not edit it by hand:

```bash
python src/human_eval.py build \
    --examples results/retrieval_examples/task4_examples_gnnlr.json \
    --endpoint /api/submit \
    --out-dir infra/listening-study
```

`api/submit.js` validates one rater's submission and stores it in Vercel Blob
under `ratings/`. It stores what was sent and nothing else: a clip the rater
could not listen to arrives as `null` and stays `null`.

Pull the submissions back and score them:

```bash
python src/fetch_ratings.py          # Blob -> results/human_eval/ratings/
python src/human_eval.py score --examples results/retrieval_examples/task4_examples_gnnlr.json
```

`score` refuses to run below five raters rather than report a partial study.
