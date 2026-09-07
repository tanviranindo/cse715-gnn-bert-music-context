# Excluded submission

`afrin-heram-*.json` carries the same 29 ratings as `md-sakibur-rahman-*.json`,
value for value. The two are not independent judgements.

Cause: until 2026-09-07 the study restored saved progress by device rather than
by person. Md. Sakibur Rahman completed the study on a device at 22:37–22:47;
Afrin Heram opened the same browser at 22:57, was offered the saved answers,
and submitted them under her own name two and a half minutes later. The
submissions carry different names, start times and durations, so nothing inside
either file reveals the problem — only comparing the rating vectors does.

The bug is fixed (entering a different name now clears saved state), but this
submission is excluded rather than counted, because including it would report
one person's judgements as two.

Excluding it leaves five raters, which is the minimum the specification
requires. It is kept here rather than deleted so the decision is auditable.
