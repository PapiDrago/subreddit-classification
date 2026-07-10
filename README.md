# Automatic Subreddit Classification of Reddit Posts

Classifiers that assign one of ten subreddits to a Reddit post from its title, body, comment count, and score. Built for a client who wants to auto-route brand-new submissions to the correct subreddit.

## Results

| Model | Train acc. | Test acc. | Macro-F1 |
|---|---|---|---|
| Multinomial NB (α=0.3) | 0.757 | 0.739 | 0.735 |
| Logistic regression (C=1) | 0.859 | 0.775 | 0.776 |
| Linear SVM (C=0.1) | 0.847 | 0.779 | 0.778 |
| **MLP (early-stopped)** | 0.861 | **0.804** | 0.804 |

All models use bag-of-words text features plus standardized numeric features, except naive Bayes (text only).

## Key findings

**Text drives the signal.** Bag-of-words over title + body takes a 10-class problem from 0.10 chance accuracy to 0.74–0.78 with linear models and 0.80 with an MLP. The remaining ceiling is set by the representation (word counts, no order or semantics), not by model capacity — neither regularization nor vocabulary pruning closes the train/test gap without hurting test accuracy.

**Do `num_comments` and `score` help?** Yes, but not usefully in production. They add ~1.8 accuracy points on historical posts, where engagement has accumulated over the post's lifetime. But a brand-new post — the object the company actually needs to classify — starts at the cold-start default (0 comments, score 1), where both features are near-constant across all subreddits. The historical gain evaporates at the point of use, so the deployed model should rely on text alone.

**The `[deleted]`/`[removed]` tags leak.** These moderation tags are produced after submission and vary strongly by subreddit, making them highly predictive but causally downstream of the label and unavailable on new posts. They're stripped from the vocabulary and only used, as an isolated status feature, to quantify the leak (+1.8 points) — not in the deployment model.

**One error persists everywhere.** `news` is the weakest class in all four models and is confused with `business` at an almost identical rate (0.21–0.24) regardless of model family. This convergence across generative, linear, and non-linear models shows the failure is a property of the dataset — `news` has no distinctive vocabulary of its own — not of any single algorithm.

## Data

- 90,000 training posts / 10,000 test posts, perfectly balanced across 10 classes (Art, Jokes, Music, anime, business, cats, gaming, movies, news, politics).
- Fields: `subreddit`, `num_comments`, `score`, `title`, `text` (pipe-separated).
- Only 12% of posts carry usable body text; the rest are empty or contain `[deleted]`/`[removed]`.

## Pipeline

1. **Text**: title + body concatenated, tags/URLs stripped, Snowball-stemmed, bag-of-words with `min_df=10` (8,124-word vocabulary).
2. **Numeric**: `log(1+x)` transform on `num_comments` and `score`, then standardized (fit on train only).
3. **Status**: body presence encoded as a 4-way categorical feature (present/empty/deleted/removed) — excluded from the deployment model due to leakage.
4. **Models**: multinomial naive Bayes, multinomial logistic regression, linear SVM (one-vs-rest), and an MLP (256 ReLU units, dropout 0.5, early stopping), all tuned by 5-fold cross-validation.

## Repository contents

- `subreddit_classification_report.tex` — full write-up (EDA, feature extraction, model comparisons, ablation study, error analysis).

## Author

Claudio Guarrasi — Department of Electrical, Computer and Biomedical Engineering, University of Pavia.

## AI-tool usage

AI assistance (Anthropic's Claude) was used for pipeline planning, code drafting/debugging, results interpretation, and report drafting. All experiments were executed and all results verified by the author.
