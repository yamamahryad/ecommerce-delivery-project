# Ecommerce Delivery Delay Prediction (DPD601)

تحليل التجارة الإلكترونية والتنبؤ بتأخر تسليم الطلبات.
Course: Programming for Data Science (DPD601), Faculty of Informatics Engineering.

## Team
alyamamah_244054
sara_219301
maher_259930
mohamad_245054
jassem_189355

## Problem
Predict, at order approval time, whether an order will be delivered after its
estimated delivery day (`is_late`). Prediction unit: one order (`order_id`).

## Data and license
Olist, & Sionek, A. (2018). Brazilian E-Commerce Public Dataset by Olist. Kaggle.
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
License: CC BY-NC-SA 4.0 (attribution, non-commercial use, share-alike).
Historical Brazilian data: results must not be generalized to other markets.
Raw files in `data/raw` are never modified; all processed files are produced by code.

## Environment
- Python: (3.14.3)
- Libraries: see `requirements.txt` (NumPy, pandas, Matplotlib, seaborn, scikit-learn)
- Tools: Git, pytest

## Setup (Windows PowerShell)
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run everything with one command
    python main.py                # full pipeline (about N minutes)
    python main.py --skip-eda     # skip EDA figures
    python main.py --from-scratch # delete data/processed first

## Run tests
    python -m pytest -q

## Pipeline steps
| Script | What it does |
|---|---|
| run_day1.py | audit of the 7 raw tables, key checks, ERD |
| run_day2.py | cleaning with a documented decision log |
| run_day3.py | aggregation to order level and one-row-per-order table |
| run_day4.py | target, eligibility, 12 derived features, feature dictionary |
| run_day5.py / run_day6.py | EDA questions 1-6 and 12 figures |
| run_day7.py | temporal split, preprocessing pipeline, baseline and two models |
| run_day8_validation.py | threshold and model selection on validation, freezes decisions |
| run_day8_test.py | single final test evaluation and error analysis |

## Folder structure
    data/raw         original tables (unmodified)
    data/processed   generated files (not in Git)
    src/             reusable modules
    tests/           automated tests
    notebooks/       EDA notebooks
    outputs/figures  all figures
    outputs/tables   all result tables, experiment log, final metrics
    docs/            daily notes, code review, contributions

## Key design rules
- Target: late if delivery DAY is after estimated DAY; excluded orders get no target (never 0).
- Forbidden inputs: order_status, delivery dates, actual duration, delay.
- Imputation and scaling are learned on the training split only (sklearn Pipeline).
- Chronological split by approval time; the test set is evaluated once with frozen decisions.

## Known limitations
- Item, payment and estimated-date details are assumed available at approval time (brief's assumption).
- The selection of eligible orders (delivered only) biases the latest months.
- The test period is only 3 months with a lower late rate than training.
- GitHub Repository Link: https://github.com/yamamahryad/ecommerce-delivery-project
Reviewed by: yamamahryad
