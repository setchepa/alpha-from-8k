# alpha-from-8k

A transformer-based framework to extract predictive signals from SEC Form 8-K filings and test for alpha beyond Fama-French risk factors.

## Features

- Downloads 8-K HTML documents from EDGAR using CIKs
- Vectorizes financial text with FinBERT, BERT, or Ollama embeddings
- Trains transformer models to predict short-term returns
- Aggregates predictions and evaluates alpha using asset pricing regressions

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/alpha-from-8k.git
cd alpha-from-8k
pip install -r requirements.txt
```

## Step 1: Download 8-K Filings

Ensure `sp500_cik.csv` is present in the root directory. It must have `CIK` and `Symbol` columns.

```bash
python 8k_download.py
```

Files will be saved to the `eight_k_htm/` directory.

## Step 2: Run the Main Pipeline

You must provide the following (not included due to data restrictions):

- `sp500_data.csv`: Stock prices and market caps
- `ff_daily_factors.csv`: Daily Fama-French 5 factors + momentum
- `ff_monthly_factors.csv`: Monthly Fama-French 5 factors + momentum

Then run:

```bash
python MainCode.py
```

## Paper

This repository supports the analysis conducted in:

**"Beyond Factors: AI on SEC Text"**  
Daniel Buckman, Santiago Etchepare, Michael Logan, Livia Mucciolo  
University of Chicago Booth School of Business, 2025

See `Final Project.pdf` for a detailed write-up.

## License

MIT License
