import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# === CONFIGURATION ===
CSV_FILE = 'sp500_cik.csv'
CHECKPOINT_FILE = 'checkpoint_8k_html.json'
SAVE_DIR = 'eight_k_htm'
HEADERS = {'User-Agent': 'Your Name your.email@example.com'}
DATE_CUTOFF = datetime(2000, 1, 1)

# === SETUP ===
os.makedirs(SAVE_DIR, exist_ok=True)
df = pd.read_csv(CSV_FILE)
df['CIK'] = df['CIK'].astype(str).str.zfill(10)
cik_to_symbol = dict(zip(df['CIK'], df['Symbol']))
ciks = df['CIK'].unique()

# Load checkpoint
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint = json.load(f)
else:
    checkpoint = {}

# === MAIN LOOP ===
for cik in tqdm(ciks, desc="Processing CIKs"):
    if cik in checkpoint and checkpoint[cik]['status'] == 'done':
        continue

    symbol = cik_to_symbol.get(cik, 'UNKNOWN')

    try:
        url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()

        filings = data.get('filings', {}).get('recent', {})
        forms = filings.get('form', [])
        dates = filings.get('filingDate', [])
        acc_nos = filings.get('accessionNumber', [])

        saved = 0
        for form, date_str, acc in zip(forms, dates, acc_nos):
            if form != '8-K':
                continue
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                if date_obj < DATE_CUTOFF:
                    continue
            except:
                continue

            acc_no_nodash = acc.replace('-', '')
            txt_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_nodash}/{acc}.txt'

            r_txt = requests.get(txt_url, headers=HEADERS)
            if r_txt.status_code != 200:
                continue

            text = r_txt.text
            docs = text.split('<DOCUMENT>')[1:]
            for doc in docs:
                try:
                    filename = doc.split('<FILENAME>')[1].split('\n', 1)[0].strip()
                    if not filename.endswith('.htm'):
                        continue

                    # New file name
                    new_filename = f"{symbol}_{cik}_{date_str.replace('-', '')}.htm"
                    save_path = os.path.join(SAVE_DIR, new_filename)
                    if os.path.exists(save_path):
                        continue

                    file_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_nodash}/{filename}'
                    r_file = requests.get(file_url, headers=HEADERS)
                    if r_file.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(r_file.content)
                        saved += 1
                except Exception:
                    continue

        checkpoint[cik] = {'status': 'done', 'saved_htm': saved}

    except Exception as e:
        checkpoint[cik] = {'status': 'error', 'message': str(e)}

    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
