"""Extract a small ticker/date subset from huge FNSPID news CSV without loading full file."""
import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--input',required=True);p.add_argument('--out',required=True)
p.add_argument('--tickers',default='AAPL,MSFT,NVDA,AMZN')
p.add_argument('--start',default='2019-01-01');p.add_argument('--end',default='2020-12-31')
a=p.parse_args()
want={x.strip().upper() for x in a.tickers.split(',')}
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():out.unlink()
written=0
for chunk in pd.read_csv(a.input, chunksize=30000,low_memory=False, on_bad_lines='skip'):
    cols={c.lower():c for c in chunk.columns}
    t=next((cols[k] for k in ['stock_symbol','ticker','stock','symbol'] if k in cols),None)
    d=next((cols[k] for k in ['date','published_at','datetime'] if k in cols),None)
    if not t or not d:raise ValueError(f'Unrecognized FNSPID columns: {list(chunk.columns)}')
    dates=chunk[d].astype(str).str[:10]
    rows=chunk[chunk[t].astype(str).str.upper().isin(want)&dates.between(a.start,a.end)].copy()
    if len(rows):
        rows.to_csv(out, mode='a',header=(written==0),index=False)
        written+=len(rows)
print(f'Wrote {written} filtered headlines to {out}')
