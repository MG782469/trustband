#!/usr/bin/env python3
"""TrustBand pre-registered-style pilot. Research only; no trading use.

REQUIRES actual real ticker news+prices CSVs. There is NO synthetic result
that should be placed in a competition abstract.

News fields (case-insensitive alternatives): ticker/stock, headline/title,
date/published_at/time; finbert_sentiment in [-1,1] OR score FinBERT locally.
Prices: ticker/stock, date, close or adj_close, volume.
"""
import argparse, json, math, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

SEED=1729
ALPHA=0.10
PRICE_FEATURES=['lag_return','vol_5','vol_20','volume_ratio','log_volume']
NEWS_FEATURES=['news_count','story_count','dup_rate','mean_sent','mean_abs_sent','sent_std','max_abs_sent']

def standardize(df, variants, label):
    df=df.copy(); found=[]
    for target, opts in variants.items():
        existing=next((k for opt in opts for k in df.columns if k.lower().strip().replace(' ','_')==opt),None)
        if existing is not None: df=df.rename(columns={existing:target});found.append(target)
    required=list(variants)
    missing=set(required)-set(found)
    if missing: raise ValueError(f'{label}: missing {missing}; columns={list(df.columns)}')
    return df

def ticker_norm(t): return str(t).strip().upper()

def cluster_key(s):
    # Conservative near-duplicate normalization. Distinct semantic stories not merged by design.
    s=str(s).lower();s=re.sub(r'^(breaking|update|updated|exclusive)\s*[:\-]\s*','',s)
    s=re.sub(r'https?://\S+','',s)
    return re.sub(r'[^a-z0-9]+',' ',s).strip()

def finbert_scores(headlines):
    # Model weights must be downloaded explicitly. No alternative sentiment scorer is substituted.
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as ex:
        raise RuntimeError('No finbert_sentiment column; install transformers, torch and download ProsusAI/finbert') from ex
    tok=AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model=AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').eval()
    labels={i:str(v).lower() for i,v in model.config.id2label.items()}
    uniq=list(dict.fromkeys(headlines))
    scores={}
    for ix in range(0,len(uniq),24):
        batch=uniq[ix:ix+24]
        x=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=160)
        with torch.inference_mode(): probs=model(**x).logits.softmax(-1).numpy()
        for s,p in zip(batch,probs):
            scores[s]=float(sum(float(v)*(1 if labels[j]=='positive' else -1 if labels[j]=='negative' else 0) for j,v in enumerate(p)))
    return scores

def prepare_news(path, valid_tickers, start=None, end=None):
    df=pd.read_csv(path,low_memory=False)
    # explicit schema, no guessing headline text from unrelated fields
    df=standardize(df, {
        'ticker':['ticker','stock','symbol','stock_symbol'],
        'headline':['headline','title','article_title'],
        'published_at':['published_at','datetime','date','published','publication_date'],
    },'News')
    df['ticker']=df['ticker'].map(ticker_norm)
    publisher_col=next((c for c in df.columns if c.lower() in ['publisher','source','source_name','news_source']),None)
    df['publisher_present']=(df[publisher_col].notna() & (df[publisher_col].astype(str).str.strip()!='')) if publisher_col else False
    df=df[df.ticker.isin(valid_tickers)].copy()
    df=df[df.headline.notna() & df.published_at.notna()].copy()
    text=df.published_at.astype(str).str.strip()
    # IMPORTANT: a date-only or midnight placeholder denotes a source calendar date,
    # NOT a midnight UTC instant. Converting midnight UTC to NY first moves many
    # records to the previous calendar date and silently defeats the safety lag.
    source_date=pd.to_datetime(text.str.extract(r'^(\d{4}-\d{2}-\d{2})',expand=False),errors='coerce')
    date_only=text.str.fullmatch(r'\d{4}-\d{2}-\d{2}')
    midnight_placeholder=text.str.match(r'^\d{4}-\d{2}-\d{2}[ T]00:00:00(?:\.0+)?(?:Z|\+00:00|\s*UTC|\s*GMT)?$',case=False)
    tz_explicit=text.str.contains(r'(?:Z|UTC|GMT|[+-]\d{2}:?\d{2})$',case=False,regex=True)
    ambiguous=(date_only|midnight_placeholder|~tz_explicit).fillna(True)
    dt=pd.to_datetime(text,errors='coerce',utc=True,format='mixed')
    valid=source_date.notna() & (ambiguous|dt.notna())
    df=df.loc[valid].copy();dt=dt.loc[valid];ambiguous=ambiguous.loc[valid];source_date=source_date.loc[valid]
    # Preserve original source calendar date for all ambiguous entries.
    # Only verified timezone-aware precise timestamps are converted to New York.
    ny=dt.dt.tz_convert('America/New_York')
    precise_local_date=ny.dt.tz_localize(None).dt.normalize()
    local_date=precise_local_date.where(~ambiguous, source_date)
    after_close=(ny.dt.hour>=16)   # at/after 16:00 is not before market close
    delay=(ambiguous|after_close).astype(int)
    df['news_local_date']=local_date
    df['eligible_after_day']=local_date+pd.to_timedelta(delay,unit='D')
    df['time_ambiguous']=ambiguous.to_numpy()
    if start:
        keep=df.news_local_date >= pd.Timestamp(start)-pd.Timedelta(days=4)
        df=df.loc[keep].copy()
    if end:
        keep=df.news_local_date <= pd.Timestamp(end)+pd.Timedelta(days=4)
        df=df.loc[keep].copy()
    df['story_key']=df.headline.map(cluster_key)
    df=df[df.story_key.str.len()>5].copy()
    if 'finbert_sentiment' in df.columns:
        df['sentiment']=pd.to_numeric(df.finbert_sentiment,errors='coerce')
    elif 'sentiment' in df.columns:
        raise ValueError('Generic sentiment column cannot be presumed FinBERT. Rename only VERIFIED FinBERT output to finbert_sentiment.')
    else:
        mapping=finbert_scores(df.headline.astype(str).tolist())
        df['sentiment']=df.headline.map(mapping)
    df=df[df.sentiment.between(-1,1)].copy()
    return df

def prepare_prices(path,tickers,start=None,end=None):
    df=pd.read_csv(path,low_memory=False)
    df=standardize(df,{'ticker':['ticker','stock','symbol','stock_symbol'], 'date':['date','datetime'], 'close':['adj_close','adj_close_price','adjusted_close','close'], 'volume':['volume','vol']},'Prices')
    df['ticker']=df.ticker.map(ticker_norm);df=df[df.ticker.isin(tickers)].copy()
    df['date']=pd.to_datetime(df.date,errors='coerce',utc=True).dt.tz_localize(None).dt.normalize()
    df['close']=pd.to_numeric(df.close,errors='coerce');df['volume']=pd.to_numeric(df.volume,errors='coerce')
    df=df.dropna(subset=['date','close','volume']);df=df[df.close>0];df=df.sort_values(['ticker','date']).drop_duplicates(['ticker','date'])
    if start:df=df[df.date>=pd.Timestamp(start)]
    if end:df=df[df.date<=pd.Timestamp(end)]
    grp=df.groupby('ticker',sort=False)
    df['lag_return']=grp.close.transform(lambda v:np.log(v).diff()) # current close known at market close
    df['vol_5']=grp.lag_return.transform(lambda s:s.shift(1).rolling(5,min_periods=5).std()) # before current close
    df['vol_20']=grp.lag_return.transform(lambda s:s.shift(1).rolling(20,min_periods=20).std())
    df['volume_ratio']=grp.volume.transform(lambda s:s/s.shift(1).rolling(20,min_periods=20).mean())
    df['log_volume']=np.log1p(df.volume)
    df['target']=grp.close.transform(lambda s:np.log(s.shift(-1)/s))
    df=df.dropna(subset=PRICE_FEATURES+['target']).copy().reset_index(drop=True)
    return df

def assign_to_trading_day(news, prices):
    # Merge_asof by ticker forwards: first trading date >= availability date.
    calendars=prices[['ticker','date']].drop_duplicates().sort_values(['date','ticker'])
    items=news.sort_values(['eligible_after_day','ticker']).copy()
    # asof sorting of merge key first required, then by ticker
    joined=pd.merge_asof(items,calendars,left_on='eligible_after_day',right_on='date',by='ticker',direction='forward',allow_exact_matches=True)
    return joined.dropna(subset=['date']).copy()

def aggregate(g):
    if len(g)==0:return dict(zip(NEWS_FEATURES,[0.]*len(NEWS_FEATURES)))
    v=g.sentiment.to_numpy(dtype=float)
    return {'news_count':float(len(g)), 'story_count':float(g.story_key.nunique()),
    'dup_rate':float(1-g.story_key.nunique()/len(g)),'mean_sent':float(v.mean()),
    'mean_abs_sent':float(np.mean(np.abs(v))),'sent_std':float(v.std()),'max_abs_sent':float(np.abs(v).max())}

def build_features(price, grouped):
    # Only pre-existing news grouped by observed cutoff is included
    agg_rows=[]
    for key,g in grouped.items():
        v=aggregate(g);v.update({'ticker':key[0],'date':key[1]});agg_rows.append(v)
    news_agg=pd.DataFrame(agg_rows,columns=['ticker','date']+NEWS_FEATURES)
    if news_agg.empty:news_agg=pd.DataFrame(columns=['ticker','date']+NEWS_FEATURES)
    df=price.merge(news_agg,on=['ticker','date'],how='left')
    for c in NEWS_FEATURES: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
    return df

def model_features(df, ticker_list, cols):
    X=df[cols].astype(float).copy()
    for t in ticker_list[:-1]:X['stock_'+t]=(df.ticker==t).astype(int)
    return X

def split_dates(df):
    unique=np.array(sorted(df.date.unique()))
    if len(unique)<85:raise ValueError('Not enough distinct trading days for four chronological partitions (need >=85).')
    ai,bi,ci=[int(len(unique)*f) for f in [0.55,0.70,0.85]]
    a,b,c=unique[ai],unique[bi],unique[ci]
    # Price label at trading day t is the close at t+1. Purge the last
    # decision date in each earlier segment, whose label lies in the next
    # segment. Otherwise the first validation/calibration/test day's close
    # is already known to the preceding segment.
    train=df[df.date<unique[ai-1]]
    val=df[(df.date>=a)&(df.date<unique[bi-1])]
    cal=df[(df.date>=b)&(df.date<unique[ci-1])]
    test=df[df.date>=c]
    sections=[train,val,cal,test]
    if min(len(s) for s in sections)<30:raise ValueError('Insufficient examples in some split.')
    return sections, [str(pd.Timestamp(x).date()) for x in [a,b,c]]

def score_predictions(model, df, ticker_list, features):
    return model.predict(model_features(df,ticker_list,features))

def leave_one_story_sensitivity(base, model, price_row, grouped, ticker_list):
    key=(price_row.ticker,price_row.date)
    g=grouped.get(key)
    if g is None or g.story_key.nunique()<2:return float('nan')
    diffs=[]
    for story in g.story_key.unique():
        clone=price_row.to_frame().T.copy()
        for c,v in aggregate(g[g.story_key!=story]).items():clone[c]=v
        change=float(score_predictions(model,clone,ticker_list,PRICE_FEATURES+NEWS_FEATURES)[0]-base)
        diffs.append(abs(change))
    return max(diffs)/max(float(price_row.vol_20),0.002)

def fragility_for(df,preds,model,grouped,ticker_list):
    out=np.empty(len(df))
    for ix, (_, row) in enumerate(df.iterrows()):
        out[ix]=leave_one_story_sensitivity(preds[ix],model,row,grouped,ticker_list)
    return out

def routed(pred_price,pred_news,fragility,thr,story_count):
    eligible=(story_count>=2)&np.isfinite(fragility)&(fragility<=thr)
    return np.where(eligible,pred_news,pred_price),eligible

def quantile(abs_err,alpha=ALPHA):
    n=len(abs_err)
    if n<30:raise ValueError('Not enough calibration examples: '+str(n))
    k=min(n,math.ceil((n+1)*(1-alpha)))
    return float(np.partition(np.asarray(abs_err),k-1)[k-1])

def metrics(y,p,q):
    y=np.asarray(y);p=np.asarray(p)
    cover=np.abs(y-p)<=q
    return {'n':len(y),'coverage':round(float(cover.mean()),4),'mean_width':round(float(2*q),6),
      'mae':round(float(np.abs(y-p).mean()),6),'q':round(float(q),6)}

def attack_news(groups,period,seed=SEED,frac=0.30,copies=10):
    r=np.random.default_rng(seed)
    attacked={k:v.copy() for k,v in groups.items()}
    usable=[(k,v) for k,v in groups.items() if k[1] in period and v.story_key.nunique()>=2]
    count=max(1,int(len(usable)*frac)) if usable else 0
    keys=[k for k,_ in usable]
    chosen=set(r.choice(len(keys), size=min(count,len(keys)),replace=False).tolist()) if keys else set()
    selected=[]
    for i,(k,g) in enumerate(usable):
        if i not in chosen:continue
        # attacker sees news sentiment/features, not next-day target
        largest=g.groupby('story_key').sentiment.mean().abs().idxmax()
        source=g[g.story_key==largest].iloc[0].copy()
        clones=pd.DataFrame([source]*copies)
        clones['headline']='UPDATE: '+str(source.headline)
        clones['story_key']=clones.headline.map(cluster_key)
        attacked[k]=pd.concat([g,clones],ignore_index=True)
        selected.append(k)
    return attacked, selected

def audit(news,prices):
    return {'news_rows':len(news),'news_ambiguous_time_or_zone_rate':round(float(news.time_ambiguous.mean()),4),
      'publisher_nonmissing_rate':round(float(news.publisher_present.mean()),4),
      'normalized_duplicate_rate':round(float(1-len(news[['ticker','news_local_date','story_key']].drop_duplicates())/len(news)),4),
      'tickers':sorted(news.ticker.unique().tolist()), 'first_news_date':str(news.news_local_date.min().date()),
      'last_news_date':str(news.news_local_date.max().date()), 'price_days':len(prices),
      'news_unique_story_keys':int(news[['ticker','story_key','news_local_date']].drop_duplicates().shape[0])}

def run(args):
    tickers=[x.strip().upper() for x in args.tickers.split(',')]
    prices=prepare_prices(args.prices,tickers,args.start,args.end)
    news=prepare_news(args.news,tickers,args.start,args.end)
    tagged=assign_to_trading_day(news,prices)
    if len(tagged)==0:raise ValueError('No news matched price calendar, check date/ticker coverage')
    groups={(t,d):g for (t,d),g in tagged.groupby(['ticker','date'],sort=False)}
    data=build_features(prices,groups)
    (train,val,cal,test), date_bounds=split_dates(data)
    mp=LGBMRegressor(random_state=SEED,n_estimators=150,max_depth=4,num_leaves=12,learning_rate=.04,verbosity=-1,n_jobs=1)
    mn=LGBMRegressor(random_state=SEED,n_estimators=150,max_depth=4,num_leaves=12,learning_rate=.04,verbosity=-1,n_jobs=1)
    mp.fit(model_features(train,tickers,PRICE_FEATURES),train.target)
    mn.fit(model_features(train,tickers,PRICE_FEATURES+NEWS_FEATURES),train.target)
    vp=score_predictions(mp,val,tickers,PRICE_FEATURES)
    vn=score_predictions(mn,val,tickers,PRICE_FEATURES+NEWS_FEATURES)
    vd=fragility_for(val,vn,mn,groups,tickers)
    finite=vd[np.isfinite(vd)]
    if len(finite)<15: raise ValueError('Too few validation ticker-days with >=2 distinct stories for meaningful jackknife threshold')
    threshold=float(np.quantile(finite,0.80)) # locked before calibration or test
    cp=score_predictions(mp,cal,tickers,PRICE_FEATURES)
    cn=score_predictions(mn,cal,tickers,PRICE_FEATURES+NEWS_FEATURES)
    cd=fragility_for(cal,cn,mn,groups,tickers)
    cr,cuse=routed(cp,cn,cd,threshold,cal.story_count.to_numpy())
    qp=quantile(np.abs(cal.target.to_numpy()-cp))
    qn=quantile(np.abs(cal.target.to_numpy()-cn))
    qr=quantile(np.abs(cal.target.to_numpy()-cr))
    tp=score_predictions(mp,test,tickers,PRICE_FEATURES)
    tn=score_predictions(mn,test,tickers,PRICE_FEATURES+NEWS_FEATURES)
    td=fragility_for(test,tn,mn,groups,tickers)
    tr,tuse=routed(tp,tn,td,threshold,test.story_count.to_numpy())
    test_dates=set(test.date.to_list())
    agroups,attacked_keys=attack_news(groups,set(test_dates),copies=args.copies)
    attacked_df=build_features(test[prices.columns],agroups)
    # keep identical test order
    attacked_df=attacked_df.set_index(['ticker','date']).loc[pd.MultiIndex.from_frame(test[['ticker','date']])].reset_index()
    an=score_predictions(mn,attacked_df,tickers,PRICE_FEATURES+NEWS_FEATURES)
    ad=fragility_for(attacked_df,an,mn,agroups,tickers)
    ar,ause=routed(tp,an,ad,threshold,attacked_df.story_count.to_numpy())
    affected=np.array([(t,d) in set(attacked_keys) for t,d in test[['ticker','date']].itertuples(index=False,name=None)])
    y=test.target.to_numpy()
    report={'experiment_type':'candidate_empirical_pilot', 'input_provenance':'USER_MUST_VERIFY_SOURCE', 'pilot':True,'seed':SEED,'cutoff_rule':'Verified precise timezone-aware publication < NY close; date-only/midnight/ambiguous eligible first subsequent trading day after original source calendar date',
      'ticker_selection':tickers,'splits':dict(zip(['validation_start','calibration_start','test_start'],date_bounds)),
      'audit':audit(news,prices),'matched_news_rows':len(tagged),'training_rows':len(train),'validation_rows':len(val),
      'calibration_rows':len(cal),'test_rows':len(test),'attack_examples':int(affected.sum()),
      'jackknife_validation_samples':int(len(finite)),'threshold':threshold,
      'price_only':metrics(y,tp,qp), 'clean_news':metrics(y,tn,qn), 'attacked_news':metrics(y,an,qn),
      'clean_routed':metrics(y,tr,qr),'attacked_routed':metrics(y,ar,qr),
      'clean_fallback_rate':round(float((~tuse).mean()),4),'attacked_fallback_rate':round(float((~ause).mean()),4),
      'attacked_subset_price_only':metrics(y[affected],tp[affected],qp) if affected.any() else None,
      'attacked_subset_news':metrics(y[affected],an[affected],qn) if affected.any() else None,
      'clean_subset_news':metrics(y[affected],tn[affected],qn) if affected.any() else None,
      'attacked_subset_routed':metrics(y[affected],ar[affected],qr) if affected.any() else None,
      'clean_subset_routed':metrics(y[affected],tr[affected],qr) if affected.any() else None,
      'warnings':['Lead-label boundary purging applied between train/val/cal/test.',
                  'Time-series and cross-ticker dependence invalidate iid conformal finite-sample guarantees.',
                  'The news source may already encode post-cutoff market information; manually audit original timestamps.',
                  'Attacks are on historical input features only, NOT a simulation of prices responding to real misinformation.',
                  'No baseline beats are assumed; report all results including null or negative findings.']}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,default=str))
    pd.DataFrame({'ticker':test.ticker,'date':test.date,'true_return':y,'news_clean':tn,'news_attack':an,'routed_clean':tr,'routed_attack':ar,'fragility_clean':td,'fragility_attack':ad,'attacked':affected}).to_csv(out/'test_predictions.csv',index=False)
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--news',required=True);ap.add_argument('--prices',required=True)
    ap.add_argument('--tickers',default='AAPL,MSFT,NVDA,AMZN')
    ap.add_argument('--start');ap.add_argument('--end');ap.add_argument('--copies',type=int,default=10)
    ap.add_argument('--out',default='pilot_results')
    run(ap.parse_args())
