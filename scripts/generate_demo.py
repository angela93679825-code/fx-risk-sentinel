"""Independent synthetic prices. No original market observations are copied."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fxrisk.data import PAIRS
from fxrisk.features import technical_indicators
root=Path(__file__).resolve().parents[1]/'data/demo'
(root/'prices').mkdir(parents=True,exist_ok=True);(root/'features').mkdir(exist_ok=True)
rng=np.random.default_rng(2025)
dates=pd.bdate_range('2020-01-01','2025-03-10')
starts={'USDJPY':110.,'EURUSD':1.1,'EURJPY':120.,'USDCNH':7.,'JPYCNY':.06}
for i,pair in enumerate(PAIRS):
 vol=.003 + .002*(1+np.sin(np.arange(len(dates))/70+i))/2
 ret=rng.normal(0,vol)
 close=starts[pair]*np.exp(np.cumsum(ret))
 opening=np.r_[close[0],close[:-1]]
 span=rng.uniform(.0005,.003,len(close))
 frame=pd.DataFrame({'Date':dates,'Open':opening,'High':np.maximum(opening,close)*(1+span),
                     'Low':np.minimum(opening,close)*(1-span),'Close':close})
 frame.to_csv(root/'prices'/f'{pair}.csv',index=False)
 technical_indicators(frame).set_index('Date').to_csv(root/'features'/f'{pair}.csv')
(root/'metadata.json').write_text(json.dumps({'dataset_kind':'synthetic_demo','seed':2025,
                                           'generator':'scripts/generate_demo.py','contains_original_market_data':False},indent=2))
