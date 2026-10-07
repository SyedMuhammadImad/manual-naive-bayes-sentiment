import json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
import pytest
from experiment import train,load_model
from manual_bayes import custom_inference

def fixture(path):
    rows=[]
    for label,words in [('positive','good excellent'),('negative','bad terrible'),('neutral','ordinary standard')]:
        for i in range(40):
            token='token'+chr(97+i//26)+chr(97+i%26)
            rows.append({'Tweet':words+' '+token,'Sentiment':label})
    rows += [rows[0].copy(),{'Tweet':None,'Sentiment':'neutral'},
             {'Tweet':'ambiguous text','Sentiment':'positive'},{'Tweet':'ambiguous text','Sentiment':'negative'}]
    pd.DataFrame(rows).to_csv(path,index=False)

def test_full_training_duplicate_conflict_controls_reload_and_prediction_cli(tmp_path):
    path=tmp_path/'tweets.csv';fixture(path);out=tmp_path/'results'
    result=train(path,out,drop_missing=True)
    assert result['source_rows']==124 and result['usable_distinct_cleaned_texts']==120
    assert result['missing_text_or_label_rows_removed']==1 and result['conflicting_cleaned_text_groups_removed']==1
    assert result['checkpoint_reload_identical'] and result['manual_reference_log_probability_parity']
    assert result['split_counts']=={'train':72,'validation':24,'test':24}
    assert all(sum(map(sum,m['confusion_matrix']))==24 for m in result['models'].values())
    model,vocab,stops=load_model(out)
    label,confidence=custom_inference('good excellent',model,vocab,stops)
    assert label=='positive' and 0<=confidence<=1
    p=subprocess.run([sys.executable,'experiment.py','predict','--model-dir',str(out),'--text','good excellent'],
        cwd=Path(__file__).parent,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['label']=='positive'

def test_missing_rows_and_unexpected_labels_require_explicit_handling(tmp_path):
    path=tmp_path/'tweets.csv';fixture(path)
    with pytest.raises(ValueError,match='Missing text or label'):train(path,tmp_path/'out')
    df=pd.read_csv(path);df.loc[0,'Sentiment']='unknown';df.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Expected negative'):train(path,tmp_path/'out',drop_missing=True)
