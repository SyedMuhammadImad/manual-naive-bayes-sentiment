"""project sentiment comparison on an explicit local CSV/XLSX dataset."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from manual_bayes import *

def score(y,pred,labels):
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}

def save_model(folder,model,vocabulary,stop_words):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    obj={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in model.items()}
    (folder/'model.json').write_text(json.dumps({'model':obj,'vocabulary':vocabulary,'stop_words':list(stop_words)}),encoding='utf-8')

def load_model(folder):
    obj=json.loads((Path(folder)/'model.json').read_text(encoding='utf-8'))
    model=obj['model']
    for key in ['classes','class_count','feature_count','log_prior','log_likelihood']:model[key]=np.asarray(model[key])
    return model,obj['vocabulary'],obj['stop_words']

def train(path,output,text_column='Tweet',label_column='Sentiment',stop_path=None,drop_missing=False):
    path=Path(path)
    if path.stat().st_size>100_000_000:raise ValueError('Dataset exceeds 100 MB')
    df=pd.read_excel(path) if path.suffix.lower()=='.xlsx' else pd.read_csv(path)
    if text_column not in df or label_column not in df:raise ValueError('Missing selected text/label columns')
    source_rows=len(df)
    missing=df[text_column].isna()|df[label_column].isna()
    if missing.any() and not drop_missing:raise ValueError('Missing text or label; use --drop-missing to explicitly remove these rows')
    df=df.loc[~missing].copy()
    texts=df[text_column];labels=df[label_column]
    stops=[] if stop_path is None else Path(stop_path).read_text(encoding='utf-8').split()
    normalized=texts.map(lambda t:' '.join(tokenize(t,stops)))
    clean=pd.DataFrame({'text':normalized,'label':labels.map(lambda y:str(y).casefold().strip())})
    if set(clean.label)!={'negative','neutral','positive'}:raise ValueError('Expected negative, neutral and positive labels')
    conflicting=set(clean.groupby('text')['label'].nunique().loc[lambda x:x>1].index)
    usable=clean.loc[~clean.text.isin(conflicting)&clean.text.ne('')].drop_duplicates('text')
    if usable.label.value_counts().min()<10:raise ValueError('Too few distinct samples per class')
    tx,hold,ty,hy=train_test_split(usable.text.to_numpy(),usable.label.to_numpy(),test_size=.4,stratify=usable.label,random_state=42)
    vx,ex,vy,ey=train_test_split(hold,hy,test_size=.5,stratify=hy,random_state=42)
    assert not set(tx)&set(vx) and not set(tx)&set(ex) and not set(vx)&set(ex)
    vocabulary=fit_vocabulary(tx,stops);x=bag_of_words(tx,vocabulary,stops);v=bag_of_words(vx,vocabulary,stops);e=bag_of_words(ex,vocabulary,stops)
    candidates=[]
    for alpha in [.1,.5,1.,2.]:
        model=train_naive_bayes(x,ty,alpha)
        candidates.append({'alpha':alpha,'validation':score(vy,predict(v,model),model['classes'])})
    best=max(candidates,key=lambda c:(c['validation']['macro_f1'],-c['alpha']))['alpha']
    manual=train_naive_bayes(x,ty,best)
    reference=MultinomialNB(alpha=best).fit(x,ty)
    ovr=OneVsRestClassifier(MultinomialNB(alpha=best)).fit(x,ty)
    np.testing.assert_allclose(predict_log_proba(e,manual),reference.predict_log_proba(e),atol=1e-10)
    results={'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seed':42,'source_rows':source_rows,
        'missing_text_or_label_rows_removed':int(missing.sum()),
        'usable_distinct_cleaned_texts':len(usable),'rows_removed_for_duplicate_empty_or_conflicting_text':len(df)-len(usable),
        'conflicting_cleaned_text_groups_removed':len(conflicting),'vocabulary_size':len(vocabulary),
        'split_counts':{'train':len(tx),'validation':len(vx),'test':len(ex)},'classes':manual['classes'].tolist(),
        'selected_alpha':best,'selection':'validation macro F1 only','validation_candidates':candidates,
        'models':{'manual_multinomial':score(ey,predict(e,manual),manual['classes']),
                  'sklearn_multinomial':score(ey,reference.predict(e),manual['classes']),
                  'sklearn_one_vs_rest':score(ey,ovr.predict(e),manual['classes'])},
        'manual_reference_log_probability_parity':True}
    save_model(output,manual,vocabulary,stops)
    restored,rv,rs=load_model(output)
    np.testing.assert_array_equal(predict(e,manual),predict(bag_of_words(ex,rv,rs),restored))
    results['checkpoint_reload_identical']=True
    (Path(output)/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({name:v['accuracy'] for name,v in results['models'].items()}))
    return results

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--dataset',required=True);tr.add_argument('--output-dir',required=True)
    tr.add_argument('--text-column',default='Tweet');tr.add_argument('--label-column',default='Sentiment');tr.add_argument('--stop-words');tr.add_argument('--drop-missing',action='store_true')
    pr=sub.add_parser('predict');pr.add_argument('--model-dir',required=True);pr.add_argument('--text',required=True)
    args=p.parse_args()
    try:
        if args.command=='train':train(args.dataset,args.output_dir,args.text_column,args.label_column,args.stop_words,args.drop_missing)
        else:
            m,v,s=load_model(args.model_dir);label,confidence=custom_inference(args.text,m,v,s)
            print(json.dumps({'label':label,'uncalibrated_model_probability':confidence}))
    except (ValueError,OSError,KeyError) as exc:p.exit(2,'Failed: '+str(exc)+'\n')

if __name__=='__main__':main()
