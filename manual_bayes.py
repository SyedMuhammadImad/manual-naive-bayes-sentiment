"""Procedural multinomial Naive Bayes retaining the project count/prior/likelihood steps."""
import re,math
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix,issparse
from scipy.special import logsumexp

def tokenize(text,stop_words=()):
    if not isinstance(text,str):raise ValueError('Expected text')
    if len(text)>100_000:raise ValueError('Text is too long')
    text=re.sub(r'(@\w+|https?://\S+|www\.\S+)',' ',text.casefold())
    text=re.sub(r'\d+',' ',text)
    text=re.sub(r'[^\w\s]',' ',text)
    stops=set(stop_words)
    return [w for w in text.split() if w not in stops]

def fit_vocabulary(texts,stop_words=(),max_features=20000):
    if type(max_features) is not int or max_features<1:raise ValueError('Invalid vocabulary size')
    counts=Counter(w for t in texts for w in tokenize(t,stop_words))
    words=sorted(counts,key=lambda w:(-counts[w],w))[:max_features]
    if not words:raise ValueError('Empty training vocabulary')
    return {word:i for i,word in enumerate(words)}

def bag_of_words(texts,vocabulary,stop_words=()):
    if sorted(vocabulary.values())!=list(range(len(vocabulary))) or not vocabulary:raise ValueError('Invalid vocabulary')
    data=[];indices=[];indptr=[0]
    for text in texts:
        counts=Counter(w for w in tokenize(text,stop_words) if w in vocabulary)
        for word,count in counts.items():indices.append(vocabulary[word]);data.append(count)
        indptr.append(len(data))
    return csr_matrix((np.array(data,dtype=np.float64),indices,indptr),shape=(len(indptr)-1,len(vocabulary)))

def _matrix(x,features=None):
    a=csr_matrix(x,dtype=np.float64)
    if a.ndim!=2 or not a.shape[0] or not a.shape[1] or (features is not None and a.shape[1]!=features):raise ValueError('Invalid feature matrix')
    if not np.isfinite(a.data).all() or (a.data<0).any():raise ValueError('Counts must be finite and nonnegative')
    a.sum_duplicates();return a

def train_naive_bayes(matrix,labels,alpha=1.):
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('Smoothing must be finite and positive')
    x=_matrix(matrix);y=np.asarray(labels)
    if y.dtype.kind=='O' and all(isinstance(value,str) for value in y):y=y.astype(str)
    if y.ndim!=1 or len(y)!=x.shape[0] or y.dtype.kind not in 'iuUS':raise ValueError('Invalid class labels')
    classes,encoded=np.unique(y,return_inverse=True)
    if len(classes)<2:raise ValueError('Need at least two classes')
    class_count=np.bincount(encoded,minlength=len(classes)).astype(np.float64)
    word_count=np.vstack([np.asarray(x[encoded==i].sum(axis=0)).ravel() for i in range(len(classes))])
    smoothed=word_count+alpha
    total=smoothed.sum(axis=1,keepdims=True)
    if not np.isfinite(total).all():raise ValueError('Counts overflow')
    return {'classes':classes,'class_count':class_count,'feature_count':word_count,
            'log_prior':np.log(class_count/class_count.sum()),'log_likelihood':np.log(smoothed/total),'alpha':alpha}

def predict_log_proba(matrix,model):
    x=_matrix(matrix,model['log_likelihood'].shape[1])
    score=x@model['log_likelihood'].T+model['log_prior']
    if not np.isfinite(score).all():raise ValueError('Non-finite score')
    return score-logsumexp(score,axis=1,keepdims=True)

def predict(matrix,model):return model['classes'][predict_log_proba(matrix,model).argmax(axis=1)]

def custom_inference(tweet,model,vocabulary,stop_words=()):
    probabilities=np.exp(predict_log_proba(bag_of_words([tweet],vocabulary,stop_words),model))[0]
    index=int(probabilities.argmax())
    return str(model['classes'][index]),float(probabilities[index])
