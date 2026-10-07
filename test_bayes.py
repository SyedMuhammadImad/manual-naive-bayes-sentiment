import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from manual_bayes import *

def test_count_prior_likelihood_and_probability_match_independent_reference():
    rng=np.random.default_rng(9);x=csr_matrix(rng.integers(0,8,(100,13)));y=np.array(['negative','neutral','positive']*33+['negative'])
    for alpha in [.1,1.,2.]:
        manual=train_naive_bayes(x,y,alpha);reference=MultinomialNB(alpha=alpha).fit(x,y)
        np.testing.assert_allclose(manual['feature_count'],reference.feature_count_)
        np.testing.assert_allclose(manual['log_prior'],reference.class_log_prior_)
        np.testing.assert_allclose(manual['log_likelihood'],reference.feature_log_prob_)
        np.testing.assert_allclose(predict_log_proba(x,manual),reference.predict_log_proba(x),atol=1e-12)
        np.testing.assert_array_equal(predict(x,manual),reference.predict(x))

def test_train_only_vocabulary_cleanup_counts_and_unseen_empty_text():
    texts=['@someone Love love the flight https://example.org 123!','Worst flight.','Ordinary flight']
    vocab=fit_vocabulary(texts,stop_words=['the']);assert 'someone' not in vocab and '123' not in vocab
    counts=bag_of_words(texts,vocab,stop_words=['the'])
    assert counts[0,vocab['love']]==2
    model=train_naive_bayes(counts,np.array(['positive','negative','neutral']))
    before=dict(vocab)
    probs=np.exp(predict_log_proba(bag_of_words(['brandnewtoken',''],vocab),model))
    np.testing.assert_allclose(probs.sum(axis=1),1)
    assert vocab==before
    label,confidence=custom_inference('love flight',model,vocab)
    assert label=='positive' and 0<=confidence<=1

def test_long_document_is_finite_without_multiplicative_underflow():
    vocab={'good':0,'bad':1}
    model=train_naive_bayes(csr_matrix([[10,1],[1,10]]),np.array(['positive','negative']))
    logprob=predict_log_proba(csr_matrix([[10000,1],[1,10000]]),model)
    assert np.isfinite(logprob).all()
    np.testing.assert_array_equal(predict(csr_matrix([[10000,1],[1,10000]]),model),['positive','negative'])

def test_invalid_counts_shapes_labels_and_smoothing_fail():
    for alpha in [0,-1,float('nan')]:
        with pytest.raises(ValueError):train_naive_bayes([[1,0],[0,1]],np.array([0,1]),alpha)
    for x,y in [([[-1,1],[0,1]],[0,1]),([[np.inf,1],[0,1]],[0,1]),([[1,0],[0,1]],[0]),([[1,0],[0,1]],[0,0])]:
        with pytest.raises(ValueError):train_naive_bayes(x,np.array(y))
    with pytest.raises(ValueError):fit_vocabulary(['!!!'])
