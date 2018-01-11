from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print("\nloading libraries...")


import os, sys, time

import tensorflow as tf
import numpy as np
import gensim
from datetime import datetime, date

import csv
import random
import math
import re
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


print("loading model...")


tfidf = gensim.models.tfidfmodel.TfidfModel.load('model/tfidf_model_180109.tfidf')
dictionary = gensim.corpora.Dictionary.load("model/tick180103.dict")
checkpoint_file = "model/adam_h1s2000_h2s1600_lr0.01_dr0.75_ds1000_bs80.ckpt"
pik_file = "model/pickle.dat"



with open(pik_file, "rb") as f:
    pickle_data = pickle.load(f)


all_ciso_to_g_T = pickle_data["all_ciso_to_g"]
c_mapping = pickle_data["c_mapping"]
i_mapping = pickle_data["i_mapping"]
s_mapping = pickle_data["s_mapping"]
o_mapping = pickle_data["o_mapping"]
g_mapping = pickle_data["g_mapping"]




mapping_to_row_name = {"c_mapping": "category", "i_mapping": "item", "s_mapping": "subitem", "o_mapping": "originator group", "g_mapping": "assigned group"}
n_words = len(dictionary.token2id)
unknowns = []


def embed_bow(bow, mode="tfidf"):
    v1h = np.zeros(n_words)
    if mode == "1hot" or mode== "nhot": 
        for word_idx in bow:
            if mode == "1hot": v1h[word_idx[0]] = 1
            if mode == "nhot": v1h[word_idx[0]] += word_idx[1]
    if mode == "tfidf" or mode == "tfidfinv": 
        words_tfidf = tfidf[bow]
        for word_tfidf in words_tfidf:
            if mode == "tfidf": v1h[word_tfidf[0]] = word_tfidf[1]
            if mode == "tfidfinv": v1h[word_tfidf[0]] = 1/word_tfidf[1]
    return v1h


def v1h(value, size):
    onehot = np.zeros(size)
    onehot[value] = 1
    return onehot


def mapping(value, mapp_str):
    global unknowns
    mapp = eval(mapp_str)
    m_idx = 0
    if value in mapp:
        m_idx = mapp.index(value)
    else:
        pass
        #unknowns[mapping_to_row_name[mapp_str]].add(value)
        #print("unknown {}: {}".format(mapping_to_row_name[mapp_str], value))
    return m_idx


def mapping1h(value, mapp_str):
    global unknowns
    mapp = eval(mapp_str)
    onehot = np.zeros(len(mapp))
    if value in mapp:
        m_idx = mapp.index(value)
        onehot[m_idx] = 1
    else:
        pass
        #unknowns[mapping_to_row_name[mapp_str]].add(value)
        #print("unknown {}: {}".format(mapping_to_row_name[mapp_str], value))
    return onehot



max_top_k = 3

INPUT_SIZE = 3340
NUM_CLASSES = 895


optim = re.match(r".*model/(.*)_h1.*$", checkpoint_file).group(1)
h1 = int(re.match(r".*_h1s(.*)_h2.*$", checkpoint_file).group(1))
h2 = int(re.match(r".*_h2s(.*)_lr.*$", checkpoint_file).group(1))


class OPTS:
    
    optimizer = optim
    
    batch_size = 30
    max_steps = 2000

    summary_steps = 100
    log_steps = 99000
    
    learning_rate = 0.005
    decay_rate = 0.9
    decay_steps = 1000
    
    hidden1 = h1
    hidden2 = h2
    
    log_to_file = False
    log_weights_histograms = False
    base_log_dir = "ntblog/180103"
    prefix = ""



g = tf.Graph()


with g.as_default():

    
    inputs_placeholder = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE), name="input")
    labels_placeholder = tf.placeholder(tf.int32, shape=(None))


    with tf.name_scope('hidden1'):
        weights1 = tf.Variable(
                tf.truncated_normal([INPUT_SIZE, OPTS.hidden1],
                stddev=1.0 / math.sqrt(float(INPUT_SIZE))))
        biases1 = tf.Variable(tf.zeros([OPTS.hidden1]))
        hidden1 = tf.nn.relu(tf.matmul(inputs_placeholder, weights1) + biases1)

    with tf.name_scope('hidden2'):
        weights2 = tf.Variable(
                tf.truncated_normal([OPTS.hidden1, OPTS.hidden2],
                stddev=1.0 / math.sqrt(float(OPTS.hidden1))))
        biases2 = tf.Variable(tf.zeros([OPTS.hidden2]))
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

    with tf.name_scope('output'):
        weights3 = tf.Variable(
                tf.truncated_normal([OPTS.hidden2, NUM_CLASSES],
                stddev=1.0 / math.sqrt(float(OPTS.hidden2))))
        biases3 = tf.Variable(tf.zeros([NUM_CLASSES]))
        logits = tf.matmul(hidden2, weights3) + biases3


        
    prediction = tf.argmax(logits, 1)

    labels = tf.to_int64(labels_placeholder)
    evals_correct = []
    
    for i in range(1, max_top_k+1):
        correct = tf.nn.in_top_k(logits, labels_placeholder, i)
        evals_correct.append(tf.reduce_sum(tf.cast(correct, tf.int32)))

    accuracy = evals_correct[0] / tf.shape(logits)[0]
    tf.summary.scalar('accuracy', accuracy)
    
    
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits, name='xentropy')

    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.summary.scalar('loss', loss)

    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(OPTS.learning_rate, global_step, OPTS.decay_steps, OPTS.decay_rate, staircase=True)

    if OPTS.optimizer == "gd": optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    if OPTS.optimizer == "adam": optimizer = tf.train.AdamOptimizer(learning_rate)
    if OPTS.optimizer == "adagrad": optimizer = tf.train.AdagradOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=global_step)


    
    if OPTS.log_weights_histograms:
        for i in range(10):
            w1in = weights1[i,:]
            w1ni = weights1[i,:]
            tf.summary.histogram('w1_{}n'.format(i), w1in)
            tf.summary.histogram('w1_n{}'.format(i), w1ni)

            
            
    summary = tf.summary.merge_all()

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()


print("\nneural network ready")



if __name__ == "__main__":
    
    import sys
    args = sys.argv[1:]

    c,i,s,o,n,npreds = args
    test_inputs = [np.concatenate((mapping1h(c, "c_mapping"), mapping1h(i, "i_mapping"), mapping1h(s, "s_mapping"), mapping1h(o, "o_mapping"), embed_bow(dictionary.doc2bow(n.split()))))]
    n_preds = int(npreds)


    with tf.Session(graph=g) as sess:
        
        saver.restore(sess, checkpoint_file)
        
        pred_logits = logits.eval(feed_dict={inputs_placeholder: test_inputs, labels_placeholder: [0]}, session=sess)[0]
        s_logits = sorted(pred_logits, reverse=True)
        preds = [g_mapping[np.where(pred_logits==x)[0][0]] for x in s_logits[:n_preds]]

        print("")
        print("Category:", c)
        print("Item:", i)
        print("Subitem:", s)
        print("Originator:", o)
        print("Narrative:", n)

        print("\nPredicted groups:")
        for idx, pred in enumerate(preds): print("{}.- {}".format(idx+1, pred))
        print("")



def predict_group(c,i,s,o,n, n_preds=3):
    
    test_inputs = [np.concatenate((mapping1h(c, "c_mapping"), mapping1h(i, "i_mapping"), mapping1h(s, "s_mapping"), mapping1h(o, "o_mapping"), embed_bow(dictionary.doc2bow(n.split()))))]


    with tf.Session(graph=g) as sess:
        
        saver.restore(sess, checkpoint_file)
        
        pred_logits = logits.eval(feed_dict={inputs_placeholder: test_inputs, labels_placeholder: [0]}, session=sess)[0]
        s_logits = sorted(pred_logits, reverse=True)
        preds = [g_mapping[np.where(pred_logits==x)[0][0]] for x in s_logits[:n_preds]]

        print("")
        print("Category:", c)
        print("Item:", i)
        print("Subitem:", s)
        print("Originator:", o)
        print("Narrative:", n)

        print("\nPredicted groups:")
        for idx, pred in enumerate(preds): print("{}.- {}".format(idx+1, pred))
        print("")

        return preds
