# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""

import queue
from random import shuffle
import codecs
import json
import glob
import numpy as np
import tensorflow as tf
import data
from nltk.tokenize import sent_tokenize

FLAGS = tf.app.flags.FLAGS
class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, review, label, vocab, hps):

    self.hps = hps



    review_words = []
    review_sentences = sent_tokenize(review)[0]

    review_words= review_sentences.split()
    if len(review_words) > hps.max_dec_steps: #:
        review_words = review_words[:hps.max_dec_steps]


    self.enc_input = [vocab.word2id(w) for w in
                      review_words]  # list of word ids; OOVs are represented by the id for UNK token

    self.enc_len = len(review_words)  # store the length after truncation but before padding
    #self.enc_sen_len = [len(sentence_words) for sentence_words in review_words]
    self.label = int(label)
    self.original_reivew = review_sentences

  def pad_encoder_input(self, max_sen_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""


    while len(self.enc_input) < max_sen_len:
            self.enc_input.append(pad_id)




class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder



  def init_encoder_seq(self, example_list, hps):

    #print (example_list)

    #max_enc_seq_len = max(ex.enc_len for ex in example_list)
    for ex in example_list:
      ex.pad_encoder_input(hps.max_dec_steps, self.pad_id)

    self.enc_batch = np.zeros((hps.batch_size,hps.max_dec_steps), dtype=np.int32)
    #self.enc_word_padding_mask = np.zeros((hps.batch_size,hps.max_enc_sen_num, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    #self.enc_sen_lens=np.zeros((hps.batch_size*hps.max_enc_sen_num), dtype=np.int32)
    self.enc_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
    self.labels = np.zeros((hps.batch_size), dtype=np.int32)
    self.original_reviews = [ex.original_reivew for ex in example_list]

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.labels[i] = ex.label
      self.enc_batch[i,:] = np.array(ex.enc_input)[:]
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
          self.enc_padding_mask[i][j] = 1



      '''for j in range(ex.enc_len):
            self.enc_padding_mask[i][j] = 1
            for k in range(max_enc_seq_len):
              if self.enc_batch[i][j][k] != self.pad_id:
                self.enc_word_padding_mask[i][j][k] =1'''






class ClaBatcher(object):
    def __init__(self, hps, vocab):
        self._vocab = vocab
        self._hps = hps


        self.train_queue = self.fill_example_queue(
            "train-original/*")
        self.test_queue = self.fill_example_queue(
            "test-original/*")


        self.train_batch = self.create_batches(mode="train", shuffleis=True)
        self.test_batch = self.create_batches(mode="test", shuffleis=False)

    def create_batches(self, mode="train", shuffleis=True):

        all_batch = []

        if mode == "train":
            num_batches = int(len(self.train_queue) / self._hps.batch_size)
            if shuffleis:
                shuffle(self.train_queue)
        elif mode == 'test':
            num_batches = int(len(self.test_queue) / self._hps.batch_size)

        for i in range(0, num_batches):
            batch = []
            if mode == 'train':
                batch += (self.train_queue[i*self._hps.batch_size:i*self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test':
                batch += (self.test_queue[i*self._hps.batch_size:i*self._hps.batch_size + self._hps.batch_size])

            all_batch.append(Batch(batch, self._hps, self._vocab))
        return all_batch

    def get_batches(self, mode="train"):

        if mode == "train":
            shuffle(self.train_batch)
            return self.train_batch

        elif mode == 'test':
            return self.test_batch


    def fill_example_queue(self, data_path, filenumber=None):

        new_queue =[]

        filelist = glob.glob(data_path)  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
        filelist = sorted(filelist)
        if filenumber !=None:
            filelist = filelist[:filenumber]
        for f in filelist:
            reader = codecs.open(f, 'r', 'utf-8')
            while True:
                string_ = reader.readline()
                if not string_: break
                dict_example = json.loads(string_)
                review = dict_example["review"]
                if review.strip() =="":
                    continue
                score = dict_example["score"]
                if int(score)<3:
                    score = 0
                elif int(score)==3:
                    continue

                else:
                    score = 1
                example = Example(review, score, self._vocab, self._hps)
                new_queue.append(example)
        return new_queue



