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

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS


class Sentimentor(object):


  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    if FLAGS.run_method == 'auto-encoder':
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_enc_steps],
                                         name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_sen_lens')
        self._weight = tf.placeholder(tf.int32, [hps.batch_size,hps.max_enc_steps], name = "weight")
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_enc_steps], name="encoder_mask")
        #self.score = tf.placeholder(tf.int32, name = 'score')


        #self._given_number = tf.placeholder(tf.int32, name = "given_number")
        #self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')


    #self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    #self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    #self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')
    self.reward = tf.placeholder(tf.float32, [hps.batch_size], name='reward')


  def _make_feed_dict(self, batch, just_enc=False):

    feed_dict = {}

    if FLAGS.run_method == 'auto-encoder':
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask


    #feed_dict[self._dec_batch] = batch.dec_batch
    #feed_dict[self.score] = batch.score
    feed_dict[self._weight] = batch.weight
    #feed_dict[self.reward]= batch.reward
    #feed_dict[self._target_batch] = batch.target_batch
    #feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict



  def _add_encoder(self, encoder_inputs, seq_len):

    with tf.variable_scope('encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      ((encoder_for,encoder_back), (fw,bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)

    return tf.concat([encoder_for,encoder_back],axis = -1)






  def _build_model(self):
    """Add the whole generator model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('sentiment'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        #embedding_score = tf.get_variable('embedding_score', [5, hps.hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

        #emb_dec_inputs = tf.nn.embedding_lookup(embedding, self._dec_batch) # list length max_dec_steps containing shape (batch_size, emb_size)
        #emb_dec_inputs = tf.unstack(emb_dec_inputs, axis=1)
        if FLAGS.run_method == 'auto-encoder':
            emb_enc_inputs = tf.nn.embedding_lookup(embedding,
                                                    self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
            emb_enc_inputs = emb_enc_inputs * tf.expand_dims(self._enc_padding_mask, axis = -1)
            hiddenstates = self._add_encoder(emb_enc_inputs, self._enc_lens)
            #self.return_hidden = fw_st.h
            #hiddenstates = tf.contrib.rnn.LSTMStateTuple(fw_st.h, fw_st.h)#self._reduce_states(fw_st, bw_st)
      w = tf.get_variable(
            'w', [hps.hidden_dim*2, 2], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
      v = tf.get_variable(
            'v', [2], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
      hiddenstates = tf.reshape(hiddenstates, [hps.batch_size*hps.max_enc_steps,-1])
      logits = tf.nn.xw_plus_b(hiddenstates,w,v)
      logits = tf.reshape(logits, [hps.batch_size, hps.max_enc_steps, 2])




      #self.decoder_outputs_pretrain, self._max_best_output =self.add_decoder(embedding, emb_dec_inputs, vsize, hps)
      loss = tf.contrib.seq2seq.sequence_loss(
          logits,
          self._weight,
          self._enc_padding_mask,
          average_across_timesteps=True,
          average_across_batch=False)
      self.max_output = tf.argmax(logits,axis=-1)

      reward_loss = tf.contrib.seq2seq.sequence_loss(
          logits,
          self._weight,
          self._enc_padding_mask,
          average_across_timesteps=True,
          average_across_batch=False) * self.reward



      # Update the cost
      self._cost = tf.reduce_mean(loss)
      self._reward_cost = tf.reduce_mean(reward_loss)
      self.optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)


  def _add_train_op(self):

    loss_to_minimize = self._cost
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer

    self._train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

  def _add_reward_train_op(self):

    loss_to_minimize = self._reward_cost
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)


    self._train_reward_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):

    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    with tf.device("/gpu:"+str(FLAGS.gpuid)):
      tf.logging.info('Building sentiment graph...')
      t0 = time.time()
      self._add_placeholders()
      self._build_model()
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self._add_train_op()
      self._add_reward_train_op()
      t1 = time.time()
      tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_pre_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'loss': self._cost,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)




  def run_train_step(self, sess, batch, reward):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    feed_dict[self.reward]=reward

    to_return = {
        'train_op': self._train_reward_op,
        'loss': self._reward_cost,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)



  def max_generator(self,sess, batch):
      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'generated': self.max_output,
      }
      return sess.run(to_return, feed_dict)


  def run_eval(self,sess, batch):
      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'generated': self.max_output,
      }

      result = sess.run(to_return, feed_dict)
      true_gold = batch.weight
      predicted = result['generated']
      right = 0
      all = 0
      for i in range(len(predicted)):
          length = batch.enc_lens[i]
          for j in range(length):
              if predicted[i][j] == true_gold[i][j] and true_gold[i][j] == 0:
                  right +=1
              if true_gold[i][j] ==0:
                  all+=1

      return right, all, predicted, true_gold




