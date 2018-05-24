import os
import time
import numpy as np 
import tensorflow as tf

# learning reference: https://github.com/hzy46/Char-RNN-TensorFlow
# python -v 3.6.3
# tensorflow-gpu -v 1.4.0 

# random generate one char
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p /= np.sum(p)
    chr = np.random.choice(vocab_size, 1, p=p)[0]
    return chr

class CharRNN:
    
    def __init__(self, num_classes, num_seqs=64, num_steps=50, 
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, 
                 use_embedding=False, embedding_size=128):
        if sampling:    # running generate text, give a start word to model 
            self.num_seqs, self.num_steps = 1, 1
        else:
            self.num_seqs, self.num_steps = num_seqs, num_steps
        self.num_classes = num_classes  # softmax output size
        self.lstm_size = lstm_size      # hidden size
        self.num_layers = num_layers    # multi layers count
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding      # embedding layers needed for Chinese, not for EN
        self.embedding_size = embedding_size    # map output size to embedding size
        
        # Clears the default graph stack and resets the global default graph
        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope(name='ipt_layer'):
            self.inputs = tf.placeholder(tf.int32, name='inputs',
                                         shape=(self.num_seqs, self.num_steps))
            self.target = tf.placeholder(tf.int32, name='targets',
                                         shape=(self.num_seqs, self.num_steps))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
            if self.use_embedding:  # for Chinese
                with tf.device("/gpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
            else:                   # for English
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)

    def build_lstm(self):
        # generate lstm cell and dropout
        def gen_cell(lstm_shape, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_shape)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop
        with tf.name_scope('lstm'):
            mcell = tf.nn.rnn_cell.MultiRNNCell([
                    gen_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)
            ])
            self.initial_stats = mcell.zero_state(self.num_seqs, tf.float32)
            self.lstm_outputs, self.final_stats = tf.nn.dynamic_rnn(mcell, self.lstm_inputs, 
                                                               initial_state=self.initial_stats)
            # get probibality from lstm outputs
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes],stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))
            
            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.prob_predict = tf.nn.softmax(self.logits, name='prob_predict')
        
    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.reshape(tf.one_hot(self.target, self.num_classes), self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_one_hot)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    def train(self, batch_generate, max_step, save_path, save_n, print_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_stats = sess.run(self.initial_stats)
            for x, y in batch_generate:
                step += 1
                start = time.time()
                feed = {
                    self.inputs: x,
                    self.target: y,
                    self.initial_stats: new_stats,
                    self.keep_prob: self.train_keep_prob
                }
                batch_loss, new_stats, _ = sess.run([self.loss,self.final_stats, 
                                                     self.optimizer], feed_dict=feed)
                # batch_loss, new_stats = sess.run([self.loss,self.final_stats], feed_dict=feed) 
                end = time.time()
                if step % print_n == 0:
                    print('step: {}/{}... '.format(step, max_step),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if step / save_n == 0:
                    print('save model in step {}'.format(step))
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step > max_step:
                    print('{} steps finished, training over'.format(max_step))
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
    
    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) # lost in last version !!
        new_stats = sess.run(self.initial_stats)
        for c in prime:
            x = np.zeros((1,1))
            x[0,0] = c
            feed = {
                self.inputs:x,
                self.initial_stats:new_stats,
                self.keep_prob:1.
            }
            pred, new_stats = sess.run([self.prob_predict, self.final_stats],feed_dict=feed)
        nc = pick_top_n(pred, vocab_size) # random generate char in vocab_size by pred ditribution
        samples.append(nc)
        for _ in range(n_samples):  # generate n_samples chars in this group
            x = np.zeros((1,1))
            x[0,0] = c
            feed = {
                self.inputs:x,
                self.initial_stats:new_stats,
                self.keep_prob:1.
            }
            pred, new_stats = sess.run([self.prob_predict, self.final_stats],feed_dict=feed)
            nc = pick_top_n(pred, vocab_size) 
            samples.append(nc)
        print('generate sample finished')
        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

# test = CharRNN(1)