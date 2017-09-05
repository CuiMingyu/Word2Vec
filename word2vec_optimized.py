"""Multi-threaded word2vec unbatched skip-gram model.
Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.
The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import codecs


from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import svm, datasets
from pylab import *
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf8')

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path",'/storeData/', "Directory to write the model.")
flags.DEFINE_string(
    "train_data",'newwords.txt',
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data",'eval.txt', "Analogy questions. "
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train",1,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 1,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # Where to write out summaries.
    self.save_path = FLAGS.save_path
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    # Eval options.

    # The text file for eval.
    self.eval_data = FLAGS.eval_data


class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph()
    self.build_eval_graph()
    self.save_vocab()

  def read_analogies(self):
    """Reads through the analogy question file.
    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(self._options.eval_data, "rb") as analogy_f:
      for line in analogy_f:
        if line.startswith(b":"):  # Skip comments.
          continue
        words = line.strip().lower().split(b" ")
        ids = [self._word2id.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", self._options.eval_data)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    self._analogy_questions = np.array(questions, dtype=np.int32)

  def build_graph(self):
    """Build the model graph."""
    opts = self._options

    # The training data. A text file.
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                                    batch_size=opts.batch_size,
                                                    window_size=opts.window_size,
                                                    min_count=opts.min_count,
                                                    subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)

    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i

    # Declare all variables we need.
    # Input words embedding: [vocab_size, emb_dim]
    w_in = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size,
             opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
        name="w_in")

    # Global step: scalar, i.e., shape [].
    w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train_word2vec(w_in,
                                          w_out,
                                          examples,
                                          labels,
                                          lr,
                                          vocab_count=opts.vocab_counts.tolist(),
                                          num_negative_samples=opts.num_samples)

    self._w_in = w_in
    self._examples = examples
    self._labels = labels
    self._lr = lr
    self._train = train
    self.global_step = global_step
    self._epoch = current_epoch
    self._words = total_words_processed

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.save_path, "newvoc.txt"), "w") as f:
      for i in xrange(opts.vocab_size):
        vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
        f.write("%s %d\n" % (vocab_word,
                             opts.vocab_counts[i]))
    with open(os.path.join(opts.save_path,"dict.txt"),"w") as file:
        for i in xrange(opts.vocab_size):
            vocab_word=tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
            file.write("%s\n" % (vocab_word))

  def build_eval_graph(self):
    """Build the evaluation graph."""
    # Eval graph
    opts = self._options

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._w_in, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, opts.vocab_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_pred_idx = pred_idx
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

    # Properly initialize all variables.
    tf.global_variables_initializer().run()

    self.saver = tf.train.Saver()

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      _, epoch = self._session.run([self._train, self._epoch])
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time = initial_words, time.time()
    while True:
      time.sleep(5)  # Reports our progress once a while.
      (epoch, step, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step,
                                                                    lr, rate),
            end="")
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

  def eval(self):
    """Evaluate analogy questions and reports accuracy."""

    # How many questions we get right at precision@1.
    correct = 0

    try:
      total = self._analogy_questions.shape[0]
    except AttributeError as e:
      raise AttributeError("Need to read analogy questions.")

    start = 0
    while start < total:
      limit = start + 2500
      sub = self._analogy_questions[start:limit, :]
      idx = self._predict(sub)
      start = limit
      for question in xrange(sub.shape[0]):
        for j in xrange(4):
          if idx[question, j] == sub[question, 3]:
            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
            correct += 1
            break
          elif idx[question, j] in sub[question, :3]:
            # We need to skip words already in the question.
            continue
          else:
            # The correct label is not the precision@1
            break
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              correct * 100.0 / total))

  def analogy(self, w0, w1, w2):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
    idx = self._predict(wid)
    for c in [self._id2word[i] for i in idx[0, :]]:
      if c not in [w0, w1, w2]:
        print(c)
        break
    print("unknown")

  def nearby(self, words, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([self._word2id.get(x, 0) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (self._id2word[neighbor], distance))

  def test(self):
      print(type(self._w_in))
      print(self._w_in)
      print(type(self._w_in.eval()))
      print(len(self._w_in.eval()))
      print(len(self._id2word))
  def findw2v(self):
      director=codecs.open('/home/cui/Public/Word2vecModel/word/Director.txt','r','utf-8')
      actor=codecs.open('/home/cui/Public/Word2vecModel/word/Actor.txt','r','utf-8')
      editor=codecs.open('/home/cui/Public/Word2vecModel/word/Editor.txt','r','utf-8')
      type1=codecs.open('/home/cui/Public/Word2vecModel/word/Type.txt','r','utf-8')
      words=open('/storeData/dict.txt','r')
      w2cdir=[[] for i in range(1038)]
      w2cact=[[] for i in range(7784)]
      w2cedi=[[] for i in range(1545)]
      w2ctyp=[[] for i in range(1884)]
      dict={}
      for x in range(29862):
          dict[words.readline().decode('utf-8')]=x
      temp=director.readline()
      for x in range(1038):
          temp = director.readline()
          #print(temp)
          #print(dict[temp])
          w2cdir[x].append(self._w_in.eval()[dict[temp]])
      #editor
      for x in range(1545):
          temp=editor.readline()
          temp1 = ''.join(temp).strip('\n')
          w2cedi[x].append(self._w_in.eval()[dict[temp1]])
      #director

      #type
      for x in range(1884):
          temp=type1.readline()
          temp1 = ''.join(temp).strip('\n')
          w2ctyp[x].append(self._w_in.eval()[dict[temp1]])
      #actor
      for x in range(7784):
          temp=actor.readline()
          temp1 = ''.join(temp).strip('\n')
          w2cact[x].append(self._w_in.eval()[dict[temp1]])

      #print(temp)
      #print(self._w_in.eval()[1])
      #tp=np.array(self._id2word).item(x)

              #num=x



def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)
def kmean(self):
    """Use for kmeans."""
    result=KMeans(n_clusters=2).fit_predict(self._w_in.eval())

    sit=self._w_in.eval()
    'y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)'
    """plt.scatter(sit[:, 0], sit[:, 1], c=result)"""
    """plt.show()"""
    pca=PCA(n_components=0.9)
    pca.fit(sit)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    sit_new=pca.transform(sit)
    plt.scatter(sit_new[:,0],sit_new[:,1],c=result)
    plt.show()

def kmean1(self):
    """Use for kmeans."""
    result=KMeans(n_clusters=10).fit_predict(self._w_in.eval())
    print(result)
    newArray=[[] for i in range(10)]
    num=0
    for x in result:
      if len(newArray[num])<4000:
        newArray[num].append(x)
      elif num==9 and len(newArray[num])==4000:
        break
      else:
        num+=1
        newArray[num].append(x)


    for x in range(10):
      print(len(newArray[x]))
    labels=[0,1,2,3,4,5,6,7,8,9]
    sit=self._w_in.eval()
    X=newArray
    clf=svm.SVC(kernel='linear',C=1.0)
    clf.fit(X,labels)
    print(clf.fit(X,labels))
    pca = PCA(n_components=2)
    b = pca.fit(X, labels)


    print("666666666666")
    print(newArray[0:10])
    print(type(newArray))
    print(type(b))
    try:
      print(np.array(b))
    except Exception as e:
      print(e)
    # print(b[0:10])
    print("666666666666")
    plt.scatter((b[:, 0]), (b[:, 1]), labels='class0')
    # plt.show()
    """w=clf.coef_[0]
    a=-w[0]/w[1]
    xx=np.linspace(-5,5)
    yy=a*xx-clf.intercept_[0]/w[1]

    ho=plt.plot(xx,yy,'k-',label='no weights')
    pca = PCA(n_components=2)
    pca.fit(newArray[x] for x in range(10))
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    sit_new = pca.transform()
    pca=PCA(n_components=2)
    pca.fit(X,labels)
    plt.scatter(newArray[:, 0], newArray[:, 1], labels='class0')
    plt.show()
    
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.subplot(2, 2, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('linear kernel')
    plt.show()"""
def findK(self):
    n,distance=8,[]
    for i in range(1,n):
      minDis=-1
      for j in range(10):
        kmeans=KMeans(n_clusters=i).fit(self._w_in.eval())
        centers=np.array([kmeans.cluster_centers_[k] for k in kmeans.labels_])
        dis=((self._w_in.eval()-centers) ** 2).sum()
        minDis=dis if minDis<0 else min(minDis,dis)
      distance.append(minDis)
    plot(range(1,n),distance)
    grid()
    show()

def svm(self):
    X=self._w_in.eval()
    labels=[0,1,2,3,4,5,6]
    clf=svm.SVC(decision_function_shape='ovo')
    clf.fit(X,labels)
    dec=clf.decision_function([[1]])
    dec.shape[1]
    clf.decision_function_shape="ovr"
    dec=clf.decision_function([[1]])
    dec.shape[1]
    print(clf.predict([1]))





def main(_):
  """Train a word2vec model."""
  if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
    print("--train_data --eval_data and --save_path must be specified.")
    sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(opts, session)
      model.read_analogies() # Read analogy questions
    for _ in xrange(opts.epochs_to_train):
      model.train()  # Process one epoch
      #model.eval()  # Eval analogies.
    # Perform a final save.
    model.findw2v()
    model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"),
                     global_step=model.global_step)

    
    if FLAGS.interactive:
      # E.g.,
      # [0]: model.analogy(b'france', b'paris', b'russia')
      # [1]:model.nearby([b'proton', b'elephant', b'maxwell'])

      _start_shell(locals())


if __name__ == "__main__":
  tf.app.run()
