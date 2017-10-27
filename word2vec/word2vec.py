import tensorflow as tf
import numpy as np

corpus_raw = 'He is the king . The king is royal . she is the roayl queen'
corpus_raw = corpus_raw.lower()

words = []
for word in corpus_raw.split():
    if word != '.':
        words.append(word)
words = set(words)

word2int = {}
int2word = {}
vocab_size = len(words)

for i, word in enumerate(words):
    word2int[word] = i
    int2word[i] = word
    vocab_size = len(words)

raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

WINDOW_SIZE = 2

ngram_data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nebor_word in sentence[max(word_index-WINDOW_SIZE, 0) : min(word_index+WINDOW_SIZE, len(sentence))+1]:
            if nebor_word != word:
                ngram_data.append([word, nebor_word])

# print(train_data)

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = []
y_train = []

for wordpair in ngram_data:
    x_train.append(to_one_hot(word2int[wordpair[0]], vocab_size))
    y_train.append(to_one_hot(word2int[wordpair[1]], vocab_size))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# Construct network

x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5

W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM], stddev=tf.sqrt(1.0/vocab_size)))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM], stddev=tf.sqrt(1.0/vocab_size)))
hidden_representation = tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size], stddev=tf.sqrt(1.0/vocab_size)))
b2 = tf.Variable(tf.random_normal([vocab_size], stddev=tf.sqrt(1.0/vocab_size)))
prediction = tf.add(tf.matmul(hidden_representation, W2), b2)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=prediction))
train_opt = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10000

for _ in range(n_iters):
    sess.run(train_opt, feed_dict={x: x_train, y_label: y_train})
    if _ % 500 == 0:
        print('loss is: %f' % (sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train})))

embedding = sess.run(W1+b1)
print('embedding:')
print(embedding)

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, embedding):
    min_dist = 10000
    min_index = -1
    query_vector = embedding[word_index]
    for index, vector in enumerate(embedding):
        if euclidean_dist(query_vector, vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(query_vector, vector)
            min_index = index
    return min_index
print("embedding fo the 'queen': ")
print(embedding[word2int['queen']])

print("embedding fo the 'king': ")
print(embedding[word2int['king']])

print('the closest word to king is:')
print(int2word[find_closest(word2int['king'], embedding)])

print('the closest word to queen is:')
print(int2word[find_closest(word2int['queen'], embedding)])

print('the closest word to he is:')
print(int2word[find_closest(word2int['he'], embedding)])