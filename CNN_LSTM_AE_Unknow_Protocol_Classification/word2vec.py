import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# prepare data
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
# word_list : ['fish', 'animal', 'milk', 'hate', 'eyes', 'i', 'music', 'apple', 'movie', 'dog', 'like', 'cat', 'book']
word_dict = {word: index for index, word in enumerate(word_list)}
# word_dict : {'fish': 0, 'animal': 1, 'milk': 2, 'hate': 3, 'eyes': 4, 'i': 5, 'music': 6, 'apple': 7, 'movie': 8, 'dog': 9, 'like': 10, 'cat': 11, 'book': 12}
print(word_dict)

batch_size = 20
embedding_size = 2
voc_size = len(word_list)  # voc_size = 13
print(word_sequence[1])

print(word_dict[word_sequence[1]])

skip_grams = []
for i in range(1, len(word_sequence) - 1): #len(word_sequence) = 42
    target = word_dict[word_sequence[i]] # 取出本单词对应的编码
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    for w in context:
        skip_grams.append([target, w])
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)
    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]]) # target one-hot encoding
        random_labels.append(data[i][1])
    return random_inputs, random_labels
input_batch, target_batch = random_batch(skip_grams, batch_size)
input_batch = np.array(input_batch, dtype=np.float32)
target_batch = np.array(target_batch, dtype=np.float32)

# build model
class Word2Vec(tf.keras.Model):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = tf.Variable(-2 * np.random.rand(voc_size, embedding_size) + 1, dtype=np.float32)
        self.WT = tf.Variable(-2 * np.random.rand(embedding_size, voc_size) + 1, dtype=np.float32)
    def call(self, X):
        hidden_layer = tf.matmul(X, self.W)
        output_layer = tf.matmul(hidden_layer, self.WT)
        return  output_layer

model = Word2Vec()
optimizer = tf.optimizers.Adam()
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',
              metrics=['acc'])
output = model(input_batch)

history = model.fit(input_batch,target_batch,epochs=2000)

for i, label in enumerate(word_list):
    W, WT = model.variables
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(6, 2), textcoords='offset points', ha='right', va='bottom')
path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/saved_model/'
plt.savefig(path + 'word2vec.png')




