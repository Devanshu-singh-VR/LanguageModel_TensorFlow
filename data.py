import re
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('train.csv')['question']
sent = [re.sub(r"([?.!',¿()])", r" \1 ", line+' <eos>') for line in data]

tokenizer = Tokenizer(10000, filters='', lower=True, oov_token='<unk>')
tokenizer.fit_on_texts(sent)

train_tensor = tokenizer.texts_to_sequences(sent)
train_tensor = pad_sequences(train_tensor, padding='post')

train_data = tf.data.Dataset.from_tensor_slices(train_tensor).batch(64).shuffle(len(sent))

# add the padding
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

class Decoder(Model):
    def __init__(self, vocab_size, hidden_size, embedding_size, dropout):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=dropout)
        self.fc = Dense(vocab_size, activation='softmax')
        self.dropout = Dropout(dropout)

    def call(self, x, hidden, cell):
        x = self.dropout(self.embedding(x))
        # x.shape(batch, 1)
        output, hidden, cell = self.lstm(x, initial_state=[hidden, cell])
        # output.shape(batch, 1, embedding)
        # hidden.shape(batch, hidden_size)
        output = self.fc(output)
        # output.shape(batch, 1, vocab_size)
        return output, hidden, cell

class Sequence(Model):
    def __init__(self, vocab_size, hidden_size, embedding_size, dropout):
        super(Sequence, self).__init__()
        self.vocab_size = vocab_size
        self.decoder = Decoder(vocab_size, hidden_size, embedding_size, dropout)
        self.hidden_size = hidden_size

    def call(self, data):
        batch_size, seq_len = data.shape

        hidden = tf.zeros((batch_size, self.hidden_size))
        cell = tf.zeros((batch_size, self.hidden_size))
        x = data[:, 0:1]

        outputs = tf.zeros((batch_size, 1, self.vocab_size))

        for t in range(1, seq_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs = tf.concat((outputs, output), axis=1)
            x = tf.expand_dims(data[:, t], axis=1)

        return outputs

    def language(self, sent, max_len=35):
        batch_size, seq_len = sent.shape

        hidden = tf.zeros((batch_size, self.hidden_size))
        cell = tf.zeros((batch_size, self.hidden_size))
        x = sent

        answer = []
        answer.append(tokenizer.index_word[int(sent[:, 0])])

        for t in range(max_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            word = tf.argmax(tf.squeeze(output, axis=1), axis=1)

            if tokenizer.index_word[int(word)] == '<eos>':
                break
            answer.append(tokenizer.index_word[int(word)])

            x = tf.expand_dims(word, axis=0)

        return answer


# HYPERPARAMETERS
hidden_size = 120
vocab_size = len(tokenizer.word_index)
dropout = 0.5
epochs = 130
embedding_size = 100
learning_rate = 0.003

model = Sequence(vocab_size, hidden_size, embedding_size, dropout)
loss_f = SparseCategoricalCrossentropy(reduction='none')
optimizer = Adam(learning_rate=learning_rate)

model.load_weights('language/')


def LOSS(real, prediction):
    loss = loss_f(real, prediction)
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=loss.dtype)
    return tf.reduce_mean((loss*mask), axis=0)


for epoch in range(epochs):
    print(f'Epochs {epoch}/{epochs}')
    losses = []
    for batch, text in enumerate(train_data):
        with tf.GradientTape() as tape:
            score = model(text)
            # score.shape(batch, seq_len, vocab_size)
            # text.shape(batch, seq_len)

            score = tf.reshape(score[1:], (-1, score.shape[2]))
            text = tf.reshape(text[1:], (-1))
            # score.shape(batch*seq_len, vocab_size)
            # test.shape(batch*seq_len)
            loss = LOSS(text, score)
            losses.append(loss)

        variables = model.trainable_variables
        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradient, variables))

    print(f'Loss = ', sum(losses) / len(losses))
    print(model.language(tf.constant([[tokenizer.word_index['this']]])))

model.save_weights('language/')