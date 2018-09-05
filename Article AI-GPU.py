
# coding: utf-8

# In[12]:


import numpy as np
#from nalu import NALU


# In[13]:


data = open('cnn.txt', 'r').read()
chars = list(set(data)) #set = get unique values
VOCAB_SIZE = len(chars)


# In[14]:


print('chars:\n{}\n\nVOCAB_SIZE: {}'.format(chars, VOCAB_SIZE))


# In[15]:


idx_to_char = {i:char for i, char in enumerate(chars)}
char_to_idx = {char:i for i, char in enumerate(chars)}


# In[16]:


SEQ_LENGTH = 60 #input sequence length
N_FEATURES = VOCAB_SIZE  # For 1 hot encoding
N_SEQ = int(np.floor((len(data) - 1)/SEQ_LENGTH))

X = np.zeros((N_SEQ, SEQ_LENGTH, N_FEATURES))
y = np.zeros((N_SEQ, SEQ_LENGTH, N_FEATURES))

for i in range(N_SEQ):
    x_sequence = data[i * SEQ_LENGTH: (i+1) * SEQ_LENGTH]
    x_sequence_ix = [char_to_idx[c] for c in x_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, N_FEATURES))
    for j in range(SEQ_LENGTH):
        input_sequence[j][x_sequence_ix[j]] = 1. # One hot encoding
    X[i] = input_sequence
    
    y_sequence = data[i * SEQ_LENGTH + 1: (i + 1) * SEQ_LENGTH + 1] # shifted 1 to the right
    y_sequence_ix = [char_to_idx[c] for c in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, N_FEATURES))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1. #again, one hot encoding
    y[i] = target_sequence


# In[17]:


from  keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, CuDNNLSTM

# try CuDNNLSTM on gpu


# In[18]:


HIDDEN_DIM = 700 
LAYER_NUM = 2
NB_EPOCHS = 200
BATCH_SIZE = 128
VAL_SPLIT = 0.1

model = Sequential()
model.add(CuDNNLSTM(HIDDEN_DIM,
               input_shape=(None, VOCAB_SIZE),
               return_sequences = True))

for _ in range(LAYER_NUM - 1):
    model.add(CuDNNLSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
#model.add(Activation('NULA'))
model.add(Activation('relu'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])


# In[19]:


def generate_text(model, length):
  ix = [np.random.randint(VOCAB_SIZE)]
  y_char = [idx_to_char[ix[-1]]]
  X = np.zeros((1, length, VOCAB_SIZE))
  for i in range(length):
    X[0, i, :][ix[-1]] = 1.
    ix = np.argmax(model.predict(X[:, :i+1,:])[0], 1)
    y_char.append(idx_to_char[ix[-1]])
  return ''.join(y_char)


# In[20]:


from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
# callback to save the model if better
filepath="tgt_model.hdf5"
save_model_cb = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callback to stop the training if no improvement
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=0)
# callback to generate text at epoch end
class generateText(Callback):
    def on_epoch_end(self, batch, logs={}):
        print(generate_text(self.model, 100))
        
generate_text_cb = generateText()

callbacks_list = [save_model_cb, early_stopping_cb, generate_text_cb]


# In[21]:


model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=NB_EPOCHS, callbacks=callbacks_list, validation_split=VAL_SPLIT)


# In[ ]:


from keras.models import load_model
filepath="tgt_model.hdf5"
model = load_model(filepath)


# In[ ]:


generate_text(model, 100)

