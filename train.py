import numpy as np 
import pandas as pd

import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import iplot

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import (LSTM,Embedding,BatchNormalization,Dense,TimeDistributed,Dropout,Bidirectional,Flatten,GlobalMaxPool1D,GlobalAveragePooling1D,MultiHeadAttention,LayerNormalization,SimpleRNN)
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    accuracy_score
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.utils import pad_sequences
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import xgboost as xgb
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping
from sklearn.neural_network import MLPClassifier

from keras_tuner import RandomSearch

import utils

def LSTM_model():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

df = pd.read_csv("spam.csv",delimiter=',',encoding='latin-1')

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
df.rename(columns = {"v1": "target", "v2": "text"}, inplace = True)

x = ['ham', 'spam']
y = df.groupby("target")["target"].agg("count").values

layout = go.Layout(title={'text':'Proportional Distribution of the Target Variable',
                         'y':0.9,
                         'x':0.5,
                         'xanchor':'center',
                         'yanchor':'top'},
                  template = 'plotly_dark')

fig = go.Figure(data=[go.Bar(
    x = x, y = y,
    text = y, textposition = 'auto',
    marker_color = "slateblue"
)], layout = layout)
#fig.show()

colors = ["slateblue", "darkred"]

fig = go.Figure(data=[go.Pie(labels = df['target'].value_counts().keys(),
                             values = df['target'].value_counts().values,
                             pull = [0, 0.25])])

fig.update_traces(hoverinfo ='label',
                  textinfo ='percent',
                  textfont_size = 20,
                  textposition ='auto',
                  marker=dict(colors=colors,
                              line = dict(color = 'lightgray',
                                          width = 1.5)))

fig.update_layout(title={'text': "Percentages of the Target Values",
                         'y':0.9,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'},
                  template='plotly_dark')

#fig.show()

df['text_len'] = df['text'].apply(lambda x: len(x.split(' ')))
#print(df.head())

ham = df[df["target"] == "ham"]["text_len"].value_counts().sort_index()
spam = df[df["target"] == "spam"]["text_len"].value_counts().sort_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x = ham.index, y = ham.values, name = "ham", 
                         fill = "tozeroy"))
fig.add_trace(go.Scatter(x = spam.index, y = spam.values, name="spam",
                        fill = "tozeroy"))
fig.update_layout(title={'text': "Distributions of Target Values",
                         'y':0.9,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'},
                  template='plotly_dark')
fig.update_xaxes(range=[0, 50])
fig.update_yaxes(range=[0, 450])
#fig.show()

df['text'] = df['text'].apply(utils.clean_text)

#print(df.head())

stop_words = stopwords.words("english")
df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

#print(df.head())

stemmer = nltk.SnowballStemmer("english")
df["text"] = df["text"].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))

#print(df.head())

text = " ".join(i for i in df.text)

wc = WordCloud(background_color = "black", width = 1200, height = 600,
               contour_width = 0, contour_color = "#410F01", max_words = 1000,
               scale = 1, collocations = False, repeat = True, min_font_size = 1)

wc.generate(text)

#plt.figure(figsize = [15, 7])
#plt.title("Top Words in the Text")
#plt.imshow(wc)
#plt.axis("off")
#plt.savefig("wordcloud.png")

lb = LabelEncoder()
df["target"] = lb.fit_transform(df["target"])

X = df['text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

vec = CountVectorizer()
vec.fit(X_train)

X_train_dtm = vec.transform(X_train)
X_test_dtm = vec.transform(X_test)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = pad_sequences(sequences,maxlen=max_len)

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]

print("Naive Bayes Multinomial: {}".format(metrics.accuracy_score(y_test, y_pred_class)))
#print(metrics.accuracy_score(y_test, y_pred_class))

pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=80,
        use_label_encoder=False,
        eval_metric='auc',
    ))
])



pipe.fit(X_train, y_train)

y_pred_class = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)

print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

model = LSTM_model()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,y_test)

model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_len))
model.add(LSTM(64, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(sequences_matrix,y_train,batch_size=128,epochs=10,
            validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

accr = model.evaluate(test_sequences_matrix,y_test)

model = Sequential([
    Embedding(max_words, 100, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 5. Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(sequences_matrix, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

# 6. Evaluate the model
test_loss, test_acc = model.evaluate(test_sequences_matrix, y_test)
print(f'Test accuracy DNN keras: {test_acc:.4f}')


model = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', early_stopping=True, random_state=42)
model.fit(sequences_matrix, y_train)
y_pred = model.predict(test_sequences_matrix)
accuracy = accuracy_score(y_test,y_pred)
print(f'Test accuracy DNN scikit-learn: {accuracy:.4f}')

# 4. Define the DNN model with tunable hyperparameters
def build_model(hp):
    embedding_dim = hp.Int('embedding_dim', min_value=32, max_value=128, step=32)
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5. Set up Keras Tuner and perform hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='sms_spam_tuner',
    project_name='sms_spam_classifier'
)

tuner.search_space_summary()

tuner.search(sequences_matrix, y_train, epochs=10,  validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

# 6. Get the best model and its hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
print(f"Embedding dimension: {best_hyperparameters.get('embedding_dim')}")
print(f"Dense units: {best_hyperparameters.get('dense_units')}")

# 7. Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(test_sequences_matrix, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# 8. Save the best model

history = best_model.fit(
    sequences_matrix, y_train, batch_size=32,
    epochs=10, validation_split=0.2
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy for DNN model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('DNN.png')


# 4. Define the Transformer model with tunable hyperparameters
def build_model(hp):
    input_dim = max_words
    output_dim = hp.Int('output_dim', min_value=32, max_value=128, step=32)
    num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
    dff = hp.Int('dff', min_value=32, max_value=128, step=32)
    rate = hp.Float('rate', min_value=0.1, max_value=0.5, step=0.1)

    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim, output_dim, input_length=max_len)(inputs)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=output_dim)(embedding_layer, embedding_layer)
    attn_output = LayerNormalization(epsilon=1e-6)(embedding_layer + attn_output)
    ffn_output = Dense(dff, activation='relu')(attn_output)
    ffn_output = Dense(output_dim)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(attn_output + ffn_output)
    pooling = GlobalAveragePooling1D()(ffn_output)
    outputs = Dense(1, activation='sigmoid')(pooling)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5. Set up Keras Tuner and perform hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='sms_spam_tuner',
    project_name='sms_spam_transformer'
)

tuner.search_space_summary()

tuner.search(sequences_matrix, y_train, epochs=100, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

#6. Get the best model and its hyperparameters

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
print(f"Output dimension: {best_hyperparameters.get('output_dim')}")
print(f"Number of heads: {best_hyperparameters.get('num_heads')}")
print(f"Feed-forward network dimensions: {best_hyperparameters.get('dff')}")
print(f"Dropout rate: {best_hyperparameters.get('rate')}")

#7. Retrain the best model on the entire dataset and get the history

history = best_model.fit(sequences_matrix, y_train, batch_size=32, epochs=10, validation_split=0.2)
# 8. Plot the training and validation accuracy graphs
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy for Transformer Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Transformer.png')

# 4. Define the RNN model with tunable hyperparameters
def build_model(hp):
    input_dim = max_words
    output_dim = hp.Int('output_dim', min_value=32, max_value=128, step=32)
    rnn_units = hp.Int('rnn_units', min_value=32, max_value=128, step=32)
    rate = hp.Float('rate', min_value=0.1, max_value=0.5, step=0.1)

    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim, output_dim, input_length=max_len)(inputs)
    rnn_layer = SimpleRNN(rnn_units, return_sequences=True, dropout=rate)(embedding_layer)
    pooling = GlobalAveragePooling1D()(rnn_layer)
    outputs = Dense(1, activation='sigmoid')(pooling)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5. Set up Keras Tuner and perform hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='sms_spam_tuner',
    project_name='sms_spam_rnn'
)

tuner.search_space_summary()

tuner.search(sequences_matrix, y_train, epochs=100, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

# 6. Get the best model and its hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
print(f"Output dimension: {best_hyperparameters.get('output_dim')}")
print(f"RNN units: {best_hyperparameters.get('rnn_units')}")
print(f"Dropout rate: {best_hyperparameters.get('rate')}")

# 7. Retrain the best model on the entire dataset and get the history
history = best_model.fit(sequences_matrix, y_train, batch_size=32, epochs=10, validation_split=0.2)

# 8. Plot the training and validation accuracy graphs
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy for RNN Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('RNN.png')

# 4. Define the LSTM model with tunable hyperparameters
def build_model(hp):
    input_dim = max_words
    output_dim = hp.Int('output_dim', min_value=32, max_value=128, step=32)
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    rate = hp.Float('rate', min_value=0.1, max_value=0.5, step=0.1)

    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim, output_dim, input_length=max_len)(inputs)
    lstm_layer = LSTM(lstm_units, return_sequences=True, dropout=rate)(embedding_layer)
    pooling = GlobalAveragePooling1D()(lstm_layer)
    outputs = Dense(1, activation='sigmoid')(pooling)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5. Set up Keras Tuner and perform hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='sms_spam_tuner',
    project_name='sms_spam_lstm'
)

tuner.search_space_summary()

tuner.search(sequences_matrix, y_train, epochs=100, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
print(f"Output dimension: {best_hyperparameters.get('output_dim')}")
print(f"LSTM units: {best_hyperparameters.get('lstm_units')}")
print(f"Dropout rate: {best_hyperparameters.get('rate')}")

# 7. Retrain the best model on the entire dataset and get the history
history = best_model.fit(sequences_matrix, y_train, batch_size=32, epochs=10, validation_split=0.2)

# 8. Plot the training and validation accuracy graphs
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy for LSTM Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('LSTM.png')
