import tensorflow as tf
import numpy as np
import time
import random

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from PIL import Image
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Concatenate
from keras.models import load_model
from keras.preprocessing.image import apply_affine_transform
from keras.losses import CategoricalCrossentropy
from django.http import HttpResponse

CHARACTER_SET = "<π!ŌŠ/нdêồfRkХÇỳñ©8•3yMôĠр>ņá(®ǒàơ¡J”èøěǔư≤.@î,тęşи âV¨s∂öằố[ừ6ḷaˇ7}ǐлpśòÖɔ±≠ºột{ů¯ﬂế\"KĐłćčÉÅçāğ†o^C14e∏¸žuh;ŞE˜=ü¥wēıểÄ°˙‚v2‹-ëуí9HČWœ€Yj:–Oụû&µк≥Q)0́äĀňAØ*̂m…xīPƒIú‘`Zšé|æăFïBṭō∞—óșãgńS£būßė%ả5G‡´+ớz¬ʻq'⁄«˚rTḍ$“¿Lấ≈ªő~¢UiXÆð§N„›"
CLASSES = ['non_character_pixle'] + [char for char in CHARACTER_SET]

def reduce_patches(input_tensor):
    pass
    # loop over each row in the tensor
    # if the pixle is not the same as the one behind the one above or the one next to above add to a list
    # skip of type non_character pixle.
    # add list to list of lists and when done normalize all lists to same length.
    # transform to tensor if you want to.


def encode_u_net(input, conv1, conv2, normalizer, pool=None):
    convoluting = conv1(input)
    convoluting = conv2(convoluting)
    normalized = normalizer(convoluting, training=False)

    if pool: return pool(normalized), normalized
    else: return normalized

def decode_u_net(prev_step_input, skip_content_input, trans, concat, conv1, conv2):
    conv_up = trans(prev_step_input)
    merge_fresh_and_processed = concat([conv_up, skip_content_input])
    convoluting = conv1(merge_fresh_and_processed)
    convoluting = conv2(convoluting)

    return convoluting

def preprocess_image(image):
    image_tensor = tf.image.decode_image(image, dtype=tf.float32)
    if len(image_tensor.shape) >= 3: rgb_image_tensor = tf.cast(image_tensor[..., :3], tf.float32)
    reduced_image = tf.reduce_mean(rgb_image_tensor, axis=2)
    expanded_dimentions = tf.expand_dims(reduced_image, axis=0)
    expanded_dimentions = tf.expand_dims(expanded_dimentions, axis=-1)

    return expanded_dimentions

def degs_to_rads(degs):
    return degs * 3.14159 / 180.0

def generate_batches(question_img, answer_img):
    batch_size = 2
    question_batch = tf.repeat(question_img, batch_size, axis=0)
    answer_batch = tf.repeat(answer_img, batch_size, axis=0)

    for i in range(batch_size):
        rotation = degs_to_rads(random.uniform(0, 10))
        shear = random.uniform(0, 0.2)

        transformed_question = apply_affine_transform(question_batch[i], theta=rotation, shear=shear)
        transformed_answer = apply_affine_transform(answer_batch[i], theta=rotation, shear=shear)

        question_batch = tf.tensor_scatter_nd_update(question_batch, [[i]], [transformed_question])
        answer_batch = tf.tensor_scatter_nd_update(answer_batch, [[i]], [transformed_answer])

    one_hot_encoded = purify_answers(answer_batch)

    # print(f'hot {one_hot_encoded.shape}')
    # print(f'not {question_batch.shape}')

    return question_batch, one_hot_encoded

def save_img_to_folder(data, name, argmax=False):
    if argmax: data = np.argmax(data, axis=-1)
    squeezed_prediction = np.squeeze(data)
    tensor = tf.convert_to_tensor(squeezed_prediction, dtype=tf.uint8)
    image = Image.fromarray(tensor.numpy() ,'L')
    image.save(f'/Users/alexander.fransson/Documents/GitHub/tastetriumph.com/backend/triumph_backend/media/{name}')

def expand_borders(value, tensor, width_axis, height_axis):
    up = tf.where(tf.equal(tf.roll(tensor, shift=1, axis=height_axis), value), value, tensor)
    down = tf.where(tf.equal(tf.roll(up, shift=-1, axis=height_axis), value), value, up)
    left = tf.where(tf.equal(tf.roll(down, shift=-1, axis=width_axis), value), value, down)
    right = tf.where(tf.equal(tf.roll(left, shift=1, axis=width_axis), value), value, left)

    return right

def purify_answers(tensors):
    tensors_scaled = tensors * 255
    blackened = tf.where(tf.equal(tensors_scaled, 255), 0, tensors_scaled)
    darkness_expanded = expand_borders(0, blackened, 1, 2) 
    one_hot_encoded = tf.keras.utils.to_categorical(tf.cast(darkness_expanded, dtype=tf.int8), num_classes=231)

    return one_hot_encoded

class OCRAI(Model):

    def __init__(self):
        super(OCRAI, self).__init__()
        self.num_classes = len(CLASSES)

        self.conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv2 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.normalize1 = BatchNormalization()
        self.pool1 = MaxPooling2D(pool_size=(2,2))

        self.conv3 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv4 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.normalize2 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(2,2))

        self.conv5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv6 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.normalize3 = BatchNormalization()
        self.pool3 = MaxPooling2D(pool_size=(2,2))

        self.conv7 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv8 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.normalize4 = BatchNormalization()
        self.pool4 = MaxPooling2D(pool_size=(2,2))

        self.bottleneck1 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.bottleneck2 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.normalize5 = BatchNormalization()

        self.trans1 = Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')
        self.concat1 = Concatenate(axis=3)
        self.conv9 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv10 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')

        self.trans2 = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')
        self.concat2 = Concatenate(axis=3)
        self.conv11 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv12 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')

        self.trans3 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')
        self.concat3 = Concatenate(axis=3)
        self.conv13 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv14 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')

        self.trans4 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')
        self.concat4 = Concatenate(axis=3)
        self.conv15 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv16 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='HeNormal')

        self.output_layer = Conv2D(self.num_classes, (1,1), activation='softmax')

    def call(self, input):

        abstracted, skip_content1 = encode_u_net(input, self.conv1, self.conv2, self.normalize1, self.pool1)
        abstracted, skip_content2 = encode_u_net(abstracted, self.conv3, self.conv4, self.normalize2, self.pool2)
        abstracted, skip_content3 = encode_u_net(abstracted, self.conv5, self.conv6, self.normalize3, self.pool3)
        abstracted, skip_content4 = encode_u_net(abstracted, self.conv7, self.conv8, self.normalize4, self.pool4)
        bottle_neck = encode_u_net(abstracted, self.bottleneck1, self.bottleneck2, self.normalize5)
        realised = decode_u_net(bottle_neck, skip_content4, self.trans1, self.concat1, self.conv9, self.conv10)
        realised = decode_u_net(realised, skip_content3, self.trans2, self.concat2, self.conv11, self.conv12)
        realised = decode_u_net(realised, skip_content2, self.trans3, self.concat3, self.conv13, self.conv14)
        realised = decode_u_net(realised, skip_content1, self.trans4, self.concat4, self.conv15, self.conv16)

        return self.output_layer(realised)

class OcraiView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):

        data = request.data
        question_img = data['image'].read()
        answer_img = data['answer_as_pixles'].read()

        question = preprocess_image(question_img)
        answer = preprocess_image(answer_img)

        question_batch, answer_batch = generate_batches(question, answer)

        save_img_to_folder(question_batch[0] * 255.0, 'question.png')
        save_img_to_folder(answer_batch[0], 'answer.png', argmax=True)

        model = load_model('/Users/alexander.fransson/Documents/GitHub/tastetriumph.com/backend/triumph_backend/triumph_backend/ocrai/nn')

        class_weights = {0:0.1}

        for i in range(230):
            class_weights[i] = 1.

        model.compile(
            loss=CategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
            metrics=['accuracy']
        )
        model.fit(
            question_batch,
            answer_batch,
            batch_size=2,
            epochs=2,
            verbose=2,
            class_weight=class_weights
        )
        model.save('/Users/alexander.fransson/Documents/GitHub/tastetriumph.com/backend/triumph_backend/triumph_backend/ocrai/nn')

        return Response({
            'message':'training success',
        })

class OCRAIPrediction(APIView):
    permission_classes = [AllowAny]

    def get(self, *args, **kwargs):
        
        with open('/Users/alexander.fransson/Documents/GitHub/tastetriumph.com/backend/triumph_backend/media/test.png', 'rb') as f:
            return HttpResponse(f.read(), content_type='image/png')

    def post(self, request):
        data = request.data
        image = data['image'].read()
        question = preprocess_image(image)
        model = OCRAI()
        prediction = model.predict(question)

        save_img_to_folder(prediction * 255, 'test.png', argmax=True)
        url = f'http://127.0.0.1:8000/ocrai/predict?t={time.time()}'

        return Response({'url':url})

##############################
# Texsorflow subclassing api #
##############################
"""
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test, x_train = x_test/255.0, x_train/255.0

# adding a channels dimension, like colour for some reason to this black and white set, it is or i to fit in the conv2d

x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# batche referres to dividing the dataset into smaller parts that can be trained on in parrarel for more effective training

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 2, activation='relu') # Convolution is the thing with matrix over matrix to recognise shapes
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

model = MyModel()

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.legacy.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#train the model

@tf.function
def train_step(images, lables):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        losses = loss(lables, predictions)
    gradients = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(losses)
    train_accuracy(lables, predictions)

# test the model

@tf.function
def test_step(images, lables):
    predictions = model(images, training=False)
    test_loss(loss(lables, predictions))
    test_accuracy(lables, predictions)

# run training and test

for epoch in range(5):
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
"""
###############################
# Recurrant neural nework RNN #
###############################

# text classification

"""

# Preprocess data, take away unnececary words, make equal length and turn words into numbers 
#############################################################################################

impot tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time

df = pd.read_csv("data/twitter_train.csv")

#df.shape() -> (7613,5)
# five columnns, id, keyword, location, text, target
# target == 1 -> Disaster, 0 -> No dissaster

df.head()

import re
import string

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

# def remove_punct(text):
    translator O str.maketrans("","", string.punctuation)
    return text.translate(translator)

df["text"] = df.text.map(remove_URL)
df["text"] = df.text.maap(remove_punct)

# remove stopwords
# Stopwords are common words that search engines have been progeammed to ignore such as "the", "a", "am" and so on.
# pip install nltk     "nltk is a natural language library".
import nltk
nltk.download('stopwords')
from nltk import stopwords

stop = set(stopwords.words("english"))

def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

df["text"] = df.text.map(remove_stopwords)

from collections import Counter

def count_unique_words(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count_word + 1
        return count

counter = count_unique_words(df.text)
# counter can return tuples with (string, ocurrances)
num_unique_words = len(counter)

# split into training and validation datasets
train_size = int(df.shape[0] * 0.8)

train_df = df[:train_size]
val_df = df[train_size:]

# split text and lables

train_sentances = train_df.text.to_numpy()
train_labels = train_df.target.to_numpy()
val_sentances = val_df.text.to_numpy()
val_labels = val_df.target.to_numpy()

# Tokenize. make words numbers

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sequences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sequences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)

# Pad all the sequences to have the same lenght.
from tesorflow.keras.preprocessing.sequence import pad_sequences

max_length = 20
train_padder = pad_sequences.(train_sequences, maxlen=max_length, padding="post", truncation="post") # post padds with zero
val_padded = pad_sequences.(train_sequences, maxlen=max_length, padding="post", truncation="post")

# Create LSTM model
####################

from tensorflow.keras import layers

model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 32, input_length=max_length))
model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(q, activation="sigmoid"))

model.summary()

model.compile(
loss = keras.losses.BinaryCrossentropy(from_logits=False)
optimizer= keras.optimizer.legacy.Adam
metrics = ["accuracy"]
)
model.fit(train_padded, train_lables, epochs=20, validation_data=(val_padded, val_lables), verbose)

"""

# RNN tutorial

"""
# Recurrant neural network allows prevous outputs to be used as inputswhile having hidden states.

# RNN layers classes include SimpleRNN, LSTM, GRU

import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, models, Input
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test, y_test = x_train/255.0, y_train/255.0
# Input size = 28 -> features
# sequence length = 28 -> time steps

model = models.Sequential()
model.add(Input(shape=(28,28))) # seq length, input size
#model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu')) # reduces rows into numbers, return sequence makes shape (28,128)
model.add(layers.GRU(128, return_sequences=False, activation='relu')) # shape (128,)

model.add(layers.Dense(10))

print(model.summary())


model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    verbose=2
)

model.evaluate(
    x_test,
    y_test,
    batch_size=64,
    verbose=2
)
"""
#####################
# transfer learning #
#####################

# Transfer learning is when you take an already trained model and only adapts the last layers of it to fit your needs
# This has the benefit of faster training
# Pretrained models can be found at http://tensorflow.org/api_docs/python/keras/applications

# kaggle is a site with exercises and datasets for machinelearning an it also has a bunch of tutorials for programming concepts
# keras.preprocessing.image.ImageDataGenerator can do various image modifications to scale and diversiy the dataset like setting 
# rotation_range, horizontal_flip, width_shift, height_shift, shear_range, zoom_range etc.

# A keras callback is a function that runs after each epoch  like stopping it if it does not improve

# Overfitting can be a real problem when a model gets over adapted to the training data

##################################################
# Recognies number and left or right handwriting #
##################################################
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input, Model, losses, optimizers
from keras.datasets import mnist
import numpy as np

# Create Model (how should it look, what should it doo)
#######################################################
inputs = Input(shape=(28,28))
flatten = layers.Flatten()
dense1 = layers.Dense(128, activation='relu')
dense2 = layers.Dense(10, activation='softmax', name='categoy_output')
dense3 = layers.Dense(1, activation='sigmoid', name='leftright_output')

x = flatten(inputs)
x = dense1(x)
outputs1 = dense2(x)
outputs2 = dense3(x)

model = Model(inputs=inputs, outputs=[outputs1, outputs2], name='mnist_model')

model.summary()

#Compile Model (how is success mesured)
#######################################
model.compile(
    loss={
        'categoy_output': losses.SparseCategoricalCrossentropy(from_logits=False),
        'leftright_output': losses.BinaryCrossentropy(from_logits=False)
    }, 
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001), 
    metrics=["accuracy"]
)

#Train model (let it train on a dataset)
# is data, y is lables
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255

# create a matrix with the shape of y_test filled with zeroes
y_leftright = np.zeros(y_train.shape, dtype=np.uint8)

# enumerate makes the index know in an array (useful)
# this is a dummy set that sets some of them to right
for index, y in enumerate(y_train):
    if y > 5:
        y_leftright[index] = 1

print(y_train.dtype, y_train[0:20])
print(y_leftright.dtype, y_leftright[0:20])

# Train
########

model.fit(
    x_train,
    y={
    'categoy_output':y_train,
    'leftright_output':y_leftright
    },
    epochs=5,
    batch_size=64,
    verbose=2
)
"""
##############################
# Functional Tenosorflow API #
##############################
"""
# more flexible than sequential api

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, models, Input, Model, applications

# one input one output
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

print(model.summary())

# funktional can have multipele outputs and inputs or paths

inputs = Input(shape=(28,28))

flatten = layers.Flatten()
dense1 = layers.Dense(128, activation='relu')
dense2 = layers.Dense(10)

# Create another output layer
dense2_2 = layers.Dense(1)

x = flatten(inputs)
x = dense1(x)

outputs = dense2(x)
outputs_2 = dense2_2(x)

# you can create more inputs just like outputs, but why though? Maybe To find diferences or similarities between inputs.
model = Model(inputs=inputs, outputs=[outputs, outputs_2], name='functional_model')

print(model.summary())

# turn a functional model to a sequential by itterating over its layers
# this however only workes if it is linear.
# new_model = models.Sequential()
# for layer in model.layers:
#     new_model.add(layer)

# to turn a functional into a sequential
# inputs = Input(shape=(28,28))
# x = new_model.layers[0](inputs)
# for layer in new_model.layers[1:-1]:
#     # here we can leave out layers by if statement  or exclude the last like [1:-1] or have any other sub array
#     x = layer(x)
# outputs = x

# print(model.summary())

# Models with multiple inputs and outputs 
# Shared layers 
# Extract and reuse nodes in the graph of layers

# get layers from model
inputs = model.inputs
outputs = model.outputs

input0 = model.layers[0].input
output0 = model.layers[0].output

print(inputs)
print(outputs)
print(input0)
print(output0)

# transfer learning.
#####################

#an already trained model that we only want to change the last layer of.
base_model = applications.VGG16()

x = base_model.layers[-2].output
new_output = layers.Dense(1)(x)
new_model = Model(inputs=base_model.inputs, outputs=new_output)

"""
##################
# Saveing the AI #
##################
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = keras.models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10)
])

model.compile(
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
model.fit(x_train, y_train, batch_size=64, epochs=5, shuffle=True, verbose=2)
model.evaluate(x_test, y_test, verbose=2)

# Ways to save the model

# 1) Save the whole model
#########################

# There are two formats, SavedModel ... and HDF5 .h5

#makes a folder
#model.save("nn")

#makes one file
# model.save("neural_net.h5")

# new_model = keras.models.load_model("neural_net.h5")
# new_model.evaluate(x_test, y_test, verbose=2)

# 2) Save the weights
######################
# model.save_weights("neural_net.h5")
# model.load_weights("neural_net.h5")

# # 3 Save the architecture to json
# #################################

# json_string = model.to_json()

# with open("nn_model.json", "w") as file:
#     file.write(json_string)

# with open("nn_model.json", "r") as file:
#     loaded_json = file.read()

# new_model = keras.models.model_from_json(loaded_json)
"""

#####################
# Convulution NN AI #
#####################

# the first parameter stands for the number of filters and are responsible for detecting shapes. one filter is responsible for one shape t.ex an edge

"""
# Uses Convulution layers

# Convolution is done by sliding a smaller matrix called a kernal over the larger input matrix to catch patterns 
# and save the ai from reading things one row at a time.

# Pooling seames to just be the diciding of the matrix and taking the largest value as a representative in the pool matrix för att minska värdena

import tensorflow as ts
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

# dataset of images
cifar10 = keras.datasets.cifar10


# get test and training images with lables
(train_images, train_lables), (test_images, test_lables) = cifar10.load_data()
print(train_images.shape)

# Normalise: dela alla tal i matriserna på 255 för lättare minnes hantering? rgb range är 255 men 0-1 är lättare för ai att hantera
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# def show():
#     plt.figure(figsize=(10,10))
#     for i in range(16):
#         plt.subplot(4,4,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(train_images[i], cmap=plt.cm.binary)
#         # Cifar lables are arrys 
#         plt.xlabel(class_names[train_lables[i][0]])
#     plt.show()

# show()

model = keras.models.Sequential()
# adds a 2d conv layer becouse remember matrixes can have many dimentions,
# the image has a width and hight of 32 and the conv matrix is a 3x3 the input also has 3 colour channels RGB
# no padding=valid meanes that the convulation matrix starts one pixle in so better not have important data on the edges, like if padding is already pressant
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu", input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

print(model.summary())

o = keras.optimizers.legacy.Adam(lr=0.001)
l = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
m = ["accuracy"]

model.compile(
    optimizer=keras.optimizers.legacy.Adam(lr=0.001), 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=["accuracy"]
)

model.fit(train_images, train_lables, epochs=10, batch_size=64, verbose=2)
model.evaluate(test_images, test_lables, batch_size=64, verbose=2)
"""

###############
# Basic NN AI #
###############
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# get data
mnist = keras.datasets.mnist

# split into train and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 

# crate a neural network with an input layer for al params, connection layers and output layers.
# keras is a set of hand written numbers 0 -> 9 as there are 10 alternatives the last layer have to be 10 as they represent an alternative.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10,)
])

# funktion that decides how to mesure inacuracy
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# function that is used to correct the model, lr = how fast it learns?
optimiser = keras.optimizers.legacy.Adam(learning_rate=0.001)

# metrics to track, right now it is just accuracy which I guess is distance from loss.
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optimiser, metrics=metrics)
model.fit(x_train, y_train, batch_size=64, epochs=5, shuffle=True, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model(x_test)
pred0 = predictions[0]
label0 = np.argmax(pred0)
print(f"{pred0}\n{label0}")

#normalisation = ((input-medelvärde)/normalfördelning) -> ger förändring i procent
#nns skapar i princip en funktion som ger ett gämförelse tal för om saker stämmer eller inte. Fler lager gör att gunktionen kan bli mindre linjär
"""
