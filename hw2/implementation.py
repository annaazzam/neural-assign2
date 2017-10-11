import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile


batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""

    # # unzip our reviews file
    # try: #unzip if it needs to be unzipped
    #   gunzip('/reviews.tar.gz')
    # except:
    #   pass

    # open neg and pos

    arr = []

    for filename in os.listdir("pos"):
        data = open("pos/" + filename, 'r', encoding="utf-8")
        data = data.read()

        #PREPROCESSING
        # convert to lowercase:
        data = data.lower() # convert to lowercase
        
        # remove punctuation:
        exclude = set("""!@#$%^&*()<>,./\[];"'?-~""")
        data = ''.join(ch for ch in data if ch not in exclude)

        # strip out unnecessary words:
        unnecessary_words = []
        data = ' '.join(word for word in data.split(" ") if word not in unnecessary_words)

        # "vectorize" it:
        row = []
        for word in data.split(" "):
            row.append(glove_dict[word] if word in glove_dict.keys() else 0)

        # zero pad:
        row = np.pad(row, (0, 40), 'constant')

        # add only first 40 words
        arr.append(row[:40])

    # repeat for negative reviews

    data = np.array(arr)



    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    #data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    
    word_index_dict = {}
    embeddings = []

    with open("glove.6B.50d.txt",'r',encoding="utf-8") as f:
        i = 1
        for line in f.readlines():
            tokens = line.split(" ")
            word = tokens[0]
            word_index_dict[word] = i
            embeddings.append(tokens[1:]) # word vector?
            i += 1

    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())


    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
