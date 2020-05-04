import pandas as pd
import numpy as np
import pretty_midi
import music21
import re
from random import sample
from keras.utils import to_categorical

# Create vocab dictionaries
def create_dictionaries(text_data):
    '''Create two dictionaries: one from character to integer,
    and one from integer to character'''
    char_list = list(text_data)
    vocab = list(set(char_list))
    
    # Dictionary with character to integer
    vocab_dict = {i: j for i,j in enumerate(vocab)}
    
    # Dictionary with integer to character
    vocab_dict_rev = {j: i for i,j in enumerate(vocab)}
        
    return vocab_dict, vocab_dict_rev


def encoder(text_data, dictionary):
    '''Convert text data into a numeric list.'''
    character_nums = list(text_data)
    
    for i in range(len(character_nums)):
        character_nums[i] = dictionary[character_nums[i]]
        
    return character_nums


def decoder(binary_matrix, dictionary):
    '''Convert numeric list a text string.'''
    text_list = []
    
    for row in binary_matrix:
        max_ind = np.argmax(row)
        text_list.append(dictionary[max_ind])
    
    return "".join(text_list)


def create_training(char_nums, str_length, vocab_size):
    '''Create training dataset with x and y values from your numeric list.
    The x data is a list of all numeric sequences, and the y data is the next character.'''
    
    # The x_values begin at the starting indices and are str_length characters long
    # The y_values are one character after the end of the x_values
    x_data = np.array(char_nums[0:str_length])
    y_data = [char_nums[str_length]]
    for i in range(1, len(char_nums)-str_length):
        x_data = np.vstack((x_data, np.array(char_nums[i:i+str_length])))
        y_data.append(char_nums[i+str_length])
    
    # Convert x and y data to tensors
    x = to_categorical(x_data, num_classes=vocab_size)
    y = to_categorical(y_data, num_classes=vocab_size)
    
    return x, y


