import os
import json
import music21 as m21
from music21 import *
from tensorflow import keras
import numpy as np 
us = environment.UserSettings()
# us.create()
us['musicxmlPath'] = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
us['musicxmlPath']

Dataset_path_kern = r"deutschl\erk"
Acceptable_durations = [
    0.25,0.5,0.75,1.0,1.5,2,3,4    
]
Save_dir = r'C:\Users\HP\Desktop\Melody_Gen\dataset'
single_file_dataset ='file_dataset'
Sequence_length = 64
Mapping_path = 'mapping.json'

def load_songs_kern(dataset_path):
    '''
    go through all the files in the dataset and load each of them with 
    the help of music21 software, additionally we want to load all the songs
    so store them in a list 
    '''
  
    songs = []
    for path , subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path,file))
            # filter out non kern files 
                songs.append(song)
    return songs

def has_acceptable_duration(song,acceptable_duration):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in Acceptable_durations:
            return False
    return True



# def transpose(song):
#     """Transposes song to C maj/A min
#     :return transposed_song (m21 stream):
#     """

#     # get key from the song
#     parts = song.getElementsByClass(m21.stream.Part)
#     measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
#     key = measures_part0[0][4]

#     # estimate key using music21
#     if not isinstance(key, m21.key.Key):
#         key = song.analyze("key")

#     # get interval for transposition. E.g., Bmaj -> Cmaj
#     if key.mode == "major":
#         interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
#     elif key.mode == "minor":
#         interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

#     # transpose song by calculated interval
#     tranposed_song = song.transpose(interval)
#     return tranposed_song


def transpose(song):
    return song




def song_encoder(song,time_step=0.25):
    encoded_song = []

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def load(file_path):
    with open(file_path,'r') as fp:
        song = fp.read()
    return song 


def create_single_file_for_dataset(dataset_path, file_dataset_path,sequence_len):
    new_song_delim = '/ ' * sequence_len  
    songs = ''
    # load enconded songs and add delimiters
    for path ,_,files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs + song + ' ' + new_song_delim
    songs = songs [:-1] # REMOVES  the space 

    #save string containing all dataset
    with open (file_dataset_path,'w')as fp:
        fp.write(songs)
    return songs 


def create_mapping(songs, mapping_path):
    """Creates a json file that maps the symbols in the song dataset onto integers
    :param songs (str): String with all songs
    :param mapping_path (str): Path where to save mapping
    :return:
    """
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

def preprocess(dataset_path):

    # load folk songs
    print("Loading songs...")
    songs = load_songs_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i,song in enumerate(songs):

        # filter out songs that have non-acceptable durations
        if not has_acceptable_duration(song, Acceptable_durations):
            continue

        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = song_encoder(song)

        # save songs to text file
        save_path = os.path.join(Save_dir,str(i))
        with open(save_path,'w')as fp:
            fp.write(encoded_song)
        


def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(Mapping_path, "r") as fp:
        mappings = json.load(fp)

    # transform songs string to list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    """Create input and output data samples for training. Each sample is a sequence.

    :param sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars

    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    # load songs and map them to int
    songs = load(single_file_dataset)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(Dataset_path_kern)
    songs = create_single_file_for_dataset(Save_dir, single_file_dataset, Sequence_length)
    create_mapping(songs, Mapping_path)
    inputs, targets = generate_training_sequences(Sequence_length)
if __name__ == '__main__':
  main()

 