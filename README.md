# Melody_Gen
A generative model with the goal to generate melodies using folk songs inspired by valerio 
3.1 Methodological Approach

The project's methodology revolves around harnessing the power of Long Short-Term Memory neural networks to generate melodically coherent sequences. LSTM networks are capable of learning intricate patterns in sequential data, making them suitable for music generation tasks. The approach involves a dual-stage process: training the LSTM model on a diverse dataset of existing melodies and subsequently leveraging the model's learned knowledge to generate novel melodies.

3.2 Methods of Data Collection and Selection

The bedrock of this project is a meticulously curated dataset of melodies representing diverse musical genres from  http://www.esac-data.org/ 
Data collecting involves sourcing .krn files into MIDI files. MIDI files offer a structured representation of musical information, facilitating the extraction of note sequences, durations, and other relevant attributes. The selection process ensures a representative range of musical styles, enabling the LSTM model to acquire a comprehensive understanding of melodic grammar.

3.3 Methods of Analysis

The generated melodies undergo a multifaceted analysis to evaluate their musical quality and coherence. Objective analysis involves quantifying musical attributes such as note distribution, pitch intervals, and rhythmic patterns. Subjective analysis enlists the judgment of human experts to assess the emotional expressiveness and aesthetic appeal of the melodies. Both approaches converge to provide a holistic perspective on the performance of the LSTM-generated melodies.





4. MUSICAL FOUNDATIONS AND FOREWORD


This section is specifically curated as prerequisite for the experimentation section as there are some basic music concepts that is need to be understood before delving into coding.

what is needed is a good understanding of the following

-	Handling time series
-	Basic understanding of symbolic music representation
-	Basic music theory concepts (pitch,duration,key)
-	Preprocess symbolic music


4.1 Decoding Musical Notation And Scientific Not5ation
A melody consists of a sequence of nodes 
The following is the notation 
 
This is the notation that has been used for decades on the Y axis is the pitch the higher it is the higher the pitch, and on the X axis is the time the longer it is the longer the hold/duration is.
 

 

 
the difference between C and C# is that C# is 2x the frequency of C which is what we call an octave.

A scientific pitch notation consists of 
Note name + octave 
for example 
C and its number for example 3  => C3
C3 means go to C for the 3rd octave / set

While this is a great way to notate music, there needs to be a different way to notate them. for example MIDI note notation.

4.2 Exploring MIDI Notes and Timing Signatures
MIDI is a protocol to play, record and edit music
The way MIDI handles notes is by mapping them to an integer
For example C4 = 60 
 






The beat is a unit of measurement for the duration  

 

 
The 4/4  on the left is called a time signature 
The numerator indicates how many beats are within a bar and the next. 
And denominator indicates the type of note used for example here its 4 so then calculate which note we need 4 of to fit a semibreve (O shape)
The bar in the middle to indicate the end or start of a beat 

Why care about a time signature 
The melodies are going to be shaped differently depending on the time signature
we need to know if the NN learned time signatures and how it can create in 4/4 or 3 / 4



4.3 The concept of KEY
Key at its core is a group of pitches that forms the centre of a piece
Key is composed of 2 elements Tonic + mode 
For example C maj , C being the tonic and maj being the mode
The tonic is defined as the harmonic centre of the piece were the the beginning and the end of a piece is found often. 
 

 
Usually, the major is associated with happiness 
And the minor is associated with sadness 
With simple calculations we come to the conclusion that there are 2 modes and 12 notes 
Which equal 24 keys.


4.4 The approach used to represent music 
-	sample the melody at each 16th of a note so each step is equal to 16th of a note 
-	Whenever a note is hit log the corresponding MIDI note into the NN
-	use ‘’_’’ if a note is longer than a 16th of a note to signal tie-ing that note into the next time set 
-	use “r’’ as the symbol to indicate a rest 
 
To further simplify the process for all the note they will be sampled 4 times for a 4 /4 time signature 
 
 

For the first example 60 is saved in which correspond to C  for 16th of a note and the rest is held for the next 3 16th  of a note, the same thing is done for the 2nd row 
And for the 3rd row there isn’t as long of a hold as the previous 2. 








In music the Score is the whole piece 
And a score consists of multiple parts 
And each part has multiple measures
Inside the measures we have multiple notes
 
This is the intuition behind music21 and this is how we can access the score  

4.5 How to generate melodies
We start with a seed melody (a few notes that we feed the model). For example
seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
And the model is expected to give a prediction for the next predicted note in the sequence/melody.
We then take the predicted note and feed it/append it again to the model and we expect another prediction from the model and continue the same process.

4.6 why not use GANs

1. Model Complexity: GANs are generally more complex to design and train, involving a generator and discriminator network in an adversarial setup. For certain projects, LSTM networks might offer a simpler approach.
2. Domain Knowledge: GANs, while capable of producing novel content, might require more careful tuning to align with musical conventions.
3. Research Focus:
The project's primary goal is to explore the capabilities of LSTM networks and understand how they work in the context of melody generation.
5.	EXPERIMENTATION


 Setting up the environment for recreating the project

Install the following

-python
The programming language used in this project

-Tensorflow
An open-source software library for machine learning and artificial intelligence

-music21
A package that lets you manipulate music data as well as load and convert music file from certain formats into other formats.  From kern , MIDI , MusicXML ->m21  and then again into -> kern , MIDI .It also comes with a lot of methods to estimating the key of a piece 
-numpy
A library adding support for large, multi-dimensional arrays and matrices

-musecore 
An open-source music notation software. It is used by musicians and composers to create, edit, and play back sheet music.

-an IDE of your choosing for this project vscode was used
	
The DATASET

 
Where the steps to fetch the data used in training are the following:
1-	Scroll down and choose humdrum data translations of the database

 

2-	Choosing Europa from the list as shown

 


3-	Choose whichever country’s folk you would like to train your model on, in our case it was deutschl melodies 

 
4-	Lastly click on the blue Z button to download all the files needed 

 



Setting up the environment to view and show the score
To play the pieces in the dataset and to check the model results musecore was used to create an environment variable for the usersettings
Only run the us.create once , within us change the xmlpath to the preferred software
us = environment.UserSettings()
# us.create()
us['musicxmlPath'] = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
us['musicxmlPath']
 
Preprocessing
The needed dependencies are 
-	Os
-	Json
-	Music21
-	Tensorflow
-	Numpy

5.1 LOADING THE DATA
Define a preprocess function that takes in the dataset
Within the function define a list that is going to be used for storing the songs and then loop through the files and check if they are .krn extension , afterwards parse the songs using music 21
def load_songs_kern(dataset_path):
  
    songs = []
    for path , subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    return songs

parameters
-	Dataset_path (the origin of the data used)

5.2 CHECKING FOR ACCEPTABLE DURATIONS
Define a function that takes in the loaded song and an acceptable duration argument 
The purpose of this function is to return the loaded song only if it’s in an acceptable duration.
The acceptable durations should be initialized earlier 
Acceptable_durations = [0.25,0.5,0.75,1.0,1.5,2,3,4]
 0.25 is our series step size and it also corresponds to 1/16th of a note as mentioned before
1/16th note = 1/4        1/2 note = 2
1/8th note = 1/2           1 note = 4
1/4th note = 1
def has_acceptable_duration(song,acceptable_duration):
    for note in song.flat.notesAndRests:
parameters
-	Song (the loaded song)
-	Acceptable_duration (the list initialized earlier)

        if note.duration.quarterLength not in Acceptable_durations:
            return False
    return True

using the .flat method convert all the detail of a score into a list and then further specify by requesting only the notes and rests using notesAndRests method
with this checking if the notes are within the specified values is achieved.

5.3 TRANSPOSING SONGS (into C major and A minor)
The function takes in the song as its parameter.
Some pieces have the key notated while others don’t so it’s imperative to take that into consideration and in such scenario estimate the key using music21
def transpose(song):
    
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
   
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
parameters
-	Song
start by getting the parts of the score and then getting the measures and then the keys where all of those return a list so slicing is needed to get the key which is usually stored at the index number 4 in this dataset.

otherwise if the key can’t be fetched estimate it using music21 by checking if it is an instance of a key and then analyze it using  .analyze and specifying the parameter  “key”.



    if key.mode == "major":
        interval = 	m21.interval.Interval(key.tonic,m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = 	m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))
    tranposed_song = song.transpose(interval)
    return tranposed_song

next get the interval for transposing the songs from all those varying keys into 2 keys only Cmaj and A minor.
For example if a song is in bmaj calculate the distance between both of them and transpose by that interval from bmaj to cmaj.

For that Start by calculating the interval and check if the key mode is in major then
use m21.interval that takes in the tonic and the pitch C to calculate the interval  
and the same process is done for minor keys but instead pass in “A” instead of C
Lastly simply use the calculated interval to transpose the song and return it.


5.4 ENCODING THE SONGS.
encoding in a manner that fits a time series to allow feeding the data into the NN
To explain encoding first off assume a pitch of 60  , and a duration of 1.0 the following should be encoded as a list where each item corresponds to 16th note =1/4beat  so for this example it will be encodes like this [60,’_’,’_’,’_’]
Additionally for this task we flatten the song to capture the notes and the rests 

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
            # if it's the first time we see a note/rest, let's encode 		it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song
parameters 
-	Song
-	time_step 
Coding this part involves checking if the I’th element in the song is a note or event and converting them to its midi form 
then cast the steps variable into an int because that variable is used in a for loop.
Afterward convert them into a representation akin to timeseries
The steps variable divides the event.duration.quarterlength of the current symbol with our time_step variable to check how many steps are there within this 1 event so 
For example if the event is [ 60 , 60 , 60 ,60]
append only the first 60 and the rest is hold in the form of “_” 
Next step is returning the encoded song in the form of a string where .map and use .join. 
The mapping here only maps them to string.

5.5 MERGING DATASET.
merging all the dataset elements into one file and separate them by some split.
The reasoning for this is convenience 
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
parameters
-	dataset path
-	file_dataset_path(the resulting/destination file)
-	sequence length
 The song_delim indicates the end of a song and the sequence len indicate how many 
Delims are passeed through when training the NN and the reason for multiplying the song delim by sequence_len is because we want to have the same amount of delim as the sequence length in this case 64 
then go through all the songs and merge them 

5.6 CREATING MAPPINGS.
After saving the file an issue arises were have underscores and “r” symbol which aren’t integers so the NN can’t learn from it. 
here is where we need to map all the symbols and integers 
def create_mapping(songs, mapping_path):
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
parameters
-	songs
-	mapping_path
In the create mapping function we would like to cut all the excess data where we only see the mapped symbol , this can be done by splitting the data with .split and then using the (set) python built in function that returns unique elements only.
Save the resulting file in json format using .dump
5.7	PREPROCESS FUNCTION
In the preprocess function use all the previously created functions in this manner
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

parameters
-	dataset_path
 additionally save each song after the preprocessing ,but we can’t have them all be the same name ,so to mitigate the issue loop through each song and its index using the enumerate python function and save the files.









5.8 CONVERT SONGS INTO INT 
Using the mapping function created pass in the songs , load the mappings.json file to loop through the list after splitting the songs and assign each value with its mapping
 Lastly append the songs into a new list  

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

    return int_songs+
parameters
-	songs
-	


5.9 GENERATING TRAINING SEQUENCE.
The train_seq_gen is for generating training sequences for the NN as this is a supervised learning model.
def generate_training_sequences(sequence_length):

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
    inputs = keras.utils.to_categorical(inputs, 	num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

parameters
-	sequence_length
The input is a fixed length sequences and the output is the value that comes after each sequence

A question arises of how to choose the number of sequences the NN goes through to learn , we will take for an example a dataset with 100 notes and considering a sequence length of 64 that would leave us with 36 sequence of train for the NN , as we shift through the first 64 out of 100 one by one .
After getting the sequence length we need to specify the inputs and the outputs of the NN the inputs are fixed length of 64 and the outputs is the single note instance that we would like to predict
We also need to get the vocab size of our songs which we will use as the number of classes when we use one hot encoding  
We use one hot encoding because it is  easiest way to deal with categorical data when training a neural network 
for checking code functionality use a test data so that the functions don’t take too long , after making sure that the functions work properly use then are the erk data which contains 17k songs to start training the NN.











TRAINING 
Needed dependencies 
Import generate_training_sequence ( from preprocess file)
Import Sequence_length (from preprocess file)
Import tensor flow 

Similar to the preprocess file create a high level function called train that will be the focal point of the file 


5.10 BUILD MODEL 
def build_model(output_units,num_units,loss,lr):

    # create model architecture
    input = keras.layers.Input(shape = (None,output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units,activation='softmax')(x)
    model = keras.Model(input,output)

    # compile the model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(lr=LR)
                  ,metrics=['accuracy'])
    model.summary()
    return model

Create a function that accepts 4 parameters which are 
-	output units indicating the number of output layer 
-	num unit refers to the number of neurons in each layer 
-	loss indicating the loss function
-	lr indicates the learning rate for the model




for creating the architecture instead of using sequential approach we will use the functional api approach (keras api) the great thing about this approach is creating complex and flexible.

the input variable in the function has none passed as the parameter because it allows feeding in however many sequence length wanted, which results in varying durations for the melodies 

The output units parameter tells the function how many items are being fed in because we need the number of output units to be the same as the vocab size  the name output units might be misleading as we are not actually passing the size of the output layer instead we are mentioning the size of the vocab which is 38 ( basically the number of columns in the one hot encoded data)

Additionally, we added a dropout layer where we turn off 20 percent of the layers to avoid overfitting

Lastly we use output layer which is almost always a dense layer and specify the activation function for that layer which is softmax which is a function used for multiclass problems 
We now need to compile the model and for this we just use the typical optimizer adam and the metric to be the accuracy and a learning rate of 0.001 to make sure we don’t offshoot the optimal solution and lastly a model summary to see the structure of our model
Epochs is the number of times the NN gets to train on the data 
And the batch size is the amount of samples the NN sees before doing backpropagation 








TEST SET?
The reason we didn’t split our data and opted to use it all for training is for the following reason
-	This model is aimed to generate music so the aim is to get out something in the end we don’t really care about how it performs on unseen data as long as it gives results that are acceptable for us.  

5.11 TRAINING…
After building the model create a function to train the model.
Def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,lr=LR):
    #generate the train seq
    inputs , targets = generate_training_sequences(Sequence_length)
    # build the NN
    model = build_model(output_units,num_units,loss,lr)

    # train the model
    model.fit(inputs,targets,epochs =EPOCHS ,batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)

parameters  
-	output 
-	num unit 
-	loss 
-	lr 

Using generate_training_sequence we generate the inputs and outputs for training and
Then simply call build model and pass in the parameters this builds the model ,
Next step is fitting the model with model.fit and choose the number of epochs desired and the batch size which is going to be 64
Lastly save the model. 




GENERATING MELODIES
Needed dependencies
-json
-numpy
-Tensorflow
-music21
-Sequence_length and mapping_path from preprocess

After fetching the model file create a wrapper class that will go around the keras model and provide the utilities to generate melodies and manipulate it from the encodings to midi.

For the melody_generator file we initiate a class that will take in the model path and load it in from keras using the keras.models.load model function 
class MelodyGenerator:
    def __init__(self, model_path="model.h5"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(Mapping_path, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * Sequence_length

The mappings constructor is not accessible outside of the class so the mapping won’t get messed with.

Within the class we also defined a function that will be used in generating melodies.
def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
	# map seed to int
       seed = [self._mappings[symbol] for symbol in seed]
The paramerters for this class are the following 
-	Seed > it’s a piece of melody that is to be entered as the input  
-	Num_steps > the number of sequences we want to generate
-	Max_sequence length  > to limit the number of sequences that are allowed to be fed into the NN (64)
-	Temperature >  is a way to sample the songs can be from 0 to infinity but we will restrict it to 0 to 1
the seed to be entered will be split to convert it into a list and then the seed will be changed to be the start symbol + seed 
To indicate the start of a song we concatenated the forward slashes with the seed 
map the seed into an int 
also declare a variable (called melody here) to store all the seeds.
        for _ in range(num_steps):
            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

we have limited the seed to be of length 64 as to not come up with faulty results because the model has been trained with a sequence length of 64 aswell
to further clarify Seed = seed [-max_sequence_length:] 
We only want to grab the last 64 elements, for example If the list is 30 elements negative indexing in python makes it so that we only fetch from index 0 even though the negative index states otherwise and if the list is 100 values the negative indexing will fetch from the -64 index position to – 1 inclusive 
            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, 				num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the 										vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]
we use OHE and specify the number of classes we have which is found in self.mappings , additionally since the predict function in keras does not receive a 2d array we add 1 more axis using numpy which is used to specify the number of inputs (1 seed here ) to the one hot encoded seed.
            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)
We pass in the one hot seed which is 3 dimension ,  the [0] indicates that we are only interested in the probabilities of the first generated sequence as we will use that to feed it into the NN again

For the output use yet another function that accepts 2 parameters called probabilities (that we have fetched earlier) and the temperature parameter  which is widely used in GPT models where the higher than number(infinity) the more homogeneous the distribution for the probabilities is , that is to say that everything has the same chance to be predicted as outcome so just predict randomly and the lower the temperature(zero) the more strict and deterministic the model is where the highest probability just becomes a 1.

We use the soft max function with combination of the temperature so that if the temperature is high then the softmax function it makes the numbers more homogeneous and  if it is low it amplifies the bigger numbers which make them within similar range.
            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v 		== output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)
        return melody
we get back an int value from the model so we want to map it to a symbol
we specifiy [0] because we want it back as int type even though the list is of 1 element.
and check if we are at the end of a melody so we can break the loop 
lastly append and return the melody with the output symbol extracted.








HOW TEMPERATURE WORKS
example
z = [2.0, 1.0, 0.5] 
We'll calculate the softmax probabilities using different temperature values: low (0.2), moderate (1.0), and high (5.0).
Low Temperature (0.2):
softmax_temperature(z_i) = exp(z_i / 0.2) / (sum(exp(z_j / 0.2)) for j=1 to 3)
Calculating the probabilities:
softmax_temperature(z) = [exp(10.0), exp(5.0), exp(2.5)] / (exp(10.0) + exp(5.0) + exp(2.5))
                       ≈ [22026, 148.41, 12.182] / 22186.582
                       ≈ [0.991, 0.006, 0.003]
As seen the bigger number is implified much more resulting in sameness when this is implemented in the model,

Moderate Temperature (1.0):
softmax_temperature(z_i) = exp(z_i / 1.0) / (sum(exp(z_j / 1.0)) for j=1 to 3)
Calculating the probabilities:
softmax_temperature(z) = [exp(2.0), exp(1.0), exp(0.5)] / (exp(2.0) + exp(1.0) + exp(0.5))
                       ≈ [7.389, 2.718, 1.649] / 11.756
                       ≈ [0.629, 0.231, 0.140]
High Temperature (5.0):
softmax_temperature(z_i) = exp(z_i / 5.0) / (sum(exp(z_j / 5.0)) for j=1 to 3)
Calculating the probabilities:
softmax_temperature(z) = [exp(0.4), exp(0.2), exp(0.1)] / (exp(0.4) + exp(0.2) + exp(0.1))
                       ≈ [1.491, 1.221, 1.105] / 3.818
                       ≈ [0.391, 0.320, 0.290]
As seen here all the numbers are within the same range which results in more/higher randomness when getting the model results
SAMPLE WITH TEMPERATURE
The goal of this function is applying a degree of randomness or strictness within the model .
def _sample_with_temperature(self, probabilites, temperature):
    
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / 							    np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index
parameters
-	probabilities calculated by tensorflow
-	Temperature (set by the user) 
The first step is to divide by the temperature 
And then get the exponent of the prediction and divide it by the sum of the already exponentiated predictions.
We want a list of all the possible notes that will be accepted as the next output note and for that we get the range length of the probabilities and use np.random.choice 
That takes in the choices to choose from and the probability for each choice 

SAVING THE MELODY 
Define a function to save the melody for running 
def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
parameters
-	melody (which is in the form of a list)
-	step duration 
-	format 
-	file name (the name of the resulting file)
        # create a music21 stream
        stream = m21.stream.Stream()
To create a midi file first create a stream object which is responsible for saving the note and rests of a score, simply using the function m21.stream.Stream()


        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest 		    objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first 			one
                if start_symbol is not None:

      		quarter_length_duration = step_duration * 					step_counter 

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)






The algorithm  
An example to understand how the algorithm parses the melodies we take in for example the following 
 60 _   _   _   r _   62   _ 
It works by going through the symbols and checking for events, whenever it sees one for example the 60 it will still check if there is an elongation after it once we see a new event it stops and take the previous note + the elongation

If we are handling a rest we should create a rest object from m21 this is also done using m21.note.Rest and we pass in the quarter_len_duration (length of the to be appended event).

And for notes we create an object for it aswell of type note we also need to cast it into  int since it’s in string type and pass in the quarter_len_duration aswell.

And we ofcourse need to reset the step counter.

After going through the loop we encounter an issue were the loop is not functioning as imagined because we can’t log in the last note in the melody because the only way our loop logs/appends it in to the stream is if there is another event after it . 
To mitigate this issue we checked for another thing which is the index where if it reaches the last item in the melody stream we still go through with the loop ( using the OR logical statement within the if statement).

We need to calculate the quarter length too since if we aren’t dealing with the first note we need to calculate the rests and the note before it and since we are dealing with music time notation this is where we need to use the step duration and step counter to figure out the length between the current note and the note that preceeds it .

We next go through all the symbols and check whether the symbol is an underscore or not , if its an underscore we increase the step counter by 1 else   

Next we parse all the symbols by going through them and create m21 objects that we will push into the stream object.

And lastly we write the stream object into a midi file 
