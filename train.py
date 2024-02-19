# from pre_process import generate_training_sequences
# from pre_process import Sequence_length
# from tensorflow import keras 
# OUTPUT_UNITS = 38
# NUM_UNITS = [256]
# LOSS = "sparse_categorical_crossentropy"
# LR = 0.001
# EPOCHS = 50
# BATCH_SIZE = 64
# SAVE_MODEL_PATH = 'model.h5'


# def build_model(output_units,num_units,loss,lr):

#     # create model architecture
#     input = keras.layers.Input(shape = (None,output_units))
#     x = keras.layers.LSTM(num_units[0])(input)
#     x = keras.layers.LSTM()
#     x = keras.layers.Dropout(0.2)(x)

#     output = keras.layers.Dense(output_units,activation='softmax')(x)
#     model = keras.Model(input,output)

#     # compile the model
#     model.compile(loss=loss,
#                   optimizer=keras.optimizers.Adam(lr=LR)
#                   ,metrics=['accuracy'])
#     model.summary()
#     return model

# def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,lr=LR):
#     #generate the train seq
#     inputs , targets = generate_training_sequences(Sequence_length)
#     # build the NN
#     model = build_model(output_units,num_units,loss,lr)

#     # train the model
#     model.fit(inputs,targets,epochs =EPOCHS ,batch_size=BATCH_SIZE)

#     # save the model
#     model.save(SAVE_MODEL_PATH)


# if __name__ == '__main__':
#     train()






from pre_process import generate_training_sequences
from pre_process import Sequence_length
from tensorflow import keras 

OUTPUT_UNITS = 41
NUM_UNITS = [256, 128, 64]  # Adding three LSTM layers with 256, 128, and 64 units
LOSS = "sparse_categorical_crossentropy"
LR = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = 'model.h5'

def build_model(output_units, num_units, loss, lr):
    # create model architecture
    input = keras.layers.Input(shape=(None, output_units))
    
    # Adding the first LSTM layer with 256 units
    x = keras.layers.LSTM(num_units[0], return_sequences=True)(input)
    x = keras.layers.Dropout(0.2)(x)

    # Adding the second LSTM layer with 128 units
    x = keras.layers.LSTM(num_units[1], return_sequences=True)(x)
    x = keras.layers.Dropout(0.2)(x)

    # Adding the third LSTM layer with 64 units
    x = keras.layers.LSTM(num_units[2])(x)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation='softmax')(x)
    model = keras.Model(input, output)

    # compile the model
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, lr=LR):
    # generate the train seq
    inputs, targets = generate_training_sequences(Sequence_length)
    # build the NN
    model = build_model(output_units, num_units, loss, lr)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == '__main__':
    train()
