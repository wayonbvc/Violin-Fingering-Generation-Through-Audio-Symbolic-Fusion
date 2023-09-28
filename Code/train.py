# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:14:40 2023

@author: WayonLin
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score

seq_len = 32
audio_dim = 2206

#%%
# Load symbolic data

TNUA_song_name_list = ["bach1","bach2","beeth1","beeth2_1","beeth2_2","elgar","flower","mend1","mend2","mend3","mozart1","mozart2_1","mozart2_2","wind"]
YTVF_song_name_list = ['Ave Maria', 'Beautiful Rosmarin', 'Beethoven Minuet', 'Boccherini Minuet', 'Caprice 24', 'Czardas', 'Estrellita',
 'Four seasons winter mvt1', 'Gluck Melodie', 'Humoresque', 'La Capricieuse', 'La Cinquantaine', 'La plus que lente', 'Meditation de Thais',
 'Mozart Minuet', 'Mozart Violin Sonata K304 Menuetto', 'Nocturne No20', 'On wings of song', 'Polonaise Brilliante', 'Praeludium and Allegro',
 'Rondino on a Theme by Beethoven', 'Schubert Serenade', 'Sicilienne', 'Song of India', 'Songs My Mother Taught Me', 'Tchaikovsky Melodie',
 'Tchaikovsky Violin Concerto mvt2', 'The Swan', 'Vitali Chaconne', 'Vocalise']

# There are 14 songs performed by 10 violinists in TNUA dataset
TNUA_song_length_lists = []
for i in range(10):
    TNUA_song_length_lists.append(np.load(f"TNUA_dataset/symbolic/vio{i+1}_length.npy"))
YTVF_song_length_list = np.load("YTVF_dataset/symbolic/length.npy")

# We use TNUA dataset as training data. The number of notes are slightly different between 10 players.
# We store all traning data with the shape of (number of players, maximum number of notes, number of features and labels).
# In the npy files, columns 1-3 are pitch, onset, duration, and columns 4-6 are string, position, finger.  
max_length = np.max(np.sum(TNUA_song_length_lists,axis=1))
train_data= np.zeros((10,max_length,6))
for j in range(10):
    index = 0
    for i in range(14):
        train_data[j][index:index+TNUA_song_length_lists[j][i]] = np.load(f"TNUA_dataset/symbolic/vio{j+1}_{TNUA_song_name_list[i]}.npy")[:,:6]
        index += TNUA_song_length_lists[j][i]

# We use YTVF dataset as testing data. There are 30 songs in YTVF dataset.
test_data= np.zeros((np.sum(YTVF_song_length_list),6))
index = 0
for i in range(30):
    test_data[index:index+YTVF_song_length_list[i]] = np.load(f"YTVF_dataset/symbolic/{YTVF_song_name_list[i]}.npy")[:,:6]
    index += YTVF_song_length_list[i]
    
#%%

# Preprocessing

# Pitch, onset, and duration are divided into 46, 81, and 65 class
# Map each feature value to the corresponding class (by finding the nearest value)

def find_nearest(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return val

pitch_category = np.linspace(55, 100, 46)
onset_category = np.linspace(-1, 4, 81)
duration_category = np.linspace(0, 4, 65)

for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        train_data[i][j][0] = find_nearest(pitch_category,train_data[i][j][0])
        train_data[i][j][1] = find_nearest(onset_category,train_data[i][j][1])
        train_data[i][j][2] = find_nearest(duration_category,train_data[i][j][2])
for i in range(test_data.shape[0]):
    test_data[i][0] = find_nearest(pitch_category,test_data[i][0])
    test_data[i][1] = find_nearest(onset_category,test_data[i][1])
    test_data[i][2] = find_nearest(duration_category,test_data[i][2])

# Map feature value to class. 0 is left for padding token.        
feature_dict_list = [{} for i in range(3)]
for i in range(3):
    feature_list = [pitch_category,onset_category,duration_category]
    for j,num in enumerate(feature_list[i]):
        feature_dict_list[i][num]=j+1
        
# Map (string, position, finger) into a number, and map the number into 240 class. 0, 1 are left for padding tokens.
spf_dict={}
spf_reverse_dict={}
count=2
for i in range(1,5):
    for j in range(1,13):
        for k in range(1,6):
            idx = i*1000+j*10+k*1
            spf_dict[idx]=count
            spf_reverse_dict[count]=idx
            count+=1
feature_dict_list.append(spf_dict)

# From class to string, position, finger
def get_spf(x):
    idx = spf_reverse_dict[x]
    s_list= ["G","D","A","E"]
    s = s_list[int(idx/1000)-1]
    p = int((idx%1000)/10)
    f = idx%10 - 1
    return [s,p,f]

feature_dict_list[3][0]=0

# Features are converted to categorical inputs
categorical_train_data = np.zeros((train_data.shape[0],train_data.shape[1],4))
for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        for k in range(3):
          categorical_train_data[i][j][k] = feature_dict_list[k][train_data[i][j][k]]
        idx = 1000*train_data[i][j][3]+10*train_data[i][j][4]+train_data[i][j][5]
        categorical_train_data[i][j][3] = feature_dict_list[3][idx]
        
categorical_test_data = np.zeros((test_data.shape[0],4))
for j in range(test_data.shape[0]):
    for k in range(3):
        categorical_test_data[j][k] = feature_dict_list[k][test_data[j][k]]
    if test_data[j][4]>=12:
        test_data[j][4]=12
    id = 1000*max(test_data[j][3],1)+10*test_data[j][4]+test_data[j][5]
    categorical_test_data[j][3] = feature_dict_list[3][id]
    
#%%
# Make dataset
# Make the symbolic data into sequences
def make_TNUA_symbolic_datasets(input_id_list,gap_len,player_id):
    X = []
    num_of_songs = len(input_id_list)
    song_num = 0
    input_sequence = []
    while(song_num<num_of_songs):
        song_id = input_id_list[song_num]
        current_id = int(sum(TNUA_song_length_lists[player_id][:song_id]))
        end_id = sum(TNUA_song_length_lists[player_id][:song_id+1])
        while (current_id < end_id):
            for j in range(seq_len):
                if (current_id+j) == end_id:
                    pad_length = seq_len - j
                    for k in range(pad_length):
                        input_sequence.append([0,0,0,0])
                    break
                else:
                    input_sequence.append(categorical_train_data[player_id][current_id+j])
            X.append(input_sequence)
            input_sequence=[]
            current_id+=int(gap_len)
        song_num+=1
    X = np.array(X)
    return X

def make_YTVF_symbolic_datasets(input_id_list,gap_len):
    X = []
    num_of_songs = len(input_id_list)
    song_num = 0
    input_sequence = []
    while(song_num<num_of_songs):
        song_id = input_id_list[song_num]
        current_id = int(sum(YTVF_song_length_list[:song_id]))
        end_id = sum(YTVF_song_length_list[:song_id+1])
        while (current_id < end_id):
            for j in range(seq_len):
                if (current_id+j) == end_id:
                    pad_length = seq_len - j
                    for k in range(pad_length):
                        input_sequence.append([0,0,0,0])
                    break
                else:
                    input_sequence.append(categorical_test_data[current_id+j])
            X.append(input_sequence)
            input_sequence=[]
            current_id+=int(gap_len)
        song_num+=1
    X =np.array(X)
    return X

# Make the audio data into sequences
def make_TNUA_audio_datasets(input_id_list, gap_len, player_id):
    X = []
    num_of_songs = len(input_id_list)
    song_num = 0
    input_sequence = []
    audio_features = np.load(f'TNUA_dataset/audio/all_audio_feature_player{player_id+1}.npy')

    while(song_num<num_of_songs):
        song_id = input_id_list[song_num]
        current_id = int(sum(TNUA_song_length_lists[player_id][:song_id]))
        end_id = sum(TNUA_song_length_lists[player_id][:song_id+1])
        while (current_id<end_id):
            for j in range(seq_len):
                if (current_id+j) == end_id:
                    pad_length = seq_len - j
                    for k in range(pad_length):
                        input_sequence.append(np.zeros(audio_dim))
                    break
                else:
                    input_sequence.append(audio_features[current_id+j])
            X.append(input_sequence)
            input_sequence=[]
            current_id+=int(gap_len)
        song_num+=1
    X =np.array(X)
    return X

def make_YTVF_audio_datasets(input_id_list,gap_len):
    X = []
    num_of_songs = len(input_id_list)
    song_num = 0
    input_sequence = []
    audio_test_features = np.load('YTVF_dataset/audio/all_audio_feature.npy')

    while(song_num<num_of_songs):
        song_id = input_id_list[song_num]
        current_id = int(sum(YTVF_song_length_list[:song_id]))
        end_id = sum(YTVF_song_length_list[:song_id+1])
        while (current_id<end_id):
            for j in range(seq_len):
                if (current_id+j) == end_id:
                    pad_length = seq_len - j
                    for k in range(pad_length):
                        input_sequence.append(np.zeros(audio_dim))
                    break
                else:
                    input_sequence.append(audio_test_features[current_id+j])
            X.append(input_sequence)
            input_sequence=[]
            current_id+=int(gap_len)
        song_num+=1
    X =np.array(X)
    return X
#%%
# Split data

train_id_list = list(range(14))
valid_id_list = [10,11,12,13]
test_id_list = list(range(30))
train_id_list = list(set(train_id_list) - set(valid_id_list))

train_player_list = [0,1,2,3]
valid_player_list = [4,5,6]

X_train = np.array([])
X_valid = np.array([])
X_test = np.array([])
X_train_audio = np.array([])
X_valid_audio = np.array([])
X_test_audio = np.array([])

for player_id in train_player_list:
    X_train_tmp = make_TNUA_symbolic_datasets(train_id_list,seq_len/2,player_id)
    X_train_audio_tmp = make_TNUA_audio_datasets(train_id_list,seq_len/2,player_id)

    X_train = np.concatenate((X_train,X_train_tmp)) if len(X_train)>0 else X_train_tmp
    X_train_audio = np.concatenate((X_train_audio,X_train_audio_tmp)) if len(X_train_audio)>0 else X_train_audio_tmp

for player_id in valid_player_list:
    X_valid_tmp = make_TNUA_symbolic_datasets(valid_id_list,seq_len,player_id)
    X_valid_audio_tmp = make_TNUA_audio_datasets(valid_id_list,seq_len,player_id)
    X_valid = np.concatenate((X_valid,X_valid_tmp)) if len(X_valid)>0 else X_valid_tmp
    X_valid_audio = np.concatenate((X_valid_audio,X_valid_audio_tmp)) if len(X_valid_audio)>0 else X_valid_audio_tmp

X_test = make_YTVF_symbolic_datasets(test_id_list,seq_len)
X_test_audio = make_YTVF_audio_datasets(test_id_list,seq_len)
#%%
# Build and train the model

def train_audio_symbolic_model(seed,initial_learning_rate,epochs,batch_size):
    # Set seed value to reproduce the result
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    
    l1l2 = regularizers.l1_l2(l1=1e-5, l2=1e-4)
    l2 = regularizers.l2(1e-4)
    embedding_size_pitch = 32
    embedding_size_start = 32
    embedding_size_duration = 32
    n_dim1 = 64
    n_dim2 = 64
    n_pitch = len(feature_list[0])+1 
    n_start = len(feature_list[1])+1 
    n_duration = len(feature_list[2])+1 
    n_spf = 242
    
    # Inputs
    in_pitch = keras.Input(shape=(seq_len,), name='pitch')
    in_start = keras.Input(shape=(seq_len,), name='start')
    in_duration = keras.Input(shape=(seq_len,), name='duration')
    in_audio = keras.Input(shape=(seq_len,audio_dim), name='audio')

    # Symbolic Embedder
    x1 = layers.Embedding(n_pitch, embedding_size_pitch, mask_zero=False, embeddings_regularizer=l1l2)(in_pitch)
    x1 = layers.Masking()(x1)
    x2 = layers.Embedding(n_start , embedding_size_start, mask_zero=False, embeddings_regularizer=l1l2)(in_start)
    x3 = layers.Embedding(n_duration, embedding_size_duration, mask_zero=False, embeddings_regularizer=l1l2)(in_duration)
    x = layers.Concatenate()([x1,x2,x3])
    x = layers.Dense(n_dim1, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(x)
    x = layers.PReLU()(x)
    out_emb = layers.LayerNormalization()(x)
    emb_in = [in_pitch, in_start, in_duration]
    emb_out = [out_emb]
    embedder = keras.Model(emb_in,emb_out, name='symbolic_embedder')
    
    # Audio Embedder
    x4 = layers.Dense(64, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(in_audio)
    x_audio = layers.Dense(n_dim1, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(x4)
    x_audio = layers.PReLU()(x_audio)
    out_emb_audio = layers.LayerNormalization()(x_audio)
    emb_in_audio = [in_audio]
    emb_out_audio = [out_emb_audio]
    embedder_audio = keras.Model(emb_in_audio,emb_out_audio, name='audio_embedder')


    #Classifier
    
    in_cla = keras.Input(shape=(seq_len, n_dim2*2 ), name='classifier_input')
    rnn_units = 128
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2, recurrent_regularizer=l1l2))(in_cla)
    x = layers.Dense(n_spf)(x)
    out_cla = layers.Activation('softmax')(x)
    cla_in = [in_cla]
    cla_out = [out_cla]
    classifier = keras.Model(cla_in,cla_out, name='classifier')

    # Build the model
    embedded_output = embedder(emb_in)
    embedded_audio_output = embedder_audio(emb_in_audio)
    embedded_concat = layers.Concatenate()([embedded_output,embedded_audio_output])
    classified_output = classifier(embedded_concat)
    audio_symbolic_model = keras.Model([emb_in,emb_in_audio], classified_output)
    
    # Train model
    monitor = 'val_accuracy'
    patience = 10 
    restore_best_weights = True  
    cal_earlystopping = keras.callbacks.EarlyStopping(
    monitor=monitor,
    patience=patience,
    verbose=2,
    restore_best_weights=restore_best_weights)

    initial_learning_rate = initial_learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=350,
        decay_rate=0.96,
        staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.001)

    audio_symbolic_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=0), optimizer=opt, metrics=['accuracy'])
    epochs = epochs
    batch_size = batch_size

    history = audio_symbolic_model.fit(
        x = [[X_train[:,:,0],X_train[:,:,1],X_train[:,:,2]],[X_train_audio]],
        y = X_train[:,:,3],
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=([[X_valid[:,:,0],X_valid[:,:,1],X_valid[:,:,2]],[X_valid_audio]], X_valid[:,:,3]),
        callbacks=[cal_earlystopping]
    )
    return audio_symbolic_model
#%%
# Train the model

seed = 3
initial_learning_rate = 0.005
epochs = 50
batch_size = 32
audio_symbolic_model = train_audio_symbolic_model (seed, initial_learning_rate, epochs, batch_size)

#%%
#Evaluation

prediction = audio_symbolic_model.predict(x=[[X_test[:,:,0],X_test[:,:,1],X_test[:,:,2]],[X_test_audio]])
prediction[:,:,:2] = 0
arg_prediction = np.argsort(-prediction,axis=-1)

string_MRR = 0
position_MRR = 0
finger_MRR = 0
total_MRR = 0

string_acc = 0
position_acc = 0
finger_acc = 0
total_acc = 0

count=0
string_predictions = []
string_labels = []
position_predictions = []
position_labels = []
finger_predictions = []
finger_labels = []
all_prediction = []
all_label = []

for i in range(arg_prediction.shape[0]):
    for j in range(arg_prediction.shape[1]):
        if X_test[i,j,3]!=0:
            count+=1
            all_prediction.append(arg_prediction[i][j][0])
            all_label.append(X_test[i,j,3])
            
            string_arg_prediction = []
            for l in range(242):
                string_prediction = spf_reverse_dict[arg_prediction[i][j][l]]//1000-1
                if string_prediction not in string_arg_prediction:
                    string_arg_prediction.append(string_prediction)
                if len(string_arg_prediction)==4:
                    break

            position_arg_prediction = []
            for l in range(242):
                if arg_prediction[i][j][l]<2:
                    continue
                position_prediction = int(spf_reverse_dict[arg_prediction[i][j][l]]%1000/10)
                if position_prediction not in position_arg_prediction:
                    position_arg_prediction.append(position_prediction)
                if len(position_arg_prediction)==12:
                    break

            finger_arg_prediction = []
            for l in range(242):
                finger_prediction = spf_reverse_dict[arg_prediction[i][j][l]]%10-1
                if finger_prediction not in finger_arg_prediction:
                    finger_arg_prediction.append(finger_prediction)
                if len(finger_arg_prediction)==5:
                    break

            string_predictions.append(string_arg_prediction[0])
            position_predictions.append(position_arg_prediction[0])
            finger_predictions.append(finger_arg_prediction[0])
            
            string_label = spf_reverse_dict[X_test[i,j,3]]//1000-1
            position_label = int(spf_reverse_dict[X_test[i,j,3]]%1000/10)
            finger_label = spf_reverse_dict[X_test[i,j,3]]%10-1            
            string_labels.append(string_label)            
            position_labels.append(position_label)            
            finger_labels.append(finger_label)

            for k in range(4):
                if string_label == string_arg_prediction[k]:
                    if k==0:
                        string_acc+=1
                    rank = k+1
                    string_MRR += 1/rank
                    break
            for k in range(12):
                if position_label == position_arg_prediction[k]:
                    if k==0:
                        position_acc+=1
                    rank = k+1
                    position_MRR += 1/rank
                    break
            for k in range(5):
                if finger_label == finger_arg_prediction[k]:
                    if k==0:
                        finger_acc+=1
                    rank = k+1
                    finger_MRR += 1/rank
                    break

            for k in range(240):
                if X_test[i,j,3]==arg_prediction[i][j][k]:
                    if k==0:
                        total_acc+=1
                    rank = k+1
                    total_MRR += 1/rank
                    break

total_MRR = total_MRR/ count
string_MRR = string_MRR/ count
position_MRR = position_MRR/ count
finger_MRR = finger_MRR/ count

total_acc = total_acc/ count
string_acc = string_acc/ count
position_acc = position_acc/ count
finger_acc = finger_acc/ count

total_f1 = f1_score(all_label, all_prediction, average='macro')
string_f1 = f1_score(string_labels, string_predictions, average='macro')
position_f1 = f1_score(position_labels, position_predictions, average='macro')
finger_f1 = f1_score(finger_labels, finger_predictions, average='macro')

print(total_MRR, string_MRR, position_MRR, finger_MRR)
print(total_acc, string_acc, position_acc, finger_acc)
print(total_f1, string_f1, position_f1, finger_f1)
