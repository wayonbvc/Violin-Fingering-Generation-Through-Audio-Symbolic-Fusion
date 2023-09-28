## Audio-Symbolic Violin Fingering Generation Model
The model contains three modules: an audio embedder, a symbolic embedder, and a classifier. The classifier consists of Bi-directional Long Short-Term Memory Network (BLSTM), with a softmax layer to predict the combination of strings, positions, and fingers.

## Download Audio Features 
Audio features could be download via these links:

TNUA dataset: https://drive.google.com/drive/folders/18LOqCe0zvscR31cGV5RqIu-OhVUmi2Hy?usp=sharing

YTVF dataset: https://drive.google.com/drive/folders/1cdwPsMCURCcAFn6bdjaJctwkGYAIAPu-?usp=sharing

The audio features should be put in the folder **TNUA_datset/audio** and **YTVF_dataset/audio**

## Implementation

**train.py** can be implememted directly after downloading the above audio features.

**generate_audio_feature.py** provides an example to extract audio features from an audio recording.

## Requirements
 * python >= 3.11.5
 * tensorflow >= 2.13.0 
 * numpy>= 1.24.3
 * scikit-learn >= 1.3.0
 * MIDIUtil >= 1.2.1
 * midi2audio >= 0.1.1
 * fluidsynth >= 2.2.5
 * librosa >= 0.10.1
