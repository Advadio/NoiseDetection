import librosa
import numpy as np
import tensorflow as tf

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

def main():
    file = input("please give the classifier the audio's file name")
    features = features_extractor(file)
    model = None
    result = model.predict(features)
    if result == 1:
        print("noise")
    else:
        print("not noise")

if __name__ == '__main__':
    main()