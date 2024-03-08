
import numpy as np
from scipy import io
import librosa

def extract_spec(wavfile, savefile, nfft, hop):
    '''Calculate magnitude spectrogram from raw audio file by librosa.
    
    Args:
        wavfile: The absolute path to the audio file (.wav, .mp4).
        savefile: The absolute path to save the extracted feature in mat format. 
    '''
    y, _sr = librosa.load(wavfile, sr=None)
    M = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop, window='hamming'))

    data = {'spec': M}
    io.savemat(savefile, data)

    print(savefile, M.shape)

if __name__ == '__main__':
    sr = 16000     # sample rate  16000 (iemocap, daiz_woc) or 44100 (meld, pitt)
    frame = 0.02   # 20ms
    nfft = int(sr*frame)
    hop = nfft//2

    #### use extract_spec
    # wavfile = xxx
    # savefile = xxx
    # extract_spec(wav_file, savefile, nfft, hop)

    # wavfile = "./SpeechFormer/metadata/dataset/data/"
    savefile = "./SpeechFormer/metadata/dataset/wav_spec_20ms_mat/"
    if not os.path.exists(savefile):
      os.makedirs(savefile)
    # wavfile = "./SpeechFormer/metadata/dataset/data/content/selected_audio/300_s0_AUDIO.wav"
    wavepath = "./SpeechFormer/metadata/dataset/data/content/selected_audio/"
    
    for f in glob(wavepath + '/*.wav'):
      file_name = os.path.split(f)[-1].split('.wav')[0]
      file_name = os.path.join(savefile, file_name)
      # print(file_name)
      extract_spec(f, file_name, nfft, hop)
    
