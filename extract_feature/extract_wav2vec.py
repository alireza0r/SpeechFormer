
from scipy import io
import soundfile
import torch
import numpy as np
import scipy.signal as signal
# from fairseq.models.wav2vec import Wav2VecModel
import fairseq
import os
from glob import glob

def extract_wav2vec(wavfile, savefile):
    '''
    Args:
        wavfile: The absolute path to the audio file (.wav, .mp4).
        savefile: The absolute path to save the extracted feature in mat format. 
    '''
    wavs, fs = soundfile.read(wavfile)
    
    if fs != sample_rate:
        result = int((wavs.shape[0]) / fs * sample_rate)
        wavs = signal.resample(wavs, result)

    if wavs.ndim > 1:
        wavs = np.mean(wavs, axis=1)

    wavs = torch.from_numpy(np.float32(wavs)).unsqueeze(0)    # (B, S)
    
    z = wav2vec.feature_extractor(wavs)
    # z = wav2vec.vector_quantizer(z)['x']    # vq-wav2vec
    feature_wav = wav2vec.feature_aggregator(z)
    feature_wav = feature_wav.transpose(1,2).squeeze().detach().numpy()   # (t, 512)
    dict = {'wav': feature_wav}
    io.savemat(savefile, dict)
    
    print(savefile, feature_wav.shape)

if __name__ == '__main__':
    '''
    Pre-trained wav2vec model is available at https://github.com/pytorch/fairseq/blob/main/examples/wav2vec.
    Download model and save at model_path.
    '''
    model_path = './pre_trained_model/wav2vec/wav2vec_large.pt'
    sample_rate = 16000    # input should be resampled to 16kHz!
    # cp = torch.load(model_path, map_location='cpu')
    # wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
    # wav2vec.load_state_dict(cp['model'])

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    wav2vec = model[0]
    
    wav2vec.eval()

    #### use extract_wav2vec
    wavfile = "metadata/dataset/data/content/selected_audio/"
    savefile = "metadata/dataset/features/"
    os.makedirs(savefile, exist_ok=True)

    for f in glob(wavfile + '/*.wav'):
      file_name = os.path.split(f)[-1].split('_')[:-1]
      file_name = '_'.join(file_name)
      file_name = os.path.join(savefile, file_name+'.mat')

      extract_wav2vec(f, file_name)
    
    
