import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioTextDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        datafiles=[''], 
        sampling_rate=32000, 
        max_clip_len=5,
        suppress_warnings=False,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                data_json = json.load(fp)['data']
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate
        self.suppress_warnings = suppress_warnings
        self.dropped_files_count = 0

    def __len__(self):
        return len(self.all_data_json)

    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        # random crop
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform

    def _read_audio(self, index, original_path=None):
        # Get the intended path *before* the try block
        intended_audio_path = self.all_data_json[index]['wav']
        if original_path is None:
            original_path = intended_audio_path # Store the path of the first attempt

        try:
            # Use intended_audio_path for loading attempt
            audio_path = intended_audio_path
            audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)
            text = self.all_data_json[index]['caption']

            # drop short utterance
            if audio_data.size(1) < self.sampling_rate * 0.5:
                raise Exception(f'{audio_path} is too short, drop it ...')

            # Return original_path along with other data
            return text, audio_data, audio_rate, original_path

        except Exception as e:
            self.dropped_files_count += 1
            if not self.suppress_warnings:
                # Use original_path for error message context
                print(f'Error: {e} occurred when loading {original_path} (intended item index {index}). Skipping this item.')
            # Return None to signal failure
            return None

    def __getitem__(self, index):
        # create a audio tensor
        # Capture the result from _read_audio
        read_result = self._read_audio(index)

        # Check if loading failed
        if read_result is None:
            return None # Propagate the failure signal

        # Unpack if successful
        text, audio_data, audio_rate, original_path = read_result

        audio_len = audio_data.shape[1] / audio_rate
        # convert stero to single channel
        if audio_data.shape[0] > 1:
            # audio_data: [samples]
            audio_data = (audio_data[0] + audio_data[1]) / 2
        else:
            audio_data = audio_data.squeeze(0)
        
        # resample audio clip
        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)
        
        audio_data = audio_data.unsqueeze(0)
        
        audio_data = self._cut_or_randomcrop(audio_data)            

        data_dict = {
            'text': text,
            'waveform': audio_data,
            'modality': 'audio_text',
            'original_audiopath': original_path
        }

        return data_dict

    def get_dropped_count(self):
        """Returns the count of files that failed to load and were replaced."""
        return self.dropped_files_count
