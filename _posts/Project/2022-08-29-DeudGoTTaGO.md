---
title: 오디오 데이터를 이용한 악보 생성
date: 2022-08-29T14:31:32.132Z
img_path: /nuyhc.github.io/assets/img
categories:
  - Project
tags:
  - Tensorflow
  - Keras
  - PyTorch
  - librosa
  - mido
---

# DeudGoTTaGo (듣고따고)
오디오 파일을 이용한 악보 전사 서비스

## 주제 선정 및 배경
- 다양한 음악을 감상뿐만 아니라 취미로 직접 연주하며 즐기고 싶은 사람들이 많음
  - 모든 음원에 대한 악보가 제공되지 않고 청음을 통한 채보가 어려움  
- 쉽게 연주할 수 있게 원하는 음악에 대해 악보를 만들어주는 서비스를 구현

---
## 1. Get Data
오디오 파일 다운로드


```python
from pytube import YouTube
from pydub import AudioSegment
import os
```


```python
wav_path = "wav_data"

def download_and_convert_wav():
    if not os.path.exists(wav_path): os.mkdir(wav_path)
    singer_song = input("가수 - 제목: ")
    yt = YouTube(input("음원 유튜브 url: "))
    yt.streams.filter(only_audio=True).first().download(output_path=wav_path, filename=singer_song+".mp3")
    AudioSegment.from_file(os.path.join(wav_path, singer_song+".mp3")).export(os.path.join(wav_path, singer_song+".wav"), format="wav")
    os.remove(os.path.join(wav_path, singer_song+".mp3"))
```

---
## 2. Classification
초기 프로젝트를 기획 할 때, 음원에서 악기별로 음원을 분리해서 사용할 생각이었는데 최종적으로 사용한 모델에서는 해당 기능을 제공하고 있어서 사용하지 않았다.
### 2-1. 가상악기로 생성한 데이터 분류
#### Library


```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
```

#### Extract Features


```python
spt = []
ins = []
duration_offset = 0
# MIDI 표준 128
for instrument, note in itertools.product(range(128), range(50)): # 데카르트 곱(cartesian product)
    y, sr = librosa.load("./GeneralMidi.wav", sr=None, offset=duration_offset, duration=2.0)
    duration_offset += 2
    # 데이터 증강을 위해 화이트 노이즈를 섞은 버전도 함께 변환
    # 옥타브당 24단계로, 총 7옥타브로 변환
    for r in (0, 1e-4, 1e-3):
        ret = librosa.cqt(y+((np.random.rand(*y.shape)-0.5)*r if r else 0),
                          sr,
                          hop_length=1024,
                          n_bins=24*7,
                          bins_per_octave=24)
        # 위상 x, 세기만 관심 있으므로 절대값을 취함
        ret = np.abs(ret)
        # 스펙토그램 저장
        spt.append(ret)
        # 악기 번호와 음 높이를 저장
        ins.append((instrument, 38+note))
# 타악기 46
for note in range(46):
    y, sr = librosa.load('./GeneralMidi.wav', sr=None, offset=duration_offset, duration=2.0)
    duration_offset += 2
    for r, s in itertools.product([0, 1e-5, 1e-4, 1e-3], range(7)):
        ret = librosa.cqt(y+((np.random.rand(*y.shape) - 0.5)*r*s if r else 0),
                          sr,
                          hop_length=1024,
                          n_bins=24*7,
                          bins_per_octave=24)
        ret = np.abs(ret)
        spt.append(ret)
        ins.append((note + 128, 0))
    
spt = np.array(spt, np.float32)
ins = np.array(ins, np.int16)

np.savez("cqt.npz", sepc=spt, instr=ins)    
```

#### Model


```python
model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(featur.shape[0], feature.shape[1], 1)))
model.add(MaxPool2D((2, 2), (2, 2), padding="valid"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2), (2, 2), padding="same"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(174, activation="softmax"))
```

### 2-2. IRMAS Dataset을 이용한 분리
#### Library


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import glob
import os
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LSTM

from sklearn.model_selection import train_test_split
```

#### Extract Features
##### CQT


```python
spt = []

for uid in tqdm(train_df.index):
    df = train_df.iloc[uid]
    path = os.path.join(train_path, df["Class"], df["FileName"])
    # load .wav
    y, sr = librosa.load(path, sr=None)
    ret = librosa.cqt(y, sr)
    ret = np.abs(ret)
    spt.append(ret)
```

##### MFCC


```python
mel = []

for uid in tqdm(train_df.index):
    df = train_df.iloc[uid]
    path = os.path.join(train_path, df["Class"], df["FileName"])
    # load .wav
    y, sr = librosa.load(path, sr=None)
    ret = librosa.feature.mfcc(y, sr)
    ret = np.abs(ret)
    mel.append(ret)
```

#### Model


```python
model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(feature.shape[0], feature.shape[1], 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2), (2, 2), padding="same"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2), (2, 2), padding="valid"))
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2), (2, 2), padding="valid"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2), (2, 2), padding="valid"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(11, activation="softmax"))
```

---
## AMT
사실상 wav 데이터를 midi 형식으로 바꿔주는 프로세스인데, 초기에 기획했던 방향은 음원을 악기별로 분리하고 분리된 악기를 해당 네트워크로 처리할 생각이었는데  
훈련에서 시간이 너무 걸리고 4일이라는 짧은 시간 동안 최적화하기 어려워서 MT3라는 모델을 사용하게 됨
## 3. Extract Features and Estimate (Model)
[# Refer:  https://paperswithcode.com/paper/residual-shuffle-exchange-networks-for-fast#code Aroksak/RSE repo.]
### 3-1. Residual Shuffle Exchange Network
#### Library


```python
import os
import warnings
import signal
import numpy as np
import torch
import torch.nn as nn
import torch_optimizer
import matplotlib.pyplot as plt
from time import time
from musicnet_dataset import MusicNet
from musicnet_model import MusicNetModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score
```

#### Setting


```python
N_EPOCHS = 100
EPOCH_SIZE = 2_000
EVAL_SIZE = 1_000
BATCH_SIZE = 4
SMOOTH = 0.1
kwargs = {'pin_memory': True}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_weights(n=2048, delta=0.1):
    xs = np.linspace(-0.25, 0.75, 2048)
    ys = 1 / np.pi * np.arctan(np.sin(2*np.pi*xs) / delta) + 0.5
    return torch.tensor(ys)
```


```python
with MusicNet('../data', train=True, download=True, window=8192, epoch_size=EPOCH_SIZE, pitch_shift=64) as train_dataset, \
     MusicNet('../data', train=False, download=False, window=8192, epoch_size=EVAL_SIZE, pitch_shift=64) as test_dataset:

    model = MusicNetModel()
    model.to(device)

    optimizer = torch_optimizer.RAdam(model.parameters(), lr=0.00125*np.sqrt(0.5))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.ones([128])*50)
    loss_fn.to(device)

    weights = get_weights(n=2048, delta=0.5).to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, **kwargs)

    for epoch in range(N_EPOCHS):
        t = tqdm(train_loader, total=EPOCH_SIZE // BATCH_SIZE, desc=f"Train. Epoch {epoch}, loss:")
        losses = []
        model.train()
        for inputs, targets in t:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(-2).to(device))
            targets = targets[:, ::4, :]
            targets = (1 - SMOOTH*2) * targets + SMOOTH
            loss = loss_fn(outputs, targets.to(device))
            loss = (loss.permute(0, 2, 1) * weights).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            t.set_description(f"Train. Epoch {epoch}, loss: {np.mean(losses[-100:]):.3f}")
        scheduler.step()

        t = tqdm(test_loader, total=EVAL_SIZE // BATCH_SIZE, desc=f"Validation. Epoch {epoch}.", leave=False)

        all_targets = []
        all_preds = []

        model.eval()
        for inputs, targets in t:
            with torch.no_grad():
                outputs = model(inputs.unsqueeze(-2).to(device))
                outputs = outputs[:, 1024, :].squeeze(1)
                targets = targets[:, 4096, :].squeeze(1)
                all_targets += list(targets.numpy())
                all_preds += list(outputs.detach().cpu().numpy())

        targets_np = np.array(all_targets)
        preds_np = np.array(all_preds)
        mask = targets_np.sum(axis=0) > 0

        print(f"Epoch {epoch}. APS: {average_precision_score(targets_np[:, mask], preds_np[:, mask]) : .2%}.")
```

### 3-2. MT3
#### Set Environment and Install Require Library
Colab에서하는게 좋음


```python
!apt-get update -qq && apt-get install -qq libfluidsynth1 build-essential libasound2-dev libjack-dev

!pip install nest-asyncio
!pip install pyfluidsynth

!pip install clu==0.0.7
!pip install clu==0.0.7

# T5X model
!git clone --branch=main https://github.com/google-research/t5x
!cd t5x; git reset --hard 2e05ad41778c25521738418de805757bf2e41e9e; cd ..
!mv t5x t5x_tmp; mv t5x_tmp/* .; rm -r t5x_tmp
!sed -i 's:jax\[tpu\]:jax:' setup.py
!python3 -m pip install -e .

# MT3
!git clone --branch=main https://github.com/magenta/mt3
!mv mt3 mt3_tmp; mv mt3_tmp/* .; rm -r mt3_tmp
!python3 -m pip install -e .

!gsutil -q -m cp -r gs://mt3/checkpoints .
!gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .
```

#### Library


```python
import functools
import os

import numpy as np
import tensorflow.compat.v2 as tf

import functools
import gin
import jax
import librosa
import note_seq
import seqio
import t5
import t5x

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

from google.colab import files

import nest_asyncio
import warnings
warnings.filterwarnings("ignore")

nest_asyncio.apply()
```


```python
SAMPLE_RATE = 16000
SF2_PATH = 'SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'

def upload_audio(sample_rate):
  data = list(files.upload().values())
  if len(data) > 1:
    print('Multiple files uploaded; using only one.')
  return note_seq.audio_io.wav_data_to_samples_librosa(
    data[0], sample_rate=sample_rate)
```


```python
class InferenceModel(object):
  """Wrapper of T5X model for music transcription."""
  def __init__(self, checkpoint_path, model_type='mt3'):
    if model_type == 'ismir2021': # 단일 피아노 모델
      num_velocity_bins = 127
      self.encoding_spec = note_sequences.NoteEncodingSpec
      self.inputs_length = 512
    elif model_type == 'mt3': # 다중 악기 모델
      num_velocity_bins = 1
      self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
      self.inputs_length = 256
    else:
      raise ValueError('unknown model_type: %s' % model_type)

    gin_files = ['/content/mt3/gin/model.gin',
                 f'/content/mt3/gin/{model_type}.gin']

    self.batch_size = 8
    self.outputs_length = 1024
    self.sequence_length = {'inputs': self.inputs_length, 
                            'targets': self.outputs_length}

    self.partitioner = t5x.partitioning.PjitPartitioner(
        num_partitions=1)

    self.spectrogram_config = spectrograms.SpectrogramConfig()
    self.codec = vocabularies.build_codec(
        vocab_config=vocabularies.VocabularyConfig(
            num_velocity_bins=num_velocity_bins))
    self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
    self.output_features = {
        'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
        'targets': seqio.Feature(vocabulary=self.vocabulary),
    }

    # Create a T5X model.
    self._parse_gin(gin_files)
    self.model = self._load_model()

    # Restore from checkpoint.
    self.restore_from_checkpoint(checkpoint_path)

  @property
  def input_shapes(self):
    return {
          'encoder_input_tokens': (self.batch_size, self.inputs_length),
          'decoder_input_tokens': (self.batch_size, self.outputs_length)
    }

  def _parse_gin(self, gin_files):
    """Parse gin files used to train the model."""
    gin_bindings = [
        'from __gin__ import dynamic_registration',
        'from mt3 import vocabularies',
        'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
        'vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS'
    ]
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          gin_files, gin_bindings, finalize_config=False)

  def _load_model(self):
    model_config = gin.get_configurable(network.T5Config)()
    module = network.Transformer(config=model_config)
    return models.ContinuousInputsEncoderDecoderModel(
        module=module,
        input_vocabulary=self.output_features['inputs'].vocabulary,
        output_vocabulary=self.output_features['targets'].vocabulary,
        optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
        input_depth=spectrograms.input_depth(self.spectrogram_config))

  def restore_from_checkpoint(self, checkpoint_path):
    train_state_initializer = t5x.utils.TrainStateInitializer(
      optimizer_def=self.model.optimizer_def,
      init_fn=self.model.get_initial_variables,
      input_shapes=self.input_shapes,
      partitioner=self.partitioner)

    restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
        path=checkpoint_path, mode='specific', dtype='float32')

    train_state_axes = train_state_initializer.train_state_axes
    self._predict_fn = self._get_predict_fn(train_state_axes)
    self._train_state = train_state_initializer.from_checkpoint_or_scratch(
        [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))

  @functools.lru_cache()
  def _get_predict_fn(self, train_state_axes):
    def partial_predict_fn(params, batch, decode_rng):
      return self.model.predict_batch_with_aux(
          params, batch, decoder_params={'decode_rng': None})
    return self.partitioner.partition(
        partial_predict_fn,
        in_axis_resources=(
            train_state_axes.params,
            t5x.partitioning.PartitionSpec('data',), None),
        out_axis_resources=t5x.partitioning.PartitionSpec('data',)
    )

  def predict_tokens(self, batch, seed=0):
    prediction, _ = self._predict_fn(
        self._train_state.params, batch, jax.random.PRNGKey(seed))
    return self.vocabulary.decode_tf(prediction).numpy()

  def __call__(self, audio):
    ds = self.audio_to_dataset(audio)
    ds = self.preprocess(ds)

    model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
        ds, task_feature_lengths=self.sequence_length)
    model_ds = model_ds.batch(self.batch_size)

    inferences = (tokens for batch in model_ds.as_numpy_iterator()
                  for tokens in self.predict_tokens(batch))

    predictions = []
    for example, tokens in zip(ds.as_numpy_iterator(), inferences):
      predictions.append(self.postprocess(tokens, example))

    result = metrics_utils.event_predictions_to_ns(
        predictions, codec=self.codec, encoding_spec=self.encoding_spec)
    return result['est_ns']

  def audio_to_dataset(self, audio):
    frames, frame_times = self._audio_to_frames(audio)
    return tf.data.Dataset.from_tensors({
        'inputs': frames,
        'input_times': frame_times,
    })

  def _audio_to_frames(self, audio):
    frame_size = self.spectrogram_config.hop_width
    padding = [0, frame_size - len(audio) % frame_size]
    audio = np.pad(audio, padding, mode='constant')
    frames = spectrograms.split_audio(audio, self.spectrogram_config)
    num_frames = len(audio) // frame_size
    times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
    return frames, times

  def preprocess(self, ds):
    pp_chain = [
        functools.partial(
            t5.data.preprocessors.split_tokens_to_inputs_length,
            sequence_length=self.sequence_length,
            output_features=self.output_features,
            feature_key='inputs',
            additional_feature_keys=['input_times']),
        preprocessors.add_dummy_targets,
        functools.partial(
            preprocessors.compute_spectrograms,
            spectrogram_config=self.spectrogram_config)
    ]
    for pp in pp_chain:
      ds = pp(ds)
    return ds

  def postprocess(self, tokens, example):
    tokens = self._trim_eos(tokens)
    start_time = example['input_times'][0]
    start_time -= start_time % (1 / self.codec.steps_per_second)
    return {
        'est_tokens': tokens,
        'start_time': start_time,
        'raw_inputs': []
    }

  @staticmethod
  def _trim_eos(tokens):
    tokens = np.array(tokens, np.int32)
    if vocabularies.DECODED_EOS_ID in tokens:
      tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
    return tokens
```


```python
MODEL = "mt3" #param["ismir2021", "mt3"]

checkpoint_path = f'/content/checkpoints/{MODEL}/'

inference_model = InferenceModel(checkpoint_path, MODEL)
```


```python
# CoLab
audio = upload_audio(sample_rate=SAMPLE_RATE)

note_seq.notebook_utils.colab_play(audio, sample_rate=SAMPLE_RATE)

# Local
# audio = os.path.join([wav_path, singer_song+".wav"])
```


```python
est_ns = inference_model(audio)

note_seq.play_sequence(est_ns, synth=note_seq.fluidsynth, 
                       sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
note_seq.plot_sequence(est_ns)
```


```python
note_seq.sequence_proto_to_midi_file(est_ns, '/tmp/transcribed.mid')
files.download('/tmp/transcribed.mid')
```

## 4. Scoring
기존에 미디 형식으로 되어있는 파일들을 채보해주는 방식을 이용했음  
최소 기준 박을 16음표로 설정하니 3연음같이 홀수 박의 채보는 잘 안되는거 같음
### Library


```python
from mido import MidiFile, MidiTrack, Message
import mido
import IPython
import matplotlib.pyplot as plt
import librosa.display
from IPython import *
from music21 import *
from music21 import converter, instrument, note, chord, stream, midi
```

### Set Environment


```python
!add-apt-repository ppa:mscore-ubuntu/mscore-stable -y
!apt-get update
!apt-get install musescore

!apt-get install xvfb

!sh -e /etc/init.d/x11-common start

import os
os.putenv('DISPLAY', ':99.0')

!start-stop-daemon --start --pidfile /var/run/xvfb.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset

us = environment.UserSettings()
us['musescoreDirectPNGPath'] = '/usr/bin/mscore'
us['directoryScratch'] = '/tmp'
```

### Setting


```python
MELODY_NOTE_OFF = 128
MELODY_NO_EVENT = 129
```


```python
def streamToNoteArray(stream):
    # Part1. extract from stream
    total_length = int(np.round(stream.flat.highestTime/0.25))
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append([np.round(element.offset/0.25), np.round(element.quarterLength/0.25), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset/0.25), np.round(element.quarterLength/0.25), element.sortAscending().pitches[-1].midi])
    np_stream_list = np.array(stream_list, dtype=int)
    df = pd.DataFrame({'pos': np_stream_list.T[0], 'dur': np_stream_list.T[1], 'pitch': np_stream_list.T[2]})
    df = df.sort_values(['pos','pitch'], ascending=[True, False])
    df = df.drop_duplicates(subset=['pos'])
    # part 2, convert into a sequence of note events
    output = np.zeros(total_length+1, dtype=np.int16) + np.int16(MELODY_NO_EVENT) 
    # Fill in the output list
    for i in range(total_length):
        if not df[df["pos"]==i].empty:
            n = df[df["pos"]==i].iloc[0]
            output[i] = n["pitch"]
            output[i+n["dur"]] = MELODY_NOTE_OFF
    return output
def noteArrayToDataFrame(note_array):
    df = pd.DataFrame({"code": note_array})
    df["offset"] = df.index
    df["duration"] = df.index
    df = df[df["code"]!=MELODY_NO_EVENT]
    df["duration"] = df["duration"].diff(-1)*-1*0.25
    df = df.fillna(0.25)
    return df[["code", "duration"]]
def noteArrayToStream(note_array):
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row["code"] == MELODY_NO_EVENT:
            new_note = note.Rest() 
        elif row["code"] == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row["duration"]
        melody_stream.append(new_note)
    return melody_stream
```


```python
mid_stream = converter.parse("./transcribed.mid")
```

#### 여러 악기들에 대한 악보 출력


```python
mid_stream.show()
```

#### 곡의 멜로디 악보 출력


```python
mid_stream_rnn = streamToNoteArray(mid_stream)
noteArrayToStream(mid_stream).show()
```

## 마무리
너무 쉽게 생각하고 덤볐던 주제였던거 같다.  
애초에 AMT(Auto Music Transciption)이라는 분야가 따로 있고, 오디오 데이터 분석을 위해 알아야한다는 사실이 많다는 것을 프로젝트를 시작하고 알았다.  
4일이라는 데이터톤 기간에 관련된 모든 내용들을 공부하고 활용하기에는 다소 큰 부담이었다.  
무엇보다 실험적인 분야라 그런지 모델 생성이나 관련 논문들을 봐도 Keras처럼 고수준으로 구현하지 못해 잘못하는 PyTorch로 꾸역꾸역 구현하려고했다..  
토치에 좀 더 익숙했다면 시간을 좀 더 많이 아낄 수 있었을지도..?  
그래도 새로운 도메인에 도전하면서 배우거나 알게된 사실들은 가장 많았던거같다 확실히 토치를 잘 활용하는게 이쪽분야에서 유리한 점이 많을꺼같다.

