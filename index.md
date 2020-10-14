## Welcome to the Data Revolution

Olá, meu nome é Rafael Guerra <[Linkedin](https://www.linkedin.com/in/rafaelestevamguerra/)>. Sou estudante de Linguística na Universidade de São Paulo (4/8), realizei cursos de Educação Executiva em Ciência de Dados e Big Data no Insper, Singularity University e na FGV EAESP. Participei ativamente e fui selecionado para 4ª Escola Avançada em Big Data Analysis promovido pelo Instituto de Ciências Matemáticas e Computação da USP São Carlos.

Sou apaixonado por dados, inteligência artificial e, principalmente, por Big Data! 

Você vai encontrar nesse repositório muitas informações que venho captando ao longo do caminho, principalmente na construção de _web-apps_ em Python usando bibliotecas como streamlit e pandas. Ademais, também trabalho com Processamento de Linguagem Natural.

Periodicamente atualizo este repositório com códigos para PLN, chatbots e, também, para deepfake e clonagem de voz. Todas licenças são MIT – sinta-se livre para utilizar estes códigos como quiser. Também não é necessário dar créditos ao criador.

Você pode me encontrar em reguerra@datadeluge.com.br

### Viz2Biz

Projeto autoral de web-app automatizado para visualização de dados com foco em inteligência de negócios. Projeto com a utilização de Inteligência Artificial e linkado a uma página web, o Viz2Biz é um sistema ready-to-use de alto nível de segurança.

```
CODE
```

### DelugerStream

Recentemente comecei a trabalhar em um web-app autoral em Python programado com a biblioteca streamlit e outras bibliotecas de análise e visualização de dados. Também estou trabalhando em cima de adicionar camadas de inteligência artificial a este programa para que as análises sejam mais robustas e sofisticadas. Abaixo você encontra o código completo.

CÓDIGO JÁ COM COMENTÁRIOS

```markdown
CODE
```

### BoneBreaker

Este é um projeto que estou trabalhando já algum tempo para aprimorar um modelo de _deepfake_ e clonagem de voz para chatbots e assistentes pessoais, meus sets são treinados utilizando GPUs comuns e programados em Python – este projeto utiliza o código-base desenhado por Edresson Casanova.

## CÓDIGO ORIGINAL

```markdown
CÓDIGO-BASE EM PYTHON BY EDRESSON CASANOVA

import os
# clone Neural Vocoder WaveRNN
!git clone https://github.com/erogol/WaveRNN.git
os.chdir('WaveRNN')
!git checkout 12c8744
!pip install -r requirements.txt

# Download WaveRNN checkpoints
!rm saver-wavernn.zip
!wget https://www.dropbox.com/s/4a60kt3detcw3r6/checkpoint-wavernn-finetunnig-tts-portuguese-corpus-560900.zip?dl=0 -O saver-wavernn.zip
!unzip saver-wavernn.zip 

os.chdir('..')

# Clone TTS repository
!git clone https://github.com/Edresson/TTS -b TTS-Portuguese

# Install TTS repository

# !python -m pip install -r TTS/requirements.txt
! python -m pip install  numpy==1.14.3 lws torch>=0.4.1 librosa==0.5.1 Unidecode==0.4.20 tensorboard tensorboardX Pillow flask scipy==0.19.0 lws tqdm phonemizer
#matplotlib==2.0.2
! apt-get install espeak

%load_ext autoreload
%autoreload 2
import os
import sys
import io
import torch 
import time
import numpy as np
from collections import OrderedDict

TTS_PATH = "../content/TTS"
WAVERNN_PATH ="../content/WaveRNN"
# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally
sys.path.append(WAVERNN_PATH) # set this if TTS is not installed globally

import pylab as plt
%pylab inline

from matplotlib import rcParams 
rcParams["figure.figsize"] = (16,5)
sys.path.append('')

import librosa
import librosa.display

from TTS.models.tacotron import Tacotron 
from TTS.layers import *
from TTS.utils.data import *
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config
from TTS.utils.text import text_to_sequence, phoneme_to_sequence
from TTS.utils.text.symbols import symbols, phonemes

import IPython
from IPython.display import Audio
from TTS.utils import *


from TTS.utils.visual import visualize
from matplotlib import pylab as plt

# TTS Weights
!wget -c -q --show-progress -O ./TTS-TL-saver.zip https://www.dropbox.com/s/91etfwt4tvzjqyz/TTS-checkpoint-phonemizer-wavernn-381000.zip?dl=0
!ls
!rm config.json
!unzip TTS-TL-saver.zip
! mv checkpoint_381000.pth.tar checkpoint.pth.tar

# Utils definitions
from TTS.utils.synthesis import visualize 
 
def plot_alignment_with_text(alignment,text, info=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(
        alignment.T, aspect='auto', origin='lower', interpolation=None)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.yticks(range(len(text)), list(text))
    plt.tight_layout()
    return fig


def synthesis(m, s, CONFIG, use_cuda, ap,language=None,WaveRNN=False):
    """ Given the text, synthesising the audio """
    if language is None:
      language=CONFIG.phoneme_language
    text_cleaner = [CONFIG.text_cleaner]
    # print(phoneme_to_sequence(s, text_cleaner))s
    # print(sequence_to_phoneme(phoneme_to_sequence(s, text_cleaner)))
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(s, text_cleaner, language),
            dtype=np.int32)
    else:
        seq = np.asarray(text_to_sequence(s, text_cleaner), dtype=np.int32)
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    if use_cuda:
        chars_var = chars_var.cuda()
    mel_spec, linear_spec, alignments, stop_tokens = m.forward(
        chars_var.long())
    linear_spec = linear_spec[0].data.cpu().numpy()
    mel_spec = mel_spec[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    if not WaveRNN:
      wav = ap.inv_spectrogram(linear_spec.T)
      wav = wav[:ap.find_endpoint(wav)]
    else:
      wav = wavernn.generate(torch.FloatTensor(mel_spec.T).unsqueeze(0).cuda(), batched=True, target=11000, overlap=550)
    return wav, alignment, linear_spec, mel_spec, stop_tokens

def tts(model, text, CONFIG, use_cuda, ap, figures=True,name='figure',language=None):
    t_1 = time.time()
    waveform, alignment, spectrogram,mel_spec, stop_tokens = synthesis(model, text, CONFIG, use_cuda, ap,language=language,WaveRNN=False) 
    print(" >  Run-time with Griffin lim: {}".format(time.time() - t_1))
    t_1 = time.time()
    waveform2, _, _,_,_= synthesis(model, text, CONFIG, use_cuda, ap,language=language,WaveRNN=True) 
    print("\n >  Run-time with WaveRNN: {}".format(time.time() - t_1))
    if figures:
        fig = plot_alignment_with_text(alignment,text)
        visualize(alignment, spectrogram, stop_tokens,text,250, CONFIG,mel_spec) 
        fig.savefig(os.path.join(OUT_FOLDER,'alig_'+name+'.png'))
    print("Vocoder WaveRNN:")    
    IPython.display.display(Audio(waveform2, rate=ap2.sample_rate))
    print("Vocoder Griffin-Lim :") 
    IPython.display.display(Audio(waveform, rate=ap.sample_rate))  
    
    return alignment, spectrogram, stop_tokens,waveform,waveform2
  
# Set constants

MODEL_PATH = 'checkpoint.pth.tar'
CONFIG_PATH =  'TTS/config.json'
OUT_FOLDER = 'samples/'
try:
  os.mkdir(OUT_FOLDER)
except:
  pass

CONFIG = load_config(CONFIG_PATH)
use_cuda = True


VOCODER_MODEL_PATH = "WaveRNN/saver.pth.tar"
VOCODER_CONFIG_PATH = "WaveRNN/config_16K.json"
VOCODER_CONFIG = load_config(VOCODER_CONFIG_PATH)
use_cuda = True

# load the model
ap2 = AudioProcessor(**VOCODER_CONFIG.audio)
ap = AudioProcessor(**CONFIG.audio)

num_chars = len(phonemes) if CONFIG.use_phonemes else len(symbols)
! mv checkpoint_381200.pth.tar checkpoint.pth.tar
model= Tacotron(num_chars, CONFIG.embedding_size, ap.num_freq, ap.num_mels, CONFIG.r, CONFIG.memory_size)

# load model state
if use_cuda:
    cp = torch.load(MODEL_PATH)
else:
    cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()

#from utils.generic_utils import load_config
from WaveRNN.models.wavernn import Model

bits = 10

wavernn = Model(
        rnn_dims=512,
        fc_dims=512,
        mode=VOCODER_CONFIG.mode,
        mulaw=VOCODER_CONFIG.mulaw,
        pad=VOCODER_CONFIG.pad,
        use_aux_net=VOCODER_CONFIG.use_aux_net,
        use_upsample_net=VOCODER_CONFIG.use_upsample_net,
        upsample_factors=VOCODER_CONFIG.upsample_factors,
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=ap2.hop_length,
        sample_rate=ap2.sample_rate,
    ).cuda()



        
check = torch.load(VOCODER_MODEL_PATH)
wavernn.load_state_dict(check['model'])
if use_cuda:
    wavernn.cuda()
wavernn.eval();
print(check['step'])
    
    # Define Test Sentences

test_sentences =[ "  A inauguração da vila é quarta ou quinta-feira",
        "  Vote se você tiver o título de eleitor",
        "  Hoje é fundamental encontrar a razão da existência humana",
        "  A temperatura é mais amena à noite",
        "  Em muitas cidades a população está diminuindo.",
        "  Nunca se deve ficar em cima do morro",
        "  Para as pessoas estranhas o panorama é desolador",
        "  É bom te ver colhendo flores menino",
        "  Eu finjo me banhar num lago ao amanhecer",
        "  Sua sensibilidade mostrará o caminho",
        "  A Amazônia é a reserva ecológica do globo",
        "  O ministério mudou demais com a eleição",
        "  Novas metas surgem na informática",
        "  O capital de uma empresa depende de sua produção",
        "  Se não fosse ela tudo teria sido melhor",
        "  A principal personagem no filme é uma gueixa",
        "  Espere seu amigo em casa",
        "  A juventude tinha que revolucionar a escola",
        "  A cantora terá quatro meses para ensaiar seu canto",
        "  Esse tema foi falado no congresso."]

model.decoder.max_decoder_steps = 250

for idx, sentence in enumerate(test_sentences):
  align, spec, stop_tokens,wav,wav2 = tts(model, sentence, CONFIG, use_cuda, ap, figures=True,name=str(idx+1))
  print("arquivo gerado:",str(idx+1)+'.wav')
  ap.save_wav(wav, os.path.join(OUT_FOLDER,str(idx+1)+'.wav'))
  ap2.save_wav(wav2, os.path.join(OUT_FOLDER,str(idx+1)+'wavernn'+'.wav'))
  
  import IPython
while True:
  frase = input("Enter sentence: ")
  if frase == 'q':
    break
  print('Input Text: ',frase)
  frase = '   '+frase
  align, spec, stop_tokens,wav,wav2 = tts(model, frase, CONFIG, use_cuda, ap, figures=True,name='teste')
  #IPython.display.display(Audio(wav, rate=ap.sample_rate))
```

## CÓDIGO ADAPTADO

```
CODE
```

### Falmisgeraldo

Este é meu projeto autoral mais recente, é um sandbox onde estou construindo diversas soluções de big data, data analytics e NLP usando Python e treinando sets utilizando GPUs comuns – não obstante, este projeto também contempla fortemente a construção de modelos de aprendizado profundo para reconhecimento de padrões em séries temporais e, também, na análise de performance de vendas.

CÓDIGO JÁ COM COMENTÁRIOS

```markdown
CODE
```
