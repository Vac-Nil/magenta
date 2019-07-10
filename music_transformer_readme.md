# Music Transformer

## Quick Start

```
sudo apt-get install libasound2-dev
sudo apt-get install libjack-dev
pip install -r requirements.txt
mkdir -p data/maestro/raw
cd data/maestro/raw
wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip
unzip maestro-v2.0.0-midi.zip 
mv maestro-v2.0.0/*/*.midi .
bash magenta/models/score2perf/data_gen.sh
CUDA_VISIBLE_DEVICES=0,1 exp_name=music_transformer_1 bash magenta/models/score2perf/run.sh
```
