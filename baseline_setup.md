### Setting up the Baseline model for training on HPC
- `ssh` into cluster: `ssh $USER@log-1.hpc.nyu.edu`
- do either `1` or `2`. Option `1` provides a simpler workflow than Julia's suggestion.
  1. follow [github's instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account#about-addition-of-ssh-keys-to-your-account) to create a new ssh key for your HPC user, add it to your github account, and clone the repo to your scratch directory. Make sure to select the `linux` option at the top of the pages!
    - after completing this run:
      ```
      cd /scratch/$USER/
      git clone git@github.com:reedrosenbluth/LASS.git
      ```
  2. transfer repo from local machine to your HPC scratch directory: `scp -r <myFolder> $USER@log-1.hpc.nyu.edu:/scratch/$USER/`
    - setup the directory as a git remote for your local repo (so you can pull changes back to your local machine) by following [Julia's instructions](https://github.com/juliawilkins/nyu-csgy9223-ML25/blob/main/HPC_tips.md)
- install singuconda
  ```
  cd /scratch/$USER/
  curl -L https://github.com/beasteers/singuconda/raw/main/singuconda --output /scratch/$USER/singuconda

  chmod +x ~/singuconda

  # run the script 
  bash singuconda
  ```
  - make sure to select `overlay-50G-10M`, CUDA GPU, and Python 3.10.9
- create the conda environment
  ```
  cd /scratch/$USER

  # activate the singularity container in read/write mode
  ./singrw

  cd /scratch/$USER/LASS

  # create the conda env and install dependencies
  conda env create -f environment.yml

  # later, when you want to activate the conda env, run:
  conda activate AudioSep
  ```
- download the datasets and captions
  ```
  cd /scratch/$USER/

  # activate singularity if not already active
  ./singrw
  cd /scratch/$USER/

  # install zenodo_get
  pip install zenodo_get

  mkdir -p /scratch/$USER/clotho_dataset
  mkdir -p /scratch/$USER/fsd50k_dataset
  mkdir -p /scratch/$USER/fsd50k_captions

  # download clotho
  cd /scratch/$USER/clotho_dataset
  zenodo_get 10.5281/zenodo.4783391

  # download fsd50k
  cd /scratch/$USER/fsd50k_dataset
  zenodo_get 10.5281/zenodo.4060432

  # download fsd50k captions
  cd /scratch/$USER/fsd50k_captions
  zenodo_get 10.5281/zenodo.10887496

  ```
- unzip the datasets
  ```
  cd /scratch/$USER/clotho_dataset

  pip requests py7zr

  # clotho dev set
  python -c "import py7zr; py7zr.SevenZipFile('clotho_audio_development.7z').extractall('/scratch/$USER/clotho_dataset/development')"

  # clotho validation set
  python -c "import py7zr; py7zr.SevenZipFile('clotho_audio_validation.7z').extractall('/scratch/$USER/clotho_dataset/validation')"

  # clotho eval set
  python -c "import py7zr; py7zr.SevenZipFile('clotho_audio_evaluation.7z').extractall('/scratch/$USER/clotho_dataset/evaluation')"

  # remove zip files
  rm *.7z

  cd /scratch/$USER/fsd50k_dataset

  # fsd50k dev set
  zip -s 0 FSD50K.dev_audio.zip --out unsplit_dev.zip
  unzip unsplit_dev.zip

  # fsd50k eval set
  zip -s 0 FSD50K.eval_audio.zip --out unsplit_eval.zip
  unzip unsplit_eval.zip

  # fsd50k metadata files
  unzip FSD50K.ground_truth.zip
  unzip FSD50K.metadata.zip
  unzip FSD50K.doc.zip

  # remove zip files
  rm *.zip *.z[0-9]*
  ```
- download the model checkpoints
  ```
  # download CLAP model weights and move into AudioSep repo
  wget https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt?download=true
  mkdir /scratch/$USER/LASS/checkpoint
  mv music_speech_audioset_epoch_15_esc_89.98.pt /scratch/$USER/LASS/checkpoint/

  # download baseline model checkpoint
  cd /scratch/$USER/
  zenodo_get 10.5281/zenodo.10887459
  ```
