# Prepare environment
1. donwload s3prl: https://github.com/s3prl/s3prl
2. run `get_dataset.sh` to download dataset, `final_download.sh` to donwload preprocessed data
3. `python ./audio/audio_extract_hubert_base.py` to obtain audio features (located at hubert_features/{train, test})

# Training Audio Model with Hubert Features
`python train.py --final_train_type audio`

# Training Hybrid Model with Hubert Features
`python train.py --final_train_type all`