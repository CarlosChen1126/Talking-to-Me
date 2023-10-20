# preload s3prl model
python -c "import torch; from s3prl.nn import S3PRLUpstream; S3PRLUpstream('hubert', refresh=False)"

# download audio files
wget https://www.dropbox.com/s/msmqumi1hfxpl9j/audio_files.zip?dl=1 -O audio_files.zip
unzip audio_files.zip
rm audio_files.zip