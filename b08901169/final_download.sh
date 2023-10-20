# preload s3prl model
python -c "import torch; from s3prl.nn import S3PRLUpstream; S3PRLUpstream('hubert_large_ll60k', refresh=False)"

# download audio files
wget https://www.dropbox.com/s/msmqumi1hfxpl9j/audio_files.zip?dl=1 -O audio_files.zip
unzip audio_files.zip
rm audio_files.zip

# download vision model ckpt
mkdir VIS_CKPT
cd VIS_CKPT
wget https://www.dropbox.com/s/aigiep7dm6292hc/A2_vis_feature_30_1e-4_one_vis.zip?dl=1 -O A2_vis_feature_30_1e-4_one_vis.zip
unzip A2_vis_feature_30_1e-4_one_vis.zip
rm A2_vis_feature_30_1e-4_one_vis.zip
cd ..

# download vision train/test split
mkdir split_frame_train
cd split_frame_train
wget https://www.dropbox.com/s/gkpwlmzr3u6vyab/1-50.zip?dl=1 -O 1-50.zip
wget https://www.dropbox.com/s/z4yz4f3uaqs4txc/51-100.zip?dl=1 -O 51-100.zip
wget https://www.dropbox.com/s/8fzp0s23j9jxu16/101-150.zip?dl=1 -O 101-150.zip
wget https://www.dropbox.com/s/v1nr22mtcuvwle9/151-200.zip?dl=1 -O 151-200.zip
wget https://www.dropbox.com/s/l3yoc3mznvfpkg9/201-250.zip?dl=1 -O 201-250.zip
wget https://www.dropbox.com/s/t0fxl0ydmxyme64/251-300.zip?dl=1 -O 251-300.zip
wget https://www.dropbox.com/s/bxxf9f5mx784irr/301-350.zip?dl=1 -O 301-350.zip
wget https://www.dropbox.com/s/kvj5595h5e89bag/351-394.zip?dl=1 -O 351-394.zip
unzip -q "./1-50.zip"
unzip -q "./51-100.zip"
unzip -q "./101-150.zip"
unzip -q "./151-200.zip"
unzip -q "./201-250.zip"
unzip -q "./251-300.zip"
unzip -q "./301-350.zip"
unzip -q "./351-394.zip"
rm *.zip

# cd ..
wget https://www.dropbox.com/s/mvdocfg88rpym0j/split_frame_test.zip?dl=1 -O split_frame_test.zip
unzip -q "./split_frame_test.zip"
rm split_frame_test.zip