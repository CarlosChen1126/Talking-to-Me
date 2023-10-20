# Make sure you have kaggle.json added to your environment
kaggle competitions download -c dlcv-final-problem1-talking-to-me
unzip dlcv-final-problem1-talking-to-me.zip

rm dlcv-final-problem1-talking-to-me.zip
mv ./student_data/student_data/test/ ./student_data/test/
mv ./student_data/student_data/train/ ./student_data/train/
mv ./student_data/student_data/videos/ ./student_data/videos/
rm ./student_data/student_data/ -r