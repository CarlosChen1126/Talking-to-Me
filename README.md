# DLCV Final Project ( Talking to me )

# How to run your code?
* Step 1

We've provided a script file which containing the prepared pictures and mfcc features:

```shell script=
cd final-project-challenge-1-bcnm
cd best
bash data_download.sh
```

If there is a new dataset that need to be preprocessed, please modify the paths of [video, seg, bbox] for the vision data in <code>vision_prep.py</code> and [video, seg] for the audio data in <code>mfcc_prep.py</code>.

Also remember to modify the output picture, audio, and feature paths.

```shell script=
# if needed
python3 vision_prep.py
python3 mfcc_prep.py
```

* Step 2

During the training phase, we train the vision model, the audio model, and the hybrid model respectively.

There are several parameters to the running script:

```shell script=
bash run_best.sh <vis/aud/comb> <train/test> <path to vision model ckpt> <path to audio model ckpt> <path to hybrid model ckpt> <path to vision training data> \
<path to vision testing data> <path to mfcc feature data> <all/part> <path to store ckpt> <path to store output csv>
```

vis: vision model  /  aud: audio model  /  comb: hybrid model

all: use all data  /  part: only use the data that including both useful visual information and mfcc feature

An training example is listed below:

```shell script=
# vision
bash run_best.sh vis train x x x ./split_frame_train ./split_frame_test ./mfcc part ./CKPT x

# audio
bash run_best.sh aud train x x x ./split_frame_train ./split_frame_test ./mfcc all ./CKPT x

# hybrid
bash run_best.sh comb train ./CKPT/vis_best.ckpt ./CKPT/aud_best.ckpt x ./split_frame_train ./split_frame_test ./mfcc all ./CKPT x
```

Note that all of the three model will only store their best ckpt.

* Step 3

During the inferencing phase, one should load in three ckpt and the path of the output csv file:

```shell script=
bash run_best.sh comb test ./CKPT/vis_best.ckpt ./CKPT/aud_best.ckpt ./CKPT/comb_best.ckpt ./split_frame_train ./split_frame_test ./mfcc all x ./pred_best.csv
```

Here we also provide a quick inference which using the ckpt downloaded in (Step 1) to skip the training of (Step 2).

```shell script=
bash run_best.sh comb test ./VIS_best.ckpt ./AUD_best.ckpt ./COMB_all_best.ckpt ./split_frame_train ./split_frame_test ./mfcc all x ./pred_best.csv
```

# Usage
To start working on this final project, you should clone this repository into your local machine by the following command:

    git clone https://github.com/ntudlcv/DLCV-Fall-2022-Final-1-<team name>.git
  
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1Y-gwBmucYgbWLLk-u6coHi7LybFLXgA9gV8KiOiKShI/edit?usp=sharing) to view the slides of Final Project - Talking to me. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

# Submission Rules
### Deadline
111/12/29 (Thur.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion
