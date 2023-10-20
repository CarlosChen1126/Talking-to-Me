import argparse

def final_tarin_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument(
        '--final_data_dir',
        type=str,
        default='./student_data/',
        help='Dataset Root. (default: ./student_data/)'
    )
	parser.add_argument(
        '--final_audio_dir',
        type=str,
        default='./student_data/audio_files/',
        help='Audio Files Root. (default: ./student_data/audio_files/)'
    )
    parser.add_argument(
        '--final_vision_train_dir',
        type=str,
        default='./split_frame_train/',
        help='Vision Train Split. (default: ./split_frame_train/)'
    )
    parser.add_argument(
        '--final_vision_test_dir',
        type=str,
        default='./split_frame_test/',
        help='Vision Test Split. (default: ./split_frame_test/)'
    )
    parser.add_argument(
        '--final_audio_feat_dir',
        type=str,
        default='./hubert_features/',
        help='Audio Features Dir. (default: ./hubert_features/)'
    )
	parser.add_argument(
		'--final_out_path',
		type=str,
		default='./final_output/pred.csv',
		help='Prediction Output Path. (default: ./final_output/pred.csv)'
	)
    parser.add_argument(
		'--final_train_type',
		type=str,
		default='all',
		choices=['all', 'vision', 'audio'],
		help='Choose training type. (default: all)\n(Options: all (vision + audio), vision (vision oonly), audio (audio only))'
	)

	return parser