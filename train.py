from model import PopMusicTransformer
import os
import tensorflow as tf

CHECKPOINT = 'model'

HSA_PATH = '/Users/hisham/University/Final Year Project/hierarchical-structure-analysis-main/POP909_HAT'


def main():
	tf.compat.v1.disable_eager_execution()
	tf.compat.v1.reset_default_graph()

	if not os.path.exists(CHECKPOINT):
		os.mkdir(CHECKPOINT)

	# declare model
	model = PopMusicTransformer(
		checkpoint=CHECKPOINT,
		is_training=True)

	with open('data/training_song_ids.txt') as f:
		song_ids = [song.strip() for song in f.readlines()]

	song_paths = [f'{HSA_PATH}/{song_id}' for song_id in song_ids]

	training_data = model.prepare_data(song_paths)
	
	model.finetune(
        training_data=training_data,
        output_checkpoint_folder='model'
	)

	model.close()

if __name__ == "__main__":
	main()

