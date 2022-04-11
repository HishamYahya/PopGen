import utils
import os
from glob import glob
import random
import pickle

HSA_PATH = '/Users/hisham/University/Final Year Project/hierarchical-structure-analysis-main/POP909_HSA'

TRAINING_SET_RATIO = 0.8

random.seed(42)

def main():
	# sort paths first because glob changes order arbitrarily
	song_paths = sorted(glob(f'{HSA_PATH}/[0-9]*'))

	random.shuffle(song_paths)

	print(len(song_paths))

	song_paths = remove_songs_with_key_changes(song_paths)
	print(len(song_paths))

	song_paths = remove_songs_with_too_long_phrases(song_paths)
	print(len(song_paths))

	train_len = int(TRAINING_SET_RATIO * len(song_paths))

	train = song_paths[:train_len]
	test = song_paths[train_len:]


	with open('data/training_song_ids.txt', 'w+') as f:
		for song in train:
			f.write(f'{song[-3:]}\n')

	with open('data/testing_song_ids.txt', 'w+') as f:
		for song in test:
			f.write(f'{song[-3:]}\n')


def remove_songs_with_key_changes(paths):
	filtered_paths = []
	for path in paths:
		with open(f'{path}/analyzed_key.txt') as f:
			keys = set(f.readlines()) - {'\n'}
			if len(keys) == 1:
				filtered_paths.append(path)
	return filtered_paths

def remove_songs_with_too_long_phrases(paths):
	filtered_paths = []
	for path in paths:
		# returns None if there is a phrase that is too long
		if utils.get_phrase_items(f"{path}/human_label1.txt") \
			or utils.get_phrase_items(f"{path}/human_label2.txt") :
			filtered_paths.append(path)

	return filtered_paths	

def generate_event_dictionary(save_path=None):
	"""
	Returns: (event2word, word2event)
	"""
	event_names = set()

	event_names.add('Bar_None')
	event_names.add('Chord_N:N')

	note2index = dict(zip(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], range(12)))

	for note in note2index.keys():
		for quality in ['maj', 'min', 'aug', 'dom', 'dim', 'sus']:
			event_names.add(f'Chord_{note}:{quality}')

		for quality in ['maj', 'min']:
			event_names.add(f'Key_{note}:{quality}')
			
		for i in range(22, 108):
			event_names.add(f'Note On_{i}')
			
		for i in range(64):
			event_names.add(f'Note Duration_{i}')
		
		for i in range(32):
			event_names.add(f'Phrase Duration_{i}')
		
		for i in range(16):
			event_names.add(f'Chord Duration_{i}')
			
		for phrase in ['i', 'A', 'B', 'C', 'D', 'E', 'x', 'b', 'X', 'o']:
			event_names.add(f'Phrase_{phrase}')
		
		for i in range(1, 17):
			event_names.add(f'Position_{i}/16')
		
	word2event = {i: event for i, event in enumerate(sorted(list(event_names)))}
	event2word = {v: k for k, v in word2event.items()}

	with open(f'{save_path}/dictionary.pkl', 'wb') as f:
		pickle.dump((event2word, word2event), f)

	return event2word, word2event

if __name__ == '__main__':
	main()