from model import PopMusicTransformer
import utils
import sys
import os
import tensorflow as tf

CHECKPOINT = 'model_25'

HSA_PATH = '/Users/hisham/University/Final Year Project/hierarchical-structure-analysis-main/POP909_HAT'

def main():
	if len(sys.argv) == 0:
		print('Enter a path')
		return
	path = sys.argv[0]
	if not os.path.exists(path):
		print('Path does not exist')
		return
	path = '/Users/hisham/University/Final Year Project/remi-master/still_alive.mid'
	tf.compat.v1.disable_eager_execution()
	tf.compat.v1.reset_default_graph()
	
	model = PopMusicTransformer(
		CHECKPOINT,
		is_training=False
	)

	words = model.generate(
		16,
		1.2,
		5,
		prompt=path,
		chord_rec_prompt=True,
		key_prompt='E:min'
	)

	events = utils.word_to_event(words, model.word2event)

	new_events = utils.add_played_chords(events)

	words = [model.event2word['{}_{}'.format(e.name, e.value)] for e in new_events]
	utils.write_midi(words, model.word2event, 'test.mid')

	with open('words.txt', 'w+') as f:
		for e in events:
			f.write('{}_{}'.format(e.name, e.value) + '\n')

	model.close()

if __name__ == '__main__':
	main()