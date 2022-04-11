from model import PopMusicTransformer
import os
import tensorflow as tf
import utils

CHECKPOINT = 'model_25'

HSA_PATH = '/Users/hisham/University/Final Year Project/hierarchical-structure-analysis-main/POP909_HAT'


def main():
	tf.compat.v1.disable_eager_execution()
	tf.compat.v1.reset_default_graph()

	# declare model
	model = PopMusicTransformer(
		checkpoint=CHECKPOINT,
		is_training=False)

	
	# generate from scratch
	words = model.generate(
		n_target_bar=16,
		temperature=1.2,
		topk=5
	)

	events = utils.word_to_event(words, model.word2event)

	new_events = utils.add_played_chords(events)

	words = [model.event2word['{}_{}'.format(e.name, e.value)] for e in new_events]
	utils.write_midi(words, model.word2event, 'test.mid')

	with open('words.txt', 'w+') as f:
		for e in events:
			f.write('{}_{}'.format(e.name, e.value) + '\n')

	model.close()

if __name__ == "__main__":
	main()

