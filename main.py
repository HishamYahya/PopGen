from model import PopMusicTransformer
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    tf.compat.v1.disable_eager_execution()

    # declare model
    model = PopMusicTransformer(
        checkpoint='/Users/hisham/Downloads/REMI-tempo-chord-checkpoint',
        is_training=False)
    
    # generate from scratch
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/from_scratch.midi',
        prompt=None)
    
#     # generate continuation
#     model.generate(
#         n_target_bar=16,
#         temperature=1.2,
#         topk=5
#         output_path='./result/continuation.midi',
#         prompt='./data/evaluation/000.midi')
    
    # close model
    model.close()

if __name__ == '__main__':
    main()
