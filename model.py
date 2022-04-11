#####################
#   Modified version of code taken from https://github.com/YatingMusic/remi
#   All modifications are commented and tagged with "%"
#####################

import tensorflow as tf
import numpy as np
import miditoolkit
import modules
import pickle
import utils
import time
import random
import os

from prepare_data import generate_event_dictionary

class PopMusicTransformer(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, checkpoint, is_training=False):
        # load dictionary
        self.dictionary_path = '{}/dictionary.pkl'.format(checkpoint)
        if not os.path.exists(self.dictionary_path):
            generate_event_dictionary(checkpoint)
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        # print(self.event2word.keys())
        # model settings
        self.x_len = 128
        self.mem_len = 256
        self.n_layer = 12
        self.d_embed = 256
        self.d_model = 256
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        self.learning_rate = 0.0002
        # load model
        self.is_training = is_training
        if self.is_training:
            self.batch_size = 4
        else:
            self.batch_size = 1
        self.checkpoint_path = checkpoint
        self.load_model()

    ########################################
    # load model
    ########################################
    def load_model(self):
        # placeholders
        self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
        # model
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        initializer = tf.compat.v1.initializers.random_normal(stddev=0.02, seed=None)
        proj_initializer = tf.compat.v1.initializers.random_normal(stddev=0.01, seed=None)
#         with tf.distribute.MirroredStrategy().scope():
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            xx = tf.transpose(self.x, [1, 0])
            yy = tf.transpose(self.y, [1, 0])
            loss, self.logits, self.new_mem = modules.transformer(
                dec_inp=xx,
                target=yy,
                mems=self.mems_i,
                n_token=self.n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_ff,
                dropout=self.dropout,
                dropatt=self.dropout,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=self.is_training,
                mem_len=self.mem_len,
                cutoffs=[],
                div_val=-1,
                tie_projs=[],
                same_length=False,
                clamp_len=-1,
                input_perms=None,
                target_perms=None,
                head_target=None,
                untie_r=False,
                proj_same_dim=True)
        self.avg_loss = tf.reduce_mean(loss)
        # vars
        all_vars = tf.compat.v1.trainable_variables()
        grads = tf.gradients(self.avg_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))
        all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        # optimizer
        decay_lr = tf.compat.v1.train.cosine_decay(
            self.learning_rate,
            global_step=self.global_step,
            decay_steps=400000,
            alpha=0.004)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decay_lr)
        self.train_op = optimizer.apply_gradients(grads_and_vars, self.global_step)
        # saver
        self.saver = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        try:
#         self.saver.save(self.sess, f'{self.checkpoint_path}/model')
            self.saver.restore(self.sess, tf.compat.v1.train.latest_checkpoint(self.checkpoint_path))
#         print('restored')
        except:
            self.saver.save(self.sess, f'{self.checkpoint_path}/model')
            

    ########################################
    # temperature sampling
    ########################################
    def temperature_sampling(self, logits, temperature, topk):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction

    ########################################
    #% generate
    ########################################
    def generate(self,
        n_target_bar,
        temperature, 
        topk, 
        output_path=None, 
        prompt=None, 
        tempo_prompt=False, 
        chord_rec_prompt=False, 
        key_prompt=None, 
        phrase_prompt=True):
        """Generates a song with bar count of >= n_target_bar, only stopping when the threshold is reached
        and when the last phrase was concluded

        Args:
            n_target_bar: minimum number of generated bars
            temperature: for temperature sampling
            topk: for temperature sampling
            output_path (default None): path of midi file to write
            prompt (default None): path of prompt midi file OR list of events of a song
            tempo_prompt (default False): whether to encode tempo information in event list
            chord_rec_prompt (default False): whether to use chord recognition to encode chords
            key_prompt (default None): string of key of piece (optional)
            phrase_prompt (default True): whether the prompt should be assumed to be a single phrase

        Returns:
            a word representation of the generated event list
        """

        # if prompt, load it. Or, random start
        if prompt:
            if type(prompt) == list:
                words = [prompt]
            else:
                events = utils.extract_events_from_path(prompt, chords=chord_rec_prompt, tempo=tempo_prompt, key=key_prompt)
                words = [[self.event2word['{}_{}'.format(e.name, e.value)] for e in events]]
                words[0].append(self.event2word['Bar_None'])
        else:
            words = []
            for _ in range(self.batch_size):
                ws = []
                
                if tempo_prompt:
                    #% initialise with random tempo
                    tempos = [v for k, v in self.event2word.items() if 'Tempo' in k]
                    ws.append(np.random.choice(tempos))

                #% add first bar    
                ws.append(self.event2word['Bar_None'])

                #% initialise key
                keys = [v for k, v in self.event2word.items() if 'Key' in k]
                
                #% try key prompt
                try:
                    key = self.event2word[f'Key_{key_prompt}']
                #% else choose random key
                except:
                    key = np.random.choice(keys)
                
                ws.append(key)


                words.append(ws)
        # initialize mem
        batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        phrase_duration = -1
            
        while current_generated_bar < n_target_bar or phrase_duration > -1:
            # input
            if initial_flag:
                temp_x = np.zeros((self.batch_size, original_length))
                for b in range(self.batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x = np.zeros((self.batch_size, 1))
                for b in range(self.batch_size):
                    temp_x[b][0] = words[b][-1]
            # prepare feed dict
            feed_dict = {self.x: temp_x}
            for m, m_np in zip(self.mems_i, batch_m):
                feed_dict[m] = m_np

            _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
            _logit = _logits[-1, 0]


            word = self.temperature_sampling(
                logits=_logit, 
                temperature=temperature,
                topk=topk)
            words[0].append(word)

                
            # if bar event (only work for batch_size=1)
            if word == self.event2word['Bar_None']:
                # print(f'BAR {current_generated_bar}')
                current_generated_bar += 1
                phrase_duration -= 1
                # print(f'PHRASE DURATION DECREMENTED: {phrase_duration}')
        
            # elif 'Phrase_' in self.word2event[word]:
            #     print(f'PHRASE AT BAR {current_generated_bar}, {self.word2event[word]}')
            
            elif 'Phrase Duration' in self.word2event[word]:
                phrase_duration = int(self.word2event[word].split('_')[1])
                # print(f'PHRASE DURATION INITIALISED: {self.word2event[word]}')

            # re-new mem
            batch_m = _new_mem
        # write

        if prompt:
            if output_path:
                utils.write_midi(
                    words=words[0],
                    word2event=self.word2event,
                    output_path=output_path,
#                     prompt_path=prompt
                )
            return words[0]
        else:
            if output_path:
                utils.write_midi(
                    words=words[0],
                    word2event=self.word2event,
                    output_path=output_path,
                    prompt_path=None)
            return words[0]
            

    ########################################
    # prepare training data
    ########################################
    def prepare_data(self, song_paths):
        # extract events
        all_events = []
        for path in song_paths:
            events = utils.extract_events(path)
            transpositions = utils.get_in_all_keys(events)
            all_events += transpositions
        
        random.shuffle(all_events)
        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # something is wrong
                    # you should handle it for your own purpose
                    print('something is wrong! {}'.format(e))
            all_words.append(words)
        # to training data
        self.group_size = 5

        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            pairs = np.array(pairs)
            # abandon the last
            for i in np.arange(0, len(pairs)-self.group_size, self.group_size*2):
                data = pairs[i:i+self.group_size]
                if len(data) == self.group_size:
                    segments.append(data)
        segments = np.array(segments)
        return segments

    ########################################
    # finetune
    ########################################
    def finetune(self, training_data, output_checkpoint_folder):
        # shuffle
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        num_batches = len(training_data) // self.batch_size
        st = time.time()
        for e in range(200):
            total_loss = []
            for i in range(num_batches):
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    # prepare feed dict
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np
                    # run
                    _, gs_, loss_, new_mem_ = self.sess.run([self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    batch_m = new_mem_
                    total_loss.append(loss_)
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st))

            if int(e) % 5 == 0:  
                self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
            # stop
            if np.mean(total_loss) <= 0.1:
                self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
                break

    ########################################
    # close
    ########################################
    def close(self):
        self.sess.close()
