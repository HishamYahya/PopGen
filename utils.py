#####################
#   Modified version of code taken from https://github.com/YatingMusic/remi
#   All modifications are commented and tagged with "%"
#####################
import chord_recognition
import numpy as np
import miditoolkit
import copy
import re
# from math import floor

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16

#% 480 = quarter note. Chords are played within a grid of quarter notes
DEFAULT_CHORD_DURATION_BINS = np.arange(480, 3841*2, 480, dtype=int)

#% 120 = sixteenth notes. Notes are played within a grid of sixteenth notes
DEFAULT_DURATION_BINS = np.arange(120, 3841*2, 120, dtype=int)

#% 120 * 16 = a bar. Phrases are defined for each bar
DEFAULT_PHRASE_DUR_BINS = np.arange(120*16, 120*16*32+1, 120*16, dtype=int)

#% slow, mid, fast
DEFAULT_TEMPO_INTERVALS = [range(30, 76), range(76, 120), range(120, 210)]

#% max number of bars a phrase can be
MAX_PHRASE_LENGTH = 32

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    tick_ratio = round(480 / midi_obj.ticks_per_beat)
    midi_obj.ticks_per_beat = 480
    midi_obj.max_tick = tick_ratio*midi_obj.max_tick
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note.start *= tick_ratio
        note.end *= tick_ratio
        note_items.append(Item(
            name='Note', 
            start=note.start, 
            end=note.end, 
            velocity=note.velocity, 
            pitch=note.pitch))
    note_items.sort(key=lambda x: x.start)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    return note_items, tempo_items

# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      

# extract chord
def extract_chords(items):
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    return output

# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups, key=None, tempo=None):
    events = []
    n_downbeat = 0
    
    #% check if there are any tempo items. Set the tempo to be the value of one of them
    for item in groups[0][1:-1]:
        if item.name == 'Tempo':
            tempo = item.pitch
    
    #% if tempo item not found and hyperparameter was not set, don't add tempo event
    if tempo is None:
        pass
    elif tempo in DEFAULT_TEMPO_INTERVALS[0]:
        events.append(Event('Tempo', None, 'slow', None))
    elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
        events.append(Event('Tempo', None, 'mid', None))
    elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
        events.append(Event('Tempo', None, 'fast', None))
    elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
        events.append(Event('Tempo', None, 'slow', None))
    elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
        events.append(Event('Tempo', None, 'fast', None))
    
    #% variable for keeping track when the last played note ends
    last_note_end = -1
    
    #% variable to keep track when the last encountered phrase ends
    last_phrase_end = -1
    
    for i in range(len(groups)):
        
        #% skip bar if there are no notes in it and if a note doesn't end during it
        #% and if a phrase still hasn't ended
        if 'Note' not in [item.name for item in groups[i][1:-1]] \
            and last_note_end <= DEFAULT_RESOLUTION * 4 * i \
            and last_phrase_end <= DEFAULT_RESOLUTION * 4 * i:
            continue
            
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        
        events.append(Event(
            name='Bar',
            time=None,
            value=None,
            text='{}'.format(n_downbeat)))
        
        #% Key event repeated alongside the Bar event
        if key:
            events.append(Event(
                name='Key',
                time=None, 
                value=key,
                text=key))


        for item in groups[i][1:-1]:
            #% skip tempo items (dealt with above)
            if item.name == 'Tempo':
                continue
            
            #% append Phrase and Phrase Duration event without a Position preceeding them (always will be right after a Bar event)
            if item.name == 'Phrase':
                events.append(Event(
                    name='Phrase', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_PHRASE_DUR_BINS-duration))
                events.append(Event(
                    name='Phrase Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_PHRASE_DUR_BINS[index])))
                # update last_phrase_end
                if item.end > last_phrase_end:
                    last_phrase_end = item.end
                continue
                
            
            # add position event
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start)))
            
            if item.name == 'Note':
                # #% VELOCITY NOT IMPLEMENTED
                # velocity_index = np.searchsorted(
                #     DEFAULT_VELOCITY_BINS, 
                #     item.velocity, 
                #     side='right') - 1
                # events.append(Event(
                #     name='Note Velocity',
                #     time=item.start, 
                #     value=velocity_index,
                #     text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
                
                #% update last_note_end
                if item.end > last_note_end:
                    last_note_end = item.end
            
            #% add Chord and Chord Duration events
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_CHORD_DURATION_BINS-duration))
                events.append(Event(
                    name='Chord Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_CHORD_DURATION_BINS[index])))
   
    return events

#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events
def write_midi(words, word2event, output_path, events=None, prompt_path=None):
    if events is None:
        events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events)-2):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')
#             events[i+1].name == 'Note Velocity' and \
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note On' and \
            events[i+2].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # velocity
#             index = int(events[i+1].value)
#             velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i+1].value)
            # duration
            index = int(events[i+2].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, 100, pitch, duration])
        elif events[i].name == 'Position' and events[i+1].name == 'Chord':
            position = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([position, events[i+1].value])
        elif events[i].name == 'Tempo':
            if events[i].value == 'slow':
                tempo = 70
            elif events[i].value == 'mid':
                tempo = 95
            elif events[i].value == 'fast':
                tempo = 130
            temp_tempos.append([0, tempo])
    
    temp_tempos = [[0, 85]]

    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == 'Bar':
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                chords.append([st, value])
    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        tick_ratio = round(DEFAULT_RESOLUTION / midi.ticks_per_beat)
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        midi.max_tick = tick_ratio*midi.max_tick
        last_time = DEFAULT_RESOLUTION * 4 * 4
        for note in midi.instruments[0].notes:
            note.start *= tick_ratio
            note.end *= tick_ratio
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)

#################################
# CODE BELOW THIS BLOCK IS NEW
#################################

# replace all flats to sharps
normalised_notes = {
    'Db': 'C#',
    'Eb': 'D#',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#'
}
def normalise_note_name(note):
    if note in normalised_notes:
        return normalised_notes[note]
    return note

# notes are assigned numbers sorted by pitch
note2index = dict(zip(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], range(12)))
index2note = {v: k for k, v in note2index.items()}

# ticks in 16th notes
ticks_16th = 120
# ticks in quarter notes
ticks_quarter = ticks_16th*4
# ticks in a bar notes
ticks_bar = ticks_16th*16

def get_phrase_items(path):
    """Reads human labeled phrases and converts them to Item objects
        Args:
            path of human_label file
        
        Returns:
            phrase items
    
    """
    with open(path) as f:
        line = f.readline().strip()
        phrase_items = []
        cur_pos = 0
        for phrase in re.findall(r'\w\d+', line):
            phrase, dur = phrase[0], int(phrase[1:])
            if dur > MAX_PHRASE_LENGTH:
                return None
            phrase_items.append(Item('Phrase', cur_pos*ticks_bar, (dur + cur_pos)*ticks_bar, 100, phrase))
            cur_pos += dur
    return phrase_items

def get_chord_name(s):
    """Takes in a string of chord notation and returns the normalised version of it
    """
    chord_name = re.search(r'.*:(maj|min|aug|dim|7|sus)', s)
    if not chord_name:
        chord_name = re.sub(r'hdim7', 'dim', s)
        chord_name = re.sub(r'9.*', 'maj', chord_name)
    else:
        chord_name = chord_name.group()
        
    root, tone = chord_name.split(':')
    if tone == '7':
        tone = 'dom'
    
    return f'{normalise_note_name(root)}:{tone}'


def extract_events(song_path, label=1):
    """Get the event representation of a given song.
    
    This function takes in a path of a song in the HSA_POP909 dataset
    and encodes the melody, phrase, and chord info into the proposed
    REMI representation

    Args:
        song_path
        label (optional, defaults to 1): which of the two human_label files to encode
    
    Returns:
        the REMI representation of the song
    """

    items = get_phrase_items(f"{song_path}/human_label{label}.txt")
    if items is None:
        if label == 1:
           items = get_phrase_items(f"{song_path}/human_label2.txt") 
        else:
           items = get_phrase_items(f"{song_path}/human_label1.txt")  
    
    # return None if both labels have a phrase that's longer than MAX_PHRASE_LENGTH
    if items is None:
        return None
            
    # add Chord items
    with open(f"{song_path}/finalized_chord.txt") as f:
        cur_pos = 0
        for line in f.readlines():
            try:
                root_note, dur = [int(x) for x in line.split()[-2:]]
                chord_name = get_chord_name(line.split()[0])
            except:
                chord_name = 'N:N'
                dur = int(re.findall(r'\d+', line)[0])

            items.append(Item('Chord', cur_pos*ticks_quarter, (dur + cur_pos)*ticks_quarter, 100, chord_name))
            
            cur_pos += dur

    # add Note items
    with open(f"{song_path}/melody.txt") as f:
        cur_pos = 0
        for line in f.readlines():
            note, dur = [int(x) for x in line.split()]
            if note != 0:
                items.append(Item('Note', cur_pos*ticks_16th, (dur + cur_pos)*ticks_16th, 100, note))
            cur_pos += dur

    
        
    max_time = sorted(items, key=lambda x: x.end if x.end else 0)[-1].end    
    groups = group_items(items, max_time)
    
    # get key of song
    with open(f"{song_path}/analyzed_key.txt") as f:
        key = f.readline().strip()
        root, tone = key.split(':')
        key = f'{normalise_note_name(root)}:{tone}'
    
    # optional Tempo item
    with open(f"{song_path}/tempo.txt") as f:
        tempo = int(f.readline())
        
    return item2event(groups, key=key)


def get_in_all_keys(events):
    """Takes in a song as a list of events and transposes it to all other keys
    
        Args:
            events: song in list of event form
        
        Returns:
            a list of 12 songs in event form that are transpositions of the input and cover all 12 roots
    """
    songs = [events]
    for i in range(-6, 6):
        if i == 0:
            continue
        
        song = copy.deepcopy(songs[0])
        for event in song:
            if event.name == 'Note On':
                event.value += i
                event.text = str(event.value)
            if event.name == 'Chord' or event.name == 'Key':
                root, quality = event.value.split(':')
                if root in note2index.keys():
                    new_note = index2note[(note2index[root]+i) % 12]
                    event.value = new_note + ':' + quality
                    event.text = new_note + ':' + quality
        songs.append(song)
        
    return songs

def add_played_chords(events):
    """Takes in a song in event list form and adds Note events to play the Chord events
    """
    new_events = []
    for i, event in enumerate(events):
        # skip Chord Duration events because they would have already been added the event prior
        if event.name == 'Chord Duration':
            continue
        new_events.append(event)
        if event.name != 'Chord' or events[i+1].name != 'Chord Duration':
            continue
        new_events.append(events[i+1])
        start = event.time
        index = (int(events[i+1].value) + 1) * 4 -1
        duration = DEFAULT_DURATION_BINS[index]

        # loop through every chord tone and play it
        for tone in get_chord_tones(event.value):
            new_events.append(events[i-1])
            new_events.append(Event(
                    name='Note On',
                    time=start, 
                    value=48+tone,
                    text='{}'.format(48+tone)))
            new_events.append(Event(
                name='Note Duration',
                time=start,
                value=index,
                text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))

    return new_events


def get_chord_tones(chord):
    """Takes in a normalised chord in string form and returns a list of the notes comprising that chord in integer notation
    """
    root, quality = chord.split(':')
    if root == 'N':
        return []
    root = note2index[root]
    tones = [root]
    # integer notation
    if quality == 'maj':
        # add major third and perfect fifth
        tones.append((root+4)%12)
        tones.append((root+7)%12)
    
    if quality == 'min':
        # add minor third and perfect fifth
        tones.append((root+3)%12)
        tones.append((root+7)%12)
        
    if quality == 'dim':
        # add minor third and flat fifth
        tones.append((root+3)%12)
        tones.append((root+6)%12)
        
    if quality == 'aug':
        # add major third and sharp fifth
        tones.append((root+4)%12)
        tones.append((root+8)%12)
        
    if quality == 'dom':
        # add major third, perfect fifth, and minor seventh
        tones.append((root+4)%12)
        tones.append((root+7)%12)
        tones.append((root+10)%12)
    
    if quality == 'sus': # normalise all sus chords to sus4
        # add perfect fourth and perfect fifth
        tones.append((root+5)%12)
        tones.append((root+7)%12)
    return tones

def extract_events_from_path(input_path, chords=True, tempo=True, key=None):
    note_items, tempo_items = read_items(input_path)
    note_items = quantize_items(note_items)
    max_time = note_items[-1].end
    items = []
    if chords:
        chord_items = extract_chords(note_items)
        items += chord_items

    if tempo:
        items+= tempo_items

    items += note_items
    groups = group_items(items, max_time)
    events = item2event(groups, key=key)
    events = add_A_phrase(events)
    return events
  
def add_A_phrase(events):
    """Add a Phrase_A event
    """
    bar_count = 0
    has_key = 0
    has_tempo = 0
    for event in events:
        if event.name == 'Bar':
            bar_count += 1
        if event.name == 'Key':
            has_key = 1
        if event.name == 'Tempo':
            has_tempo = 1
    phrase = [
        Event('Phrase', None, 'A', 'A'),
        Event('Phrase Duration', None, str(bar_count-1), str(bar_count-1))
    ]

    return events[:1+has_key+has_tempo] + phrase + events[1+has_key+has_tempo:]