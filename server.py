from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import ORJSONResponse, FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import PopMusicTransformer
import utils
import tensorflow as tf
from datetime import datetime
import os
import s3fs
from pathlib import Path
from typing import List
from starlette.background import BackgroundTask

CHECKPOINT = 'model'

if not os.path.exists('responses'):
	os.mkdir('responses')

if not os.path.exists('model'):
	os.mkdir('model')
	fs = s3fs.S3FileSystem()

	bucket = 's3://popgen-model/model/'
	files = fs.ls(bucket)

	for f in files:
		name = f.split('/')[-1]
		fs.download(f, f"{CHECKPOINT}/{name}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

model = PopMusicTransformer(
		checkpoint=CHECKPOINT,
		is_training=False)

@app.get('/generate')
async def generate(n_target_bar: int = 5, temperature: float = 1.2, topk: int = 5, key: str = None, with_chords: bool = True):
	words = model.generate(n_target_bar, temperature, topk, key_prompt=key)
	events = utils.word_to_event(words, model.word2event)
	if with_chords:
		events = utils.add_played_chords(events)
	words = [model.event2word['{}_{}'.format(e.name, e.value)] for e in events]

	path = f'responses/{str(datetime.now())}.mid'
	utils.write_midi(words, model.word2event, path)

	with open(path, 'rb') as f:
		res = f.read()

	buffer = []
	for c in res:
		buffer.append(int(c))

	eventStrings = [f'{e.name}_{e.value}' for e in events]
	return JSONResponse({
		'buffer': buffer,
		'events': eventStrings
	}, background=BackgroundTask(lambda: os.remove(path)))


@app.get('/download')
async def download(events: List[str] = Query(...)):
	path = f'responses/{str(datetime.now())}.mid'
	events = [utils.Event(name, None, value, None) for name, value in [e.split('_') for e in events]]
	utils.write_midi(None, None, path, events)
	return FileResponse(path, background=BackgroundTask(lambda: os.remove(path)))
