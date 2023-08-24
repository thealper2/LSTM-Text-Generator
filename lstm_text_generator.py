import tensorflow as tf
import numpy as np
import argparse
import colorama
import os

from colorama import Fore, Style, Back
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

colorama.init(autoreset=True)

def build_lstm(vocab_size, embedding_dim, rnn_units, batch_size):
	model = Sequential([
		Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
		LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
		Dense(vocab_size)
	])
	model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
	return model

def train(file, checkpoint_dir, rnn_units, batch_size, buffer_size, embedding_dim, seq_length, epochs, start_string, temperature, generate):
	data = open(file, "rb").read().decode(encoding="utf-8")
	vocab = sorted(set(data))
	vocab_size = len(vocab)

	char2idx = {c:i for i, c in enumerate(vocab)}
	idx2char = np.array(vocab)
	text_as_int = np.array([char2idx[i] for i in data])

	example_per_epoch = len(data) // (seq_length + 1)

	char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
	sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
	dataset = sequences.map(lambda chunk: (chunk[:-1], chunk[1:]))
	dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

	model = build_lstm(vocab_size, embedding_dim, rnn_units, batch_size)

	checkpoint_prefix = os.path.join(checkpoint_dir, "lstm_{epoch}")
	checkpoint_callback = ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

	history = model.fit(dataset, epochs=epochs, callbacks=checkpoint_callback)

	model = build_lstm(vocab_size, embedding_dim, rnn_units, batch_size=1)
	model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
	model.build(tf.TensorShape([1, None]))

	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval, 0)

	text_generated = []

	model.reset_states()
	for i in range(generate):
		preds = model(input_eval)
		preds = tf.squeeze(preds, 0)
		preds = preds / temperature
		pred_id = tf.random.categorical(preds, num_samples=1)[-1, 0].numpy()
		input_eval = tf.expand_dims([pred_id], 0)
		text_generated.append(idx2char[pred_id])

	print(f"{Fore.GREEN}" + start_string + "".join(text_generated) + f"{Fore.RESET}")

if __name__ == "__main__":
	argument_parser = argparse.ArgumentParser(description="desc")
	argument_parser.add_argument("--file", required=True, help="File")
	argument_parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint dir")
	argument_parser.add_argument("--rnn_units", default=1024, required=False, help="RNN Units")
	argument_parser.add_argument("--batch_size", default=64, required=False, help="Batch Size")
	argument_parser.add_argument("--buffer_size", default=1000, required=False, help="Buffer Size")
	argument_parser.add_argument("--embedding_dim", default=256, required=False, help="Embedding Dim")
	argument_parser.add_argument("--epochs", default=30, required=False, help="Epochs")
	argument_parser.add_argument("--start_string", default="a", required=False, help="Start string")
	argument_parser.add_argument("--generate", default=100, required=False, help="Generate")
	argument_parser.add_argument("--temperature", default=1.0, required=False, help="Temperature")
	argument_parser.add_argument("--seq_length", default=100, required=False, help="Sequence Length")
	arguments = argument_parser.parse_args()

	if (arguments.file and arguments.checkpoint_dir):
		train(
			file=arguments.file,
			checkpoint_dir=arguments.checkpoint_dir,
			rnn_units=arguments.rnn_units,
			batch_size=arguments.batch_size,
			buffer_size=arguments.buffer_size,
			embedding_dim=arguments.embedding_dim,
			epochs=arguments.epochs,
			start_string=arguments.start_string,
			generate=arguments.generate,
			temperature=arguments.temperature,
			seq_length=arguments.seq_length
		)
