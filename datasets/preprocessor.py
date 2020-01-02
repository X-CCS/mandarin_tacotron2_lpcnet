import glob, os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from datasets import audio


# def build_from_path(hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
def build_from_path(hparams, input_dirs, wav_dir, mel_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		trn_files = glob.glob(os.path.join(input_dir, 'data', '*.trn'))
		for trn in trn_files:
			with open(trn) as f:
				basename = trn[:-4]
				if basename.endswith('.wav'):
					# THCHS30
					f.readline()
					wav_file = basename
				else:
					wav_file = basename + '.wav'
				wav_path = wav_file
				basename = basename.split('/')[-1]
				text = f.readline().strip()
		# with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:

		# 	for line in f:
		# 		parts = line.strip().split('|')
		# 		basename = parts[0]
		# 		wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
		# 		text = parts[2]
		# 		futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
				futures.append(executor.submit(partial(_process_utterance, wav_dir, mel_dir, basename, wav_path, text, hparams)))
				index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]


# def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
def _process_utterance(wav_dir, mel_dir, index, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#[-1, 1]
	quant = encode_mu_law(wav, mu=512)
	constant_values = 0.
	out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	# if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
	if mel_frames > hparams.max_mel_frames or len(text) > hparams.max_text_length:
		return None

	#Compute the linear scale spectrogram from the wav
	linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

	#Ensure time resolution adjustement between audio and mel-spectrogram
	fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
	l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

	#Zero pad for quantized signal
	out = np.pad(quant, (l, r), mode='constant', constant_values=constant_values)
	assert len(out) >= mel_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)

	# # Write the spectrogram and audio to disk
	# audio_filename = 'audio-{}.npy'.format(index)
	# mel_filename = 'mel-{}.npy'.format(index)
	# linear_filename = 'linear-{}.npy'.format(index)
	# # np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	filename = '{}.npy'.format(index)
	np.save(os.path.join(wav_dir, filename), quant.astype(np.int16), allow_pickle=False)
	np.save(os.path.join(mel_dir, filename), mel_spectrogram.T, allow_pickle=False)
	# np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	# return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)
	return (filename, time_steps, mel_frames, text)



def label_2_float(x, bits) :
	return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits) :
	assert abs(x).max() <= 1.0
	x = (x + 1.) * (2**bits - 1) / 2
	return x.clip(0, 2**bits - 1)


def encode_mu_law(x, mu) :
	mu = mu - 1
	fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
	return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True) :
	# TODO : get rid of log2 - makes no sense
	if from_labels : y = label_2_float(y, math.log2(mu))
	mu = mu - 1
	x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
	return x