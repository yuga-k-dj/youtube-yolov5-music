# Copyright 2022 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MusicVAE generation script."""

# TODO(adarob): Add support for models with conditioning.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import os
import sys
import time

import socket

from magenta.models.music_vae import TrainedModel
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf
from magenta.models.music_vae.base_model import MusicVAE
from magenta.models.music_vae import lstm_models

from magenta.common import merge_hparams
from magenta.contrib import training as contrib_training
from magenta.models.music_vae import data
from magenta.models.music_vae import data_hierarchical


HParams = contrib_training.HParams

bs=256
msl=256
zs=512
ers=[2048, 2048]
drs=[1024, 1024]
fb=256
mb=0.2

class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path', 'tfds_name'])):

  def values(self):
    return self._asdict()

Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
  config_dict = config.values()
  config_dict.update(update_dict)
  return Config(**config_dict)


CONFIG_MAP = {}

# 16-bar Melody Models
mel_16bar_converter = data.OneHotMelodyConverter(
    skip_polyphony=False,
    max_bars=100,  # Truncate long melodies before slicing.
    slice_bars=16,
    steps_per_quarter=4)

CONFIG_MAP['hierdec-mel_16bar'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.CategoricalLstmDecoder(),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

# BELOW MUSICVAE

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags_config="hierdec-mel_16bar"
num_outputs=1
checkpoint_file="hierdec-mel_16bar.tar"
mode="sample" # <-- mode selection

output_dir="/Users/yugakoba/Keio_Univ_SFC/CCLab-XMusic/DJ_Learning_AI/DJ_mixes_the_world/music_gen_with_vae/music"

flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
    'output_dir', '/tmp/music_vae/generated',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'config', None,
    'The name of the config to use.')
flags.DEFINE_string(
    'mode', 'sample',
    'Generate mode (either `sample` or `interpolate`).')
flags.DEFINE_string(
    'input_midi_1', None,
    'Path of start MIDI file for interpolation.')
flags.DEFINE_string(
    'input_midi_2', None,
    'Path of end MIDI file for interpolation.')
flags.DEFINE_integer(
    'num_outputs', 5, #　<-- This parameter
    'In `sample` mode, the number of samples to produce. In `interpolate` '
    'mode, the number of steps (including the endpoints).')
flags.DEFINE_integer(
    'max_batch_size', 8,
    'The maximum batch size to use. Decrease if you are seeing an OOM.')
flags.DEFINE_float(
    'temperature', 0.1, # <-- This parameter
    'The randomness of the decoding process.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(
      np.dot(np.squeeze(p0/np.linalg.norm(p0)),
             np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def run(config_map, recv_msg): #def run(config_map)
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  print(recv_msg)
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

  if FLAGS.run_dir is None == checkpoint_file is None:
    raise ValueError(
        'Exactly one of `--run_dir` or `--checkpoint_file` must be specified.')
  if output_dir is None:
    raise ValueError('`--output_dir` is required.')
  tf.gfile.MakeDirs(output_dir)
  if mode != 'sample' and mode != 'interpolate':
    raise ValueError('Invalid value for `--mode`: %s' % mode)

  if flags_config not in config_map:
    raise ValueError('Invalid config name: %s' % flags_config)
  config = config_map[flags_config]
  config.data_converter.max_tensors_per_item = None

  if mode == 'interpolate':
    if FLAGS.input_midi_1 is None or FLAGS.input_midi_2 is None:
      raise ValueError(
          '`--input_midi_1` and `--input_midi_2` must be specified in '
          '`interpolate` mode.')
    input_midi_1 = os.path.expanduser(FLAGS.input_midi_1)
    input_midi_2 = os.path.expanduser(FLAGS.input_midi_2)
    if not os.path.exists(input_midi_1):
      raise ValueError('Input MIDI 1 not found: %s' % FLAGS.input_midi_1)
    if not os.path.exists(input_midi_2):
      raise ValueError('Input MIDI 2 not found: %s' % FLAGS.input_midi_2)
    input_1 = note_seq.midi_file_to_note_sequence(input_midi_1)
    input_2 = note_seq.midi_file_to_note_sequence(input_midi_2)

    def _check_extract_examples(input_ns, path, input_number):
      """Make sure each input returns exactly one example from the converter."""
      tensors = config.data_converter.to_tensors(input_ns).outputs
      if not tensors:
        print(
            'MusicVAE configs have very specific input requirements. Could not '
            'extract any valid inputs from `%s`. Try another MIDI file.' % path)
        sys.exit()
      elif len(tensors) > 1:
        basename = os.path.join(
            output_dir,
            '%s_input%d-extractions_%s-*-of-%03d.mid' %
            (flags_config, input_number, date_and_time, len(tensors)))
        for i, ns in enumerate(config.data_converter.from_tensors(tensors)):
          note_seq.sequence_proto_to_midi_file(
              ns, basename.replace('*', '%03d' % i))
        print(
            '%d valid inputs extracted from `%s`. Outputting these potential '
            'inputs as `%s`. Call script again with one of these instead.' %
            (len(tensors), path, basename))
        sys.exit()
    logging.info(
        'Attempting to extract examples from input MIDIs using config `%s`...',
        flags_config)
    _check_extract_examples(input_1, FLAGS.input_midi_1, 1)
    _check_extract_examples(input_2, FLAGS.input_midi_2, 2)

  logging.info('Loading model...')
  if FLAGS.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
        os.path.join(FLAGS.run_dir, 'train'))
  else:
    checkpoint_dir_or_path = os.path.expanduser(checkpoint_file)
  model = TrainedModel(
      config, batch_size=min(FLAGS.max_batch_size, num_outputs),
      checkpoint_dir_or_path=checkpoint_dir_or_path)

  if mode == 'interpolate': # <-- Here can be edited
    logging.info('Interpolating...')
    _, mu, _ = model.encode([input_1, input_2])
    z = np.array([
        _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, num_outputs)])
    res
    ults = model.decode(
        length=config.hparams.max_seq_len,
        z=z,
        # temperature=FLAGS.temperature,
        temperature=recv_msg) # <-- temperature
  elif mode == 'sample':
    logging.info('Sampling...')
    results = model.sample(
        n=num_outputs,
        length=config.hparams.max_seq_len,
        temperature=FLAGS.temperature) # <-- temperature

  basename = os.path.join(
      output_dir,
      '%s_%s_%s-*-of-%03d.mid' %
      (flags_config, mode, date_and_time, num_outputs))
  logging.info('Outputting %d files as `%s`...', num_outputs, basename)
  for i, ns in enumerate(results):
    note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

  logging.info('Done.')


def socket_receive():
    # received message must be fixed (now the type may be byte. e.g. b'74')
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((socket.gethostname(), 7010))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            msg = s.recv(1024)
            print('Original Message =' + str(msg))
            numbered_msg = int.from_bytes(msg, byteorder='big')
            # numbered_msg = int(msg)
            print('Got Message =' + str(numbered_msg))
            run(CONFIG_MAP, msg)
  
        except ConnectionRefusedError:
            msg = 'Waiting...'
            print(msg)
            time.sleep(5)
            continue


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  socket_receive() #socket receive == True -> make MusicVAE, False -> nothing
  # run(CONFIG_MAP, msg)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
