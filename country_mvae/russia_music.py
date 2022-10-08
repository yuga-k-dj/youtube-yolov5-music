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

from collections import defaultdict
from note_seq.midi_io import midi_file_to_note_sequence
from note_seq.midi_io import sequence_proto_to_pretty_midi

from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder


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

def note_sequence_to_tokens(seq) -> str:
    """ `note_sequence_to_tokens_for_M4L` converts magenta NoteSequence data
    to a single string. The string data (Token) is the sequence of chunk which
    represents single Note with 4 numbers formatted as below

    pitch[int] velocity[int] start_time(sec)[float] end_time(sec)[float],...,

    Args:
        seq (NoteSequence): single track NoteSequence instance

    Returns:
        str: Token
    """
    output_data1 = ""
    output_data2 = ""
    output_data3 = ""

    maped_output1 = None
    maped_output2 = None
    maped_output3 = None

    output_midi1 = defaultdict(list)
    output_midi2 = defaultdict(list)
    output_midi3 = defaultdict(list)

    for seq_note in seq.notes:
        
        if seq_note.instrument == 0:    
            start_time = seq_note.start_time * 1000
            end_time = seq_note.end_time * 1000
            output_midi1['notes'].append([seq_note.pitch, seq_note.velocity, '{:.2f}'.format(
                start_time), '{:.2f}'.format(end_time)])
            maped_output1 = map(
                str, sum(output_midi1['notes'], []))
            output_data1 = ' '.join(maped_output1)

        elif seq_note.instrument == 1:    
            start_time = seq_note.start_time * 1000
            end_time = seq_note.end_time * 1000
            output_midi2['notes'].append([seq_note.pitch, seq_note.velocity, '{:.2f}'.format(
                start_time), '{:.2f}'.format(end_time)])
            maped_output2 = map(
                str, sum(output_midi2['notes'], []))
            output_data2 = ' '.join(maped_output2)

        elif seq_note.instrument == 2:  
            start_time = seq_note.start_time * 1000
            end_time = seq_note.end_time * 1000
            output_midi3['notes'].append([seq_note.pitch, seq_note.velocity, '{:.2f}'.format(
                start_time), '{:.2f}'.format(end_time)])
            maped_output3 = map(
                str, sum(output_midi3['notes'], []))
            output_data3 = ' '.join(maped_output3)

    client = udp_client.UDPClient('127.0.0.1', 8500) # connect with max

    msg1 = OscMessageBuilder(address="/track22")
    msg1.add_arg(output_data1)
    m1 = msg1.build()

    msg2 = OscMessageBuilder(address="/track23")
    msg2.add_arg(output_data2)
    m2 = msg2.build()

    msg3 = OscMessageBuilder(address="/track24")
    msg3.add_arg(output_data3)
    m3 = msg3.build()

    print(m1, m2, m3)

    client.send(m1)
    client.send(m2)
    client.send(m3)

CONFIG_MAP = {}

# Trio Models
trio_16bar_converter = data.TrioConverter(
    steps_per_quarter=4,
    slice_bars=16,
    gap_bars=2)

CONFIG_MAP['hierdec-trio_16bar'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.SplitMultiOutLstmDecoder(
                core_decoders=[
                    lstm_models.CategoricalLstmDecoder(),
                    lstm_models.CategoricalLstmDecoder(),
                    lstm_models.CategoricalLstmDecoder()],
                output_depths=[
                    90,  # melody
                    90,  # bass
                    512,  # drums
                ]),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=bs,
            max_seq_len=msl,
            z_size=zs,
            enc_rnn_size=ers,
            dec_rnn_size=drs,
            free_bits=fb,
            max_beta=mb,
        )),
    note_sequence_augmenter=None,
    data_converter=trio_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

# BELOW MUSICVAE

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags_config='hierdec-trio_16bar'
num_outputs=1
checkpoint_file='hierdec-trio_16bar.tar'
mode='sample' # <-- mode selection

output_dir='/Users/yugakoba/Keio_Univ_SFC/CCLab-XMusic/DJ_Learning_AI/DJ_mixes_the_world/music_gen_with_vae/music/'

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
  # print(recv_msg)
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
        # print(
        #     'MusicVAE configs have very specific input requirements. Could not '
        #     'extract any valid inputs from `%s`. Try another MIDI file.' % path)
        sys.exit()
      elif len(tensors) > 1:
        basename = os.path.join(
            output_dir,
            '%s_input%d-extractions_%s-*-of-%03d.mid' %
            (flags_config, input_number, date_and_time, len(tensors)))
        for i, ns in enumerate(config.data_converter.from_tensors(tensors)):
          note_seq.sequence_proto_to_midi_file(
              # ns, basename.replace('*', '%03d' % i))
              # ns, output_dir + 'shibuya' + str(i) + str(recv_msg) + '.mid')
              ns, output_dir + 'russia' + str(i) + '.mid')
        # print(
        #     '%d valid inputs extracted from `%s`. Outputting these potential '
        #     'inputs as `%s`. Call script again with one of these instead.' %
        #     (len(tensors), path, basename))
        sys.exit()
    # logging.info(
    #     'Attempting to extract examples from input MIDIs using config `%s`...',
    #     flags_config)
    _check_extract_examples(input_1, FLAGS.input_midi_1, 1)
    _check_extract_examples(input_2, FLAGS.input_midi_2, 2)

  # logging.info('Loading model...')
  if FLAGS.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
        os.path.join(FLAGS.run_dir, 'train'))
  else:
    checkpoint_dir_or_path = os.path.expanduser(checkpoint_file)
  model = TrainedModel(
      config, batch_size=min(FLAGS.max_batch_size, num_outputs),
      checkpoint_dir_or_path=checkpoint_dir_or_path)

  if mode == 'interpolate': # <-- Here can be edited
    # logging.info('Interpolating...')
    _, mu, _ = model.encode([input_1, input_2])
    z = np.array([
        _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, num_outputs)])
    results = model.decode(
        length=config.hparams.max_seq_len,
        z=z,
        # temperature=FLAGS.temperature,
        temperature=recv_msg) # <-- temperature = FLAGS.temperature
  elif mode == 'sample':
    # logging.info('Sampling...')
    results = model.sample(
        n=num_outputs,
        length=config.hparams.max_seq_len,
        temperature=recv_msg) # <-- temperature = FLAGS.temperature or recv_msg

  basename = os.path.join(
      output_dir,
      '%s_%s_%s-*-of-%03d.mid' %
      (flags_config, mode, date_and_time, num_outputs))
  # logging.info('Outputting %d files as `%s`...', num_outputs, basename)
  for i, ns in enumerate(results):
    # note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))
    # note_seq.sequence_proto_to_midi_file(ns, output_dir + 'shibuya' + str(i) + str(recv_msg) + '.mid')
    note_seq.sequence_proto_to_midi_file(ns, output_dir + 'russia' + str(i) + '.mid')
  
  print(results)

  midi = midi_file_to_note_sequence(output_dir + 'russia' + str(i) + '.mid')
  note_sequence_to_tokens(midi)

  logging.info('Done.')


def socket_receive():
  print('Receive mode started')
  while True:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((socket.gethostname(), 5010))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        msg = s.recv(1024)
        print('Original Message = ' + str(msg))
        numbered_msg = int.from_bytes(msg, byteorder='big')
        use_msg = '0.' + str(numbered_msg)
        # numbered_msg = int(msg)
        print('Got Message =' + use_msg)
        run(CONFIG_MAP, float(use_msg))

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
