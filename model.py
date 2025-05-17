# Imports for VITS model
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from phonemizer.backend.espeak.wrapper import EspeakWrapper

if sys.platform == "darwin":
    _ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)

import commons
import utils
from text import text_to_sequence
from text.symbols import symbols
from models import SynthesizerTrn

# Helper functions/classes for VITs model
class VitsModel:
    def __init__(self, hparams_path, checkpoint_path, sample_rate=16000, cuda=False):
        self.cuda_flag = cuda
        self.hps = utils.get_hparams_from_file(hparams_path)
        self.model = self.load_model(hparams_path, checkpoint_path, cuda)
        self.model.eval()
        self.resampler = torchaudio.transforms.Resample(self.hps.data.sampling_rate, sample_rate)

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def generate_speech(self, txt, speaker_id=0, noise_bounds=(0.667, 0.667), duration_bounds=(1.0, 1.0)):
        stn_tst = self.get_text(txt)
        noise_scale = np.random.uniform(noise_bounds[0], noise_bounds[1])
        duration_scale = np.random.uniform(duration_bounds[0], duration_bounds[1])
        with torch.no_grad():
            try:
                if self.cuda_flag:
                    x_tst = stn_tst.cuda().unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                    sid = torch.LongTensor(speaker_id).cuda()
                    audio = \
                        self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=0.8,
                                         length_scale=duration_scale)[0][0, 0].data.cpu().float()
                else:
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                    sid = torch.LongTensor(speaker_id)
                    audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        sid=sid,
                        noise_scale=noise_scale,
                        noise_scale_w=0.8,
                        length_scale=duration_scale
                    )[0][0, 0].data.float()

                audio = self.resampler(audio)  # resample to 16khz
                audio = (audio * 32767).numpy().astype(np.int16)  # convert to 16-bit PCM format
            except AssertionError:
                audio = None

        return audio

    def load_model(self, hparams_path, checkpoint_path, cuda):
        net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model)
        if cuda:
            net_g.cuda()

        _ = net_g.eval()

        _ = utils.load_checkpoint(checkpoint_path, net_g, None)

        return net_g
