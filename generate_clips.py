# Imports and arguments

import argparse
import itertools as it
import math
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from scipy.io.wavfile import write
from tqdm import tqdm


# Helper functions
def generate_speaker_ids(N, n_speakers, n_mix=2):
    ids_original = [[i] for i in np.arange(0, n_speakers)]
    if N < n_speakers:
        ids = ids_original[0:N]
    else:
        max_per_speaker = math.ceil(N / n_speakers)
        ids_original = (ids_original * max_per_speaker)[0:N]
        ids_random = [np.random.randint(0, n_speakers, n_mix).tolist() for _ in range(N - len(ids_original))]
        ids = ids_original + ids_random
    return ids


# Build sentence variations
def get_random_variation(txt):
    # Get possible words
    positions = txt.split()
    words = [i.split("|") for i in positions]

    # Build variations
    variation = " ".join([random.choice(i) for i in words])
    return re.sub("\s+", " ", variation)


def generate_clips(
        text: Union[List[str], str],
        number: int,
        output_dir: str,
        noise_bounds=(0.667, 1.5),
        duration_bounds=(0.8, 1.2)
):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "vits-chinese-aishell3"))
    from model import VitsModel

    if isinstance(text, list):
        text = it.cycle(text)
    else:
        text = it.cycle([text])

    try:
        # Create output directory if it doesn't exist
        out_dir = Path(output_dir)
        if not out_dir.exists():
            os.mkdir(out_dir)

        enable_gpu = False
        if torch.cuda.is_available():
            print("Found CUDA, enable GPU")
            enable_gpu = True

        if enable_gpu == True:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Load VITS model
        print("Loading VITS model...")
        model = VitsModel(
            hparams_path=os.path.join(Path(__file__).parent.absolute(), "models/vits-chinese-aishell3/configs/baker_base.json"),
            checkpoint_path=os.path.join(Path(__file__).parent.absolute(), "models/vits-chinese-aishell3/pretrained_models/G_AISHELL.pth"),
            cuda=True if enable_gpu else False
        )

        # Get speaker ids and text for each generation
        ids = generate_speaker_ids(number, n_speakers=174, n_mix=5)
        texts = []
        for i in range(len(ids)):
            texts.append(next(text))

        # Generate audio
        sr = 16000

        cnt = 0
        for i, text in tqdm(zip(ids, texts), total=len(ids), desc="Generating clips"):
            audio = model.generate_speech(
                txt=text,
                speaker_id=i,
                noise_bounds=noise_bounds,
                duration_bounds=duration_bounds
            )

            # Save clips
            if audio is not None:
                write(os.path.join(output_dir, uuid.uuid4().hex + ".wav"), sr, audio)
                cnt += 1

        print(f"{cnt} clips generated!")
    finally:
        sys.path.pop(0)

def main():
    parser = argparse.ArgumentParser("Generates 16khz, 16-bit PCM, single channel synthetic speech to serve as training data for wakeword detection systems")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--text", metavar='', type=str, help="The text to generate")
    requiredNamed.add_argument("--N", metavar='', type=int, default=1, help="""How many total phrases to generate. Note that sometimes generation fails, so the total number of valid saved clips may be < N.""")
    requiredNamed.add_argument("--output_dir", metavar='', type=str, help="The target directory for the generated clips")
    parser.add_argument("--noise_bounds", nargs="+", type=float, default=[0.667, 1.4])
    parser.add_argument("--duration_bounds", nargs="+", type=float, default=[0.6, 1.2])
    args = parser.parse_args()
    generate_clips(
        text=args.text,
        number=args.N,
        output_dir=args.output_dir,
        noise_bounds=tuple(args.noise_bounds),
        duration_bounds=tuple(args.duration_bounds)
    )

if __name__ == '__main__':
    main()
