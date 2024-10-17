import argparse
import os
import sys
import time
from pathlib import Path
import pip
import numpy as np

import librosa
import soundfile
from tqdm import tqdm


try:
    import onnxruntime as ort
except ModuleNotFoundError:
    pip.main(['install', "onnxruntime"])
    import onnxruntime as ort
    
base = Path(os.path.dirname(__file__))
onnx_dir = base / "data/onnx"
onnx_filenames = filter(lambda filename: filename.endswith(".onnx"), os.listdir(onnx_dir), )
default_sample_rate = 16_000

model_names = [".".join(filename.split(".")[:-1]) for filename in onnx_filenames]
parser = argparse.ArgumentParser(prog="clean-unet", description="Remove noise from audio files.")

parser.add_argument(
    "--model", "-m",
    help="Model name",
    choices=model_names,
    default="dns-large-full",
)
parser.add_argument(
    "--batch", "-b",
    help="Batch size",
    default=1,
    type=int,
)
parser.add_argument(
    "--output-sample-rate", "-s",
    help="File output sample rate",
    default=16_000,
    type=int,
)
parser.add_argument(
    "--provider", "-p",
    help="ONNX provider. Use CPUExecutionProvider for CPU computation or CUDAExecutionProvider for NVIDIA-based GPU",
    default="CPUExecutionProvider",
)
parser.add_argument(
    "--verbose", "-v",
    action="store_true",
)
parser.add_argument(
    "--input",
    "-i",
    action="append",
    nargs="?",
)
parser.add_argument(
    "output",
    nargs="*",
)

args = parser.parse_args()

if not args.input:
    print("Error: No input files given.", file=sys.stderr)
    exit(-1)

if len(args.input) != len(args.output):
    print("Error: The number of input files must match number of output files.", file=sys.stderr)
    exit(-1)

ort_sess = ort.InferenceSession(onnx_dir / f"{args.model}.onnx", providers=[
    "CPUExecutionProvider",
])

files = list(zip(args.input, args.output))

for batch_index in tqdm(range(0, len(files), args.batch)):
    subset_filenames = files[batch_index:min(batch_index + args.batch, len(files))]

    input_audio_data = [librosa.load(in_filename, sr=default_sample_rate)[0] for in_filename, _ in subset_filenames]

    # Pad files to longest so they share all the same dimension size
    max_sample = max([len(data) for data in input_audio_data])
    padded_input_data = np.stack([np.pad(data, (0, max_sample - len(data))) for data in input_audio_data])

    t = time.time()
    output_data = ort_sess.run(None, {"noisy_audio_16khz": padded_input_data})
    if args.verbose:
        print(f"Processing time: {time.time() - t:.2f}s, Audio duration: {max_sample / default_sample_rate:.2f}s")

    resized_output_data = [np.resize(data, (input_audio_data[i].shape[0],)) for i, data in enumerate(list(output_data[0]))]

    for out_file_data, subset_filename in zip(resized_output_data, subset_filenames):
        out_filename = subset_filename[1]

        # Resample if the chosen output sample rate is different from 16000
        if args.output_sample_rate != default_sample_rate:
            out_file_data = librosa.resample(out_file_data, orig_sr=default_sample_rate, target_sr=args.output_sample_rate)

        soundfile.write(out_filename, out_file_data, args.output_sample_rate)

if args.verbose:
    print("Done")
