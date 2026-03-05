"""
Preprocess audio files into RVQ tokens.

Usage:
    python preprocess.py --input_dir ./data/audio --output_dir ./data/dac_tokens --split train
    python preprocess.py --input_dir ./data/audio_val --output_dir ./data/dac_tokens --split val

This converts raw audio (.wav, .mp3, .flac) into pre-tokenized .pt files
containing (n_codebooks, T) tensors of codebook indices.
Pre-tokenizing avoids running DAC during training.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm


def preprocess(
    input_dir: str,
    output_dir: str,
    split: str = "train",
    max_duration_sec: float = 24.0,
    device: str = "cuda",
    max_files: int = None,
    codec: str = "dac_44khz",
    encodec_bandwidth: float = 6.0,
):
    from src.data.tokenizer import DACTokenizer, Encodec24kTokenizer

    input_path = Path(input_dir)
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)
    root_output_path = Path(output_dir)
    root_output_path.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = sorted([f for f in input_path.rglob("*") if f.suffix.lower() in audio_exts])
    print(f"Found {len(files)} audio files in {input_dir}")

    if len(files) == 0:
        print("No audio files found! Check your input directory.")
        return

    if max_files is not None:
        files = files[:max_files]
        print(f"Using first {len(files)} files (--max_files={max_files})")

    # Initialize tokenizer.
    if codec == "dac_44khz":
        tokenizer = DACTokenizer(model_type="44khz", device=device)
    elif codec == "encodec_24khz":
        tokenizer = Encodec24kTokenizer(device=device, bandwidth=encodec_bandwidth)
    else:
        raise ValueError(f"Unsupported codec '{codec}'. Use dac_44khz or encodec_24khz.")

    codec_meta_path = root_output_path / "codec_meta.json"
    codec_meta_written = False

    success = 0
    errors = 0

    for audio_file in tqdm(files, desc=f"Tokenizing {split}"):
        try:
            codes = tokenizer.encode(
                str(audio_file),
                max_duration_sec=max_duration_sec,
            )

            if not codec_meta_written:
                codebook_size = int(tokenizer.codebook_size)
                codec_meta = {
                    "codec": codec,
                    "sample_rate": int(tokenizer.sample_rate),
                    "n_codebooks": int(codes.shape[0]),
                    "codebook_size": codebook_size,
                    "pad_token": codebook_size,
                    "bos_token": codebook_size + 1,
                    "eos_token": codebook_size + 2,
                    "vocab_size": codebook_size + 3,
                }
                if codec == "encodec_24khz":
                    codec_meta["encodec_bandwidth"] = float(encodec_bandwidth)
                with open(codec_meta_path, "w") as f:
                    json.dump(codec_meta, f, indent=2)
                codec_meta_written = True
                print(f"Wrote codec metadata to {codec_meta_path}: {codec_meta}")

            # Preserve directory structure to avoid basename collisions.
            rel_audio_path = audio_file.relative_to(input_path)
            out_file = (output_path / rel_audio_path).with_suffix(".pt")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(codes.cpu(), out_file)
            success += 1

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            errors += 1

    print(f"\nDone! {success} files tokenized, {errors} errors.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio to DAC tokens")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/dac_tokens")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_duration_sec", type=float, default=24.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--codec",
        type=str,
        default="dac_44khz",
        choices=["dac_44khz", "encodec_24khz"],
        help="Audio codec used to produce token IDs",
    )
    parser.add_argument(
        "--encodec_bandwidth",
        type=float,
        default=6.0,
        help="EnCodec target bandwidth in kbps (used when --codec encodec_24khz)",
    )
    args = parser.parse_args()

    preprocess(**vars(args))
