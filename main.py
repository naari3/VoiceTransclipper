import glob
import torch
from transformers import pipeline
from datasets import load_dataset, Audio
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm


generate_kwargs = {
    "language": "Japanese",
    "no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0,
}
pipe = pipeline(
    "automatic-speech-recognition",
    model="litagin/anime-whisper",
    device="cuda",
    torch_dtype=torch.float16,
    chunk_length_s=30.0,
    batch_size=64,
)

# globが渡された場合
files = glob.glob(r"F:\UVRs\ひだまりスケッチ\ひだまりスケッチ - 01_(Vocals) large-v2\*")

dataset = load_dataset(
    "audiofolder", data_files=files, split="train", cache_dir="./cache"
)

for i, data in enumerate(
    tqdm(
        pipe(
            KeyDataset(dataset, "audio"),
            generate_kwargs=generate_kwargs,
        )
    )
):
    print(data["text"])
