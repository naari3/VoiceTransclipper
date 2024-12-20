from __future__ import annotations
import csv
import itertools
import shutil
import time

import torch
from transformers import pipeline, Pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent

from audio_separator.separator import Separator

import os
import glob
import errno
from tqdm.auto import tqdm

from yt_dlp.utils import sanitize_filename

import fire

import logging
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# input_file_pathの音声ファイルを分割する
def clip_wav_file_to_chunks(
    input_file_path: str,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    margin_sec: int = 100,
) -> tuple[list[AudioSegment], list[list[int]]]:
    # 音声読み込み
    input_file_ext = os.path.splitext(input_file_path)[1]
    is_wav = None
    if input_file_ext == ".wav":
        is_wav = True
    sound = AudioSegment.from_file(input_file_path, format="wav" if is_wav else None)
    sound = effects.normalize(sound)

    # 無音部分を検出
    non_silent_range = detect_nonsilent(
        sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )

    if isinstance(keep_silence, bool):
        keep_silence = len(sound) if keep_silence else 0

    output_ranges = [
        [start - keep_silence, end + keep_silence] for (start, end) in non_silent_range
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end + next_start) // 2
            range_ii[0] = range_i[1]

    # 無音部分で分割
    chunks = [
        sound[max(start, 0) : min(end, len(sound))] for start, end in output_ranges
    ]

    audio_length = sound.duration_seconds * 1000
    file_num = len(chunks)

    # 余白追加処理
    voice_range = []
    for i, rng in enumerate(non_silent_range):
        st = rng[0]  # startTime
        ed = rng[1]  # endTime

        if i == 0:
            mid_time_s = 0
        else:
            mid_time_s = int((st + non_silent_range[i - 1][1]) / 2)

        if i == file_num - 1:
            mid_time_e = audio_length
        else:
            mid_time_e = int((ed + non_silent_range[i + 1][0]) / 2)

        st = max(st - margin_sec, mid_time_s)
        ed = min(ed + margin_sec, mid_time_e)

        voice_range.append([st, ed])

    return chunks, voice_range


# wavFileListの音声ファイルをそれぞれ文字起こしする
# 戻り値：なし
def transcribe_chunks(
    chunks: list[AudioSegment],
    voice_range: list[list[int]],
    # model: WhisperModel,
    pipe: Pipeline,
    generate_kwargs: dict,
    dirname: str,
    basename_no_ext: str,
    language: str,
    no_speech_label: str,
    filename_len_max: int,
    hallucination_thresh: int,
    initial_prompt: str,
    no_speech_prob_thresh: float,
):
    file_num = len(chunks)
    fmt = "0" + str(len(str(file_num - 1)) + 1)

    output_dir = f"{dirname}/{basename_no_ext}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 文字起こし処理
    result_list = []

    output_file_paths = []
    for i, chunk in enumerate(chunks):
        out_file_path = f"{output_dir}/{basename_no_ext}_out{format(i + 1, fmt)}.wav"
        chunk.export(out_file_path, format="wav")

        # segments, info = model.transcribe(
        #     out_file_path, language=language, initial_prompt=initial_prompt
        # )
        # out_text = ""

        # for j, segment in enumerate(segments):
        #     # 効果音・異常値判定
        #     if segment.no_speech_prob > no_speech_prob_thresh:
        #         out_text = no_speech_label
        #         break
        #     else:
        #         if j == 0:
        #             out_text = segment.text
        #         else:
        #             out_text += " " + segment.text
        output_file_paths.append(out_file_path)
    LOG.info(f"files: {len(output_file_paths)} {output_file_paths}")
    dataset = load_dataset(
        "audiofolder", data_files=output_file_paths, split="train", cache_dir="./cache"
    )

    for i, out in enumerate(
        tqdm(
            pipe(
                KeyDataset(dataset, "audio"),
                generate_kwargs=generate_kwargs,
            ),
            leave=False,
        )
    ):
        st = voice_range[i][0]
        ed = voice_range[i][1]
        out_text = out["text"]

        out_text = out_text.strip('"')

        # 効果音・異常値判定
        if len(out_text) * 1000 / (ed - st) > hallucination_thresh:
            out_text = no_speech_label

        if len(out_text) > filename_len_max:
            out_text = out_text[:filename_len_max] + "…"

        result = [i + 1, out_text, st, ed]
        result_list.append(result)

        out_file_path = output_file_paths[i]
        dirname = os.path.dirname(out_file_path)
        # 音声ファイルリネーム
        new_filename = sanitize_filename(f"{format(i + 1, fmt)}_{out_text}.wav")
        new_filename = os.path.join(dirname, new_filename)
        replace(out_file_path, new_filename)

    return result_list


def export_result_csv(csv_file_path, result_list):
    header = ["id", "text", "start", "end"]
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(result_list)


def replace(src: str, dst: str):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)


def audio_separate(file_path: str, separator: Separator):
    filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    file_dir = os.path.dirname(file_path)

    # https://github.com/fsspec/filesystem_spec/issues/838 の問題により、ファイル名に[]が含まれると一覧として認識されないため、除去
    vocals_file_path = (
        os.path.join(file_dir, f"{filename_without_ext}_vocals.wav")
        .replace("[", "")
        .replace("]", "")
    )
    instrumental_file_path = (
        os.path.join(file_dir, f"{filename_without_ext}_instrumental.wav")
        .replace("[", "")
        .replace("]", "")
    )

    files = separator.separate(
        file_path,
        {
            "Vocals": f"{filename_without_ext}_vocals.wav",
            "Instrumental": f"{filename_without_ext}_instrumental.wav",
        },
    )
    replace(files[0], instrumental_file_path)
    replace(files[1], vocals_file_path)
    return {
        "Vocals": vocals_file_path,
        "Instrumental": instrumental_file_path,
    }


def cli(
    file_path: str,
    audio_separation: bool = True,
    audio_separator_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    min_silence_len: int = 200,
    silence_thresh: int = -40,
    keep_silence: int = 500,
    language: str = "ja",
    no_speech_label: str = "(SE)",
    filename_len_max: int = 200,
    hallucination_thresh: int = 20,
    no_speech_prob_thresh: float = 0.95,
    initial_prompt: str = "",
):
    start = time.perf_counter()
    files = glob.glob(file_path)

    if len(files) == 0:
        if not os.path.exists(file_path):
            LOG.error(
                "指定したファイルが見つかりません。FILE_PATHの値を確認してください"
            )
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
        files.append(file_path)

    LOG.info("音声ファイル数:", len(files))
    for filepath in files:
        LOG.info(f"\t- {os.path.basename(filepath)}")

    vocal_file_paths = []
    if audio_separation:
        LOG.info("音声分離モデル読み込み中...")
        separator = Separator()
        separator.load_model(audio_separator_model)

        LOG.info("音声分離中...")
        for filepath in tqdm(files):
            outputs = audio_separate(filepath, separator)
            vocal_file_path = outputs["Vocals"]
            vocal_file_paths.append(vocal_file_path)
        del separator
        LOG.info("音声分離完了")
    else:
        vocal_file_paths = files

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

    for filepath in tqdm(vocal_file_paths):
        dirname = os.path.dirname(filepath)
        basename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
        LOG.info(f"Filename: {os.path.basename(filepath)}")
        LOG.info("音声ファイル分割中...")
        chunks, voice_range = clip_wav_file_to_chunks(
            filepath, min_silence_len, silence_thresh, keep_silence
        )
        LOG.info("音声ファイル文字起こし中...")
        results = transcribe_chunks(
            chunks=chunks,
            voice_range=voice_range,
            # model=model,
            pipe=pipe,
            generate_kwargs=generate_kwargs,
            dirname=dirname,
            basename_no_ext=basename_no_ext,
            language=language,
            no_speech_label=no_speech_label,
            filename_len_max=filename_len_max,
            initial_prompt=initial_prompt,
            no_speech_prob_thresh=no_speech_prob_thresh,
            hallucination_thresh=hallucination_thresh,
        )

        csv_file_path = f"{dirname}/{basename_no_ext}_result.csv"
        export_result_csv(csv_file_path, results)

    end = time.perf_counter()
    LOG.info(f"完了: {(end - start)/60:.2f}分")


def main():
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        fire.Fire(cli)


if __name__ == "__main__":
    main()
