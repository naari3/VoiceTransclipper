from __future__ import annotations
import csv
import itertools

from faster_whisper import WhisperModel
import fire

from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent

import os
import glob
import errno

from yt_dlp.utils import sanitize_filename


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
    model: WhisperModel,
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
    for i, chunk in enumerate(chunks):
        out_file_path = f"{output_dir}/{basename_no_ext}_out{format(i + 1, fmt)}.wav"
        st = voice_range[i][0]
        ed = voice_range[i][1]
        chunk.export(out_file_path, format="wav")

        segments, info = model.transcribe(
            out_file_path, language=language, initial_prompt=initial_prompt
        )
        out_text = ""

        for j, segment in enumerate(segments):
            # 効果音・異常値判定
            if segment.no_speech_prob > no_speech_prob_thresh:
                out_text = no_speech_label
                break
            else:
                if j == 0:
                    out_text = segment.text
                else:
                    out_text += " " + segment.text

        out_text = out_text.strip('"')

        # 効果音・異常値判定
        if len(out_text) * 1000 / (ed - st) > hallucination_thresh:
            out_text = no_speech_label

        if len(out_text) > filename_len_max:
            out_text = out_text[:filename_len_max] + "…"

        result = [i + 1, out_text, st, ed]
        result_list.append(result)

        dirname = os.path.dirname(out_file_path)
        # 音声ファイルリネーム
        new_filename = sanitize_filename(f"{format(i + 1, fmt)}_{out_text}.wav")
        new_filename = os.path.join(dirname, new_filename)
        os.rename(out_file_path, new_filename)

        pro_bar = ("=" * ((i + 1) * 20 // file_num)) + (
            " " * (20 - (i + 1) * 20 // file_num)
        )
        print("\r[{0}] {1}/{2} {3}".format(pro_bar, i + 1, file_num, out_text), end="")

    print("")

    return result_list


def export_result_csv(csv_file_path, result_list):
    header = ["id", "text", "start", "end"]
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(result_list)


def cli(
    file_path: str,
    min_silence_len: int = 200,
    silence_thresh: int = -40,
    keep_silence: int = 500,
    language: str = "ja",
    no_speech_label: str = "(SE)",
    filename_len_max: int = 80,
    hallucination_thresh: int = 20,
    no_speech_prob_thresh: float = 0.95,
    initial_prompt: str = "",
):
    model_size = "turbo"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # globが渡された場合
    files = glob.glob(file_path)

    # フォルダ内に.wavファイルがない場合
    if len(files) == 0:
        # ファイルがない場合
        if not os.path.exists(file_path):
            print("指定したファイルが見つかりません。FILE_PATHの値を確認してください")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
        files.append(file_path)

    print("音声ファイル数:", len(files))
    for filepath in files:
        print("\t-", os.path.basename(filepath))

    for filepath in files:
        dirname = os.path.dirname(filepath)
        basename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
        print("Filename:", os.path.basename(filepath))
        chunks, voice_range = clip_wav_file_to_chunks(
            filepath, min_silence_len, silence_thresh, keep_silence
        )
        results = transcribe_chunks(
            chunks=chunks,
            voice_range=voice_range,
            model=model,
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


def main():
    fire.Fire(cli)


if __name__ == "__main__":
    main()
