from __future__ import annotations

from faster_whisper import WhisperModel
import fire

from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import librosa

import os
import glob
import errno

from yt_dlp.utils import sanitize_filename


# inputFilePathの音声ファイルを分割する
# 戻り値：分割後の音声ファイルのリスト
def clipWavFiles(
    inputFilePath: str, min_silence_len: int, silence_thresh: int, keep_silence: int
) -> list[str]:
    # ファイルが存在しない場合は終了
    if not os.path.exists(inputFilePath):
        raise FileNotFoundError

    dirname = os.path.dirname(inputFilePath)
    basenameNoExt = os.path.splitext(os.path.basename(inputFilePath))[0]

    # 出力先のフォルダ作成
    outputDirname = f"{dirname}/{basenameNoExt}"
    if os.path.exists(outputDirname) == False:
        os.mkdir(outputDirname)

    # 音声ファイル読み込み
    sound = AudioSegment.from_file(inputFilePath, format="wav")
    # ノーマライズ処理
    sound = effects.normalize(sound)
    # 無音部分で分割
    chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
    )

    fileNum = len(chunks)
    fmt = "0" + str(len(str(fileNum - 1)) + 1)
    wavFileList = []
    for i, chunk in enumerate(chunks):
        outFilePath = f"{outputDirname}/out_{format(i + 1, fmt)}.wav"
        wavFileList.append(outFilePath)
        chunk.export(outFilePath, format="wav")

        pro_bar = ("=" * ((i + 1) * 20 // fileNum)) + (
            " " * (20 - (i + 1) * 20 // fileNum)
        )
        print(
            "\r[{0}] {1}/{2} {3}".format(pro_bar, i + 1, fileNum, outFilePath), end=""
        )

    print("")
    return wavFileList


# wavFileListの音声ファイルをそれぞれ文字起こしする
# 戻り値：なし
def transcribeWavFiles(
    wavFileList: list[str],
    model: WhisperModel,
    no_voice_label: str = "(SE)",
    filename_len_max: int = 50,
):
    if len(wavFileList) == 0:
        raise FileNotFoundError

    fmt = "0" + str(len(str(len(wavFileList) - 1)) + 1)
    fileNum = len(wavFileList)
    for i, f in enumerate(wavFileList):
        # 効果音判定
        y, sr = librosa.load(f, sr=44100)
        fmin, fmax = 64, 520
        fo_pyin, voiced_flag, voiced_prob = librosa.pyin(y, fmin=fmin, fmax=fmax)

        # 効果音の場合は文字起こしなし
        if voiced_flag.sum() == 0:
            outputText = no_voice_label
        else:
            segments, info = model.transcribe(f, language="ja")
            # mel = log_mel_spectrogram(f)
            # # 30秒データに整形
            # segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(torch.float16)
            # # デコード
            # result = model.decode(segment)
            # # トークナイザ取得
            # tokenizer = get_tokenizer(
            #     multilingual=True, language="ja", task="transcribe"
            # )
            # # トークナイザのデコード
            # outputText = tokenizer.decode(result.tokens)
            outputText = ""
            for segment in segments:
                outputText += segment.text

        if len(outputText) > filename_len_max:
            outputText = outputText[:filename_len_max] + "…"

        dirname = os.path.dirname(f)

        # 音声ファイルリネーム
        newfilename = sanitize_filename(f"{format(i + 1,fmt)}_{outputText}")
        newname = os.path.join(dirname, f"{newfilename}.wav")
        os.rename(f, newname)

        pro_bar = ("=" * ((i + 1) * 20 // fileNum)) + (
            " " * (20 - (i + 1) * 20 // fileNum)
        )
        print("\r[{0}] {1}/{2} {3}".format(pro_bar, i + 1, fileNum, outputText), end="")
    print("")


def cli(
    file_path: str,
    min_silence_len: int = 200,
    silence_thresh: int = -40,
    keep_silence: int = 500,
    no_voice_label: str = "(SE)",
    filename_len_max: int = 50,
):
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # ファイルorフォルダがない場合
    if not os.path.exists(file_path):
        print("指定したファイルが見つかりません。FILE_PATHの値を確認してください")
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    # 複数ファイルがある場合
    files = glob.glob(f"{file_path}/*.wav")

    # フォルダ内に.wavファイルがない場合
    if len(files) == 0:
        files.append(file_path)

    print("音声ファイル数:", len(files))
    for filepath in files:
        print("\t-", os.path.basename(filepath))

    for filepath in files:
        print("Filename:", os.path.basename(filepath))
        wavFileList = clipWavFiles(
            filepath, min_silence_len, silence_thresh, keep_silence
        )
        transcribeWavFiles(wavFileList, model, no_voice_label, filename_len_max)


def main():
    fire.Fire(cli)


if __name__ == "__main__":
    main()
