# voicetransclipper

```
uv run python .\src\voicetransclipper\cli.py E:\mazoku\uvr\voices\
```

- E-tumさんのやつを手元で動かした
- faster-whisper にした

```
❯ uv run .\src\voicetransclipper\cli.py --help
INFO: Showing help with the command 'cli.py -- --help'.

NAME
    cli.py

SYNOPSIS
    cli.py FILE_PATH <flags>

POSITIONAL ARGUMENTS
    FILE_PATH
        Type: 'str'

FLAGS
    -m, --min_silence_len=MIN_SILENCE_LEN
        Type: 'int'
        Default: 200
    -s, --silence_thresh=SILENCE_THRESH
        Type: 'int'
        Default: -40
    -k, --keep_silence=KEEP_SILENCE
        Type: 'int'
        Default: 500
    -l, --language=LANGUAGE
        Type: 'str'
        Default: 'ja'
    -n, --no_speech_label=NO_SPEECH_LABEL
        Type: 'str'
        Default: '(SE)'
    -f, --filename_len_max=FILENAME_LEN_MAX
        Type: 'int'
        Default: 80
    -h, --hallucination_thresh=HALLUCINATION_THRESH
        Type: 'int'
        Default: 20
    -i, --initial_prompt=INITIAL_PROMPT
        Type: 'str'
        Default: ''

NOTES     
    You can also use flags syntax for POSITIONAL ARGUMENTS
```
