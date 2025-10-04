# heb-tts-benchmark

This repository contains the code for the Hebrew TTS benchmark using [Whisper IPA model](https://github.com/thewh1teagle/whisper-heb-ipa).

See [Hebrew TTS Benchmark](https://thewh1teagle.github.io/heb-tts-benchmark) for the live results.


See [Phonikud](https://phonikud.github.io) for more information.

## Add new model

1. Synthesis with TTS from the file `saspeech_100_gold_synthesis.csv` 

- either from the text/vocalized/phonemes columns
- make sure to save in new folder where each wav file name is the id from the csv file

2. Transcribe the wav files to IPA transcripts

```console
git clone https://github.com/thewh1teagle/heb-tts-benchmark
cd heb-tts-benchmark
uv run hf download --repo-type model thewh1teagle/whisper-heb-ipa-large-v3-turbo-ct2 --local-dir ./whisper-heb-ipa-ct2
uv run src/transcribe.py ./your-model-name ./transcripts # name the folder with wav files with the model name
uv run src/prepare_transcripts.py ./transcripts ./transcripts_clean
```

3. Evaluate the transcription

```console
uv run src/prepare_evaluation.py ./transcripts_clean ./transcripts_eval
```

4. Add the model to the summary
```console
uv run src/prepare_summary.py ./transcripts_eval ./web/summary.json
```

Now you can open ./web/index.html with live server to see the results. (or use `uv run python -m http.server 8000 -d ./web`)

5. Open new PR with your model results 🎉

# Ground truth

The GT is the file `saspeech_100_gold_gt.csv`.
Hand annotated IPA transcripts of random 100 samples from SASPEECH dataset.

## Gotchas

on macOS you may want to run the `transcribe.py` with `--compute-type int8_float32` to avoid errors.