# heb-tts-benchmark

This repository contains the code for the Hebrew TTS benchmark.


See phonikud.github.io for more information.

## Add new model

1. Synthesis with TTS frm the file `saspeech_100_gold_synthesis.csv` 

- either from the text/vocalized/phonemes columns
- make sure to save in new folder where each wav file name is the id from the csv file

2. Transcribe the wav files to IPA transcripts

```console
uv run hf download --repo-type model thewh1teagle/whisper-heb-ipa-ct2 --local-dir ./whisper-heb-ipa-ct2
uv run src/transcribe.py ./your-model-name ./transcripts
```

3. Evaluate the transcription

```console
uv run src/prepare_evaluation.py ./transcripts ./transcripts_eval
```

4. Add the model to the summary
```console
uv run src/prepare_summary.py ./transcripts_eval ./web/summary.json
```

Now you can open ./web/index.html with live server to see the results.

5. Open new PR with your model results 🎉

# Ground truth

The GT is the file `saspeech_100_gold_gt.csv`.
Hand annotated IPA transcripts of random 100 samples from SASPEECH dataset.