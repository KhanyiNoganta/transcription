---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "YcaRGcNBvcVb"}

# ASR Tool Access

The tool can be accessed from the following link [kanyekuthi-asr-tool](https://huggingface.co/spaces/kanyekuthi/kanyekuthi-dsn_afrispeech) and the model information can be found [kanyekuthi-model-information](https://huggingface.co/kanyekuthi/dsn_afrispeech)

The tool can also be accessed from the transformers library using the example script below:

```{code-cell} ipython3
:id: zvbc8oxzttnN

import torch
from transformers import pipeline
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 53
id: N_J5HS9PtxfM
jupyter:
  outputs_hidden: true
outputId: e06cd33b-d101-43f5-ec9e-82553dca811a
---
%%capture
pipe = pipeline("automatic-speech-recognition", model="kanyekuthi/dsn_afrispeech")
```

## Examples

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: FKoHCsZEt33D
outputId: 4ca64131-00c1-4e81-c2cb-e507b7dfa959
---
# audio file 1 (Human)
transcribe = pipe("asr.mp4",chunk_length_s=30)
print(transcribe["text"][:500])
```

Hello, so we just like to use this audio for testing purposes. We are using it to test the transcription tool that Kanyi has developed.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: tl4YPvK0w6wP
outputId: 2253ce0f-0fea-4936-d544-7a7ec049b62b
---
# audio file 2 (Human)
transcribe = pipe("whatsapp_test.ogg",chunk_length_s=30)
print(transcribe["text"][:500])
```

Hey this is just a test audio to basically just check if the transcription tool is working. I was wondering if we can use this just to see what the transcription would look like. Let me know if it works it needs to be shorter than 30 seconds. Thank you.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: UvIy5HQ-xEed
outputId: c28dbf13-b6a6-4b01-a49c-eb4bc1737d9e
---
# audio file 3 (Automated Voice)
transcribe = pipe("automatic_speech.mp3",chunk_length_s=30)
print(transcribe["text"][:500])
```

Automatic speech recognition means converting spoken language to written text allowing for textual analysis, search and processing of spoken content.
