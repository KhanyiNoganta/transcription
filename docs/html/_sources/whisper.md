# Whisper Transformers

Whisper {cite}`radford2022robust` is an Automatic Speech Recognition (ASR) model that utilizes the transformer architecture {cite}`vaswani2017attention`. It belongs to a diverse group of transformer-based ASR models, which have been trained extensively on labeled audio transcription data. These models possess the remarkable ability to accurately transcribe speech without necessitating additional training on specific tasks or domains.

![Image Alt Text](whisper_architecture.png)

The framework's encoder-decoder architecture in the figure above effectively converts audio spectrogram features into corresponding text tokens. To acquire discrete speech units. The model strategically employs sampling from the Gumbel Softmax distribution. The key component of the Whisper model is its transformer encoder, which efficiently processes latent feature vectors using a series of transformer blocks. Impressively Whisper has demonstrated exceptional performance on well-known speech recognition benchmarks like LibriSpeech and Common Voice thus achieving state-of-the-art levels of accuracy.

![Image Alt Text](whisper_architecture_opp.png)

As shown in the figure above, the Whisper architecture is a straightforward end-to-end methodology , which is realized as an encoder-decoder Transformer. The audio input is segmented into 30-second intervals, transformed into a logarithmic Mel spectrogram, and subsequently fed into an encoder. The decoder is trained to generate the corresponding textual caption while incorporating special tokens that guide the single model to execute various tasks including language identification, phrase-level timestamps, multilingual speech transcription, and speech translation to English.