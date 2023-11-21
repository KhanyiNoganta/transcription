# Pre-processing and Configuration

![Alt Text](audio_waveform.gif)

The metadata encompasses various attributes for each data partition within the datasets. In the context of Automatic Speech Recognition (ASR), our focus was on transcripts and audio columns. Consequently, additional features are eliminated from being considered in the ASR task. The pre-processor holds significant importance within the Whisper pipeline, as it prepares the audio data for the ASR model. This preparation process plays a crucial role in enhancing transcription accuracy. Three pre-processing steps were performed, namely; audio resampling, data pre-processing, filtering and feature extraction.

## Audio resampling

The initial sampling rate of the dataset was set at 48KHz. In order to meet the required sampling rate of 16KHz, we performed down-sampling on the origional audio sampling rate

## Data Pre-processing

This procedure entails the conversion of the audio file into a format that is amenable to processing using the Automatic Speech Recognition (ASR) model. A pre-existing processor object from the Huggingface transformer library was utilized. The function of this processor involves the initial step of pre-processing the audio data, wherein it is transformed into input features. Additionally, the target data are tokenized, resulting in the creation of text labels. Subsequently, the training samples were processed to prepare our model, wherein all the data preparation techniques were applied uniformly across the training split.  During this stage, audio segments are transformed into spectrograms, and various preprocessing techniques, such as normalization or augmentation, are applied to the spectrograms as required. Subsequently, the text and audio columns were eliminated from the dataset, as the audio data had already undergone preprocessing to generate input features, and the text data were tokenized to create labels.

## Filtering

To mitigate the potential issues of audio sample truncation or memory errors, we implemented a filtering process to exclude audio samples exceeding 30s.

## Feature extraction

The feature extractor is a component used in various machine learning algorithms to transform raw data into a set of meaningful features that can be used for training. The spectrogram is employed as a graphical depiction of the audio signal's temporal and frequency characteristics, which are derived from the pre-processing of the audio data into a suitable format for model utilization. Additionally, normalized text was used to ensure textual consistency. The feature extractor performs two essential operations: first, it applies padding or truncation to a batch of audio samples to maintain a consistent input length, and second, it converts the padded audio arrays into log-Mel spectograms. The log-Mel spectograms are subsequently employed as input to the transformer encoder, which encodes the spectogram to generate a sequence of hidden encoder states. This process prepares the model to transcribe audio by establishing a connection between the features of the spectrogram and a sequence of text tokens, thereby enabling the model to transcribe unseen audio into text.