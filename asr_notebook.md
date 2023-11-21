---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "yzjZqUC1f3Qq"}

### Install Required Packages

```{code-cell}
:id: EsIGLexKgTKV

# Update Unix Package ffmpeg to version 4
!add-apt-repository -y ppa:jonathonf/ffmpeg-4
!apt update
!apt install -y ffmpeg
```

```{code-cell}
:id: t5MDUMlagdc0

# Install additional python packages we will be using
!pip install datasets>=2.6.1
!pip install git+https://github.com/huggingface/transformers
!pip install librosa
!pip install evaluate>=0.30
!pip install jiwer
```

+++ {"id": "OI6WYuLoiXrJ"}

### Load the dataset

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 00q4ZFQriJbP
outputId: 62a25360-1fab-41bf-cc54-b8456fec3e3b
---
from datasets import load_dataset

# Load the Afrispeech dataset
dataset = load_dataset("tobiolatunji/afrispeech-200", "all", streaming=True)
print("Number of examples", len(dataset))

# Access the filtered examples
for example in dataset:
    print(example)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: JHTjUP2njS0A
outputId: f85da81f-e30e-4d9e-ac6b-5c20cf52bc49
---
#display metadata
dataset_head = dataset['train'].take(2)
list(dataset_head)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 110
id: Ptzg_uBXjT-k
outputId: 5a25e7b1-3ed3-4c47-a959-7f2ed7699cec
---
# display audio sample and transcript
sample =list(dataset_head)[1]
audio = sample["audio"]

print(sample["transcript"])
ipd.Audio(data=audio["array"], autoplay=False, rate=audio["sampling_rate"])
```

+++ {"id": "XDXjshxZjvnf"}

### Dropping some of the unused columns

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: jMu5OqjXj0I5
outputId: 73773029-fd8f-415e-8938-4f48c6752cb0
---
# drop unused columns
dataset_cln = dataset.remove_columns(['speaker_id', 'path', 'age_group', 'gender', 'accent', 'domain', 'country', 'duration'])
print(dataset_cln)
```

+++ {"id": "d4nt4RllokGM"}

### Prepare Feature Extractor, Tokenizer and Data

- A feature extractor pre-processes the raw audio inputs
- The model performs the sequence-to-sequence mapping
- A tokenizer post-processes the model outputs to text format

+++ {"id": "naH3zHqypmeg"}

load the feature extractor from the pre-trained checkpoint with the default values:

```{code-cell}
:id: Qu1GUCMjoeup

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
```

+++ {"id": "P69rDmUmpuPd"}

#### load WhisperTokenizer

The Whisper model outputs a sequence of token ids. The tokenizer maps each of these token ids to their corresponding text string. For the English language, we can load the pre-trained tokenizer and use it for fine-tuning without any further modifications. We simply have to specify the target language and the task. These arguments inform the tokenizer to prefix the language and task tokens to the start of encoded label sequences:

```{code-cell}
:id: 1V9Nmo0xpfFM

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
```

+++ {"id": "qHlO3m8LqF3O"}

#### Combine To create A WhisperProcessor

To simplify using the feature extractor and tokenizer, we can wrap both into a single WhisperProcessor class. This processor object inherits from WhisperFeatureExtractor and WhisperProcessor, and can be used on the audio inputs and model predictions as required. In doing so, we only need to keep track of two objects during training: the processor and the model:

```{code-cell}
:id: EDtFjKxzqEHc

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
```

+++ {"id": "PAb2Fr5QrJLy"}

### Data Preparation

Lets print the first example of the AfriSpeech dataset to see what form the data is in:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: co4Bc0izrEXI
outputId: 10b2cdb5-b2e7-4784-9c5c-2d9c8c7b41ee
---
print(next(iter(dataset_cln["train"])))
```

+++ {"id": "yW7TEuZRrrVT"}

Since our input audio is sampled at 44100Hz, we need to downsample it to 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model.

We'll set the audio inputs to the correct sampling rate using dataset's cast_column method. This operation does not change the audio in-place, but rather signals to datasets to resample audio samples on the fly the first time that they are loaded:

```{code-cell}
:id: ls_6eoXmzJWs

from datasets import Audio
dataset_cln = dataset_cln.cast_column("audio", Audio(sampling_rate=16000))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: cNXWhKxSzMRx
outputId: 4e83f94d-40d9-48c4-aaa4-04c02c647997
---
print(next(iter(dataset_cln["train"])))
```

+++ {"id": "5ID8X1Gzzseh"}

## A function to prepare our data for the model:

We first provide set constants for the maximum duration of audio in seconds, the maximum input length, and the maximum label length. The maximum input length and label length are important for defining the constraints and processing requirements for our audio data and its corresponding transcriptions within the model.

We then load the resampled audio data by calling batch["audio"].
We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
We encode the transcriptions to label ids through the use of the tokenizer.

```{code-cell}
:id: BlQKOSs3apxG

# Pre-processing to suit the max audio length required by Whisper
MAX_DURATION_IN_SECONDS = 30.0
MAX_INPUT_LENGTH = MAX_DURATION_IN_SECONDS * 16000
MAX_LABEL_LENGTH = 448

# Add these functions to your code
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    batch["input_length"] = len(batch["audio"])
    batch["labels_length"] = len(tokenizer(batch["transcript"], add_special_tokens=False).input_ids)
    return batch
def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < input_length < MAX_INPUT_LENGTH
def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < MAX_LABEL_LENGTH
```

```{code-cell}
:id: u9QmkFvma4d_

dataset_cln = dataset_cln.map(prepare_dataset)
dataset_cln = dataset_cln.filter(filter_inputs, input_columns=["input_length"])
dataset_cln = dataset_cln.filter(filter_labels, input_columns=["labels_length"])
dataset_cln = dataset_cln.remove_columns(['labels_length', 'input_length'])
```

```{code-cell}
:id: kfKc27YLzdQn

val_dataset = dataset_cln['train'].take(3000)
train_dataset = dataset_cln['train'].skip(3000)

dataset_cln['train'] = train_dataset
dataset_cln['validation'] = val_dataset
```

```{code-cell}
:id: YH9chxGVzdNe

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
```

```{code-cell}
:id: GgIWr5bvzdJj

#  initialising the data collator defined above:
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

```{code-cell}
:id: gAz0yWibzdFn

import evaluate
metric = evaluate.load("wer")
```

```{code-cell}
:id: l3FcjuYdzdBz

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
```

```{code-cell}
:id: w0z9a2dYzc-D

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
```

```{code-cell}
:id: QVM5EKjOzc5A

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
```

```{code-cell}
:id: SgQMhGA92qp7

!pip install accelerate -U
```

```{code-cell}
:id: 83x1IeD1zcte

# Training Args
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./dsn_afrispeech",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    resume_from_checkpoint=True,
)
```

```{code-cell}
:id: mP88IkMGzbDb

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_cln['train'],
    eval_dataset=dataset_cln['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
```

```{code-cell}
:id: QnIcSltpzMNa

processor.save_pretrained(training_args.output_dir)
```

```{code-cell}
:id: 5WkCeCWKzMHT

# empty the GPU
torch.cuda.empty_cache()
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 832
id: jb8-UERnI74A
outputId: 6337fefe-39ec-4f43-ac58-32829db58986
---
trainer.train()
```

+++ {"id": "oIspY5mjSOHW"}

###Saving the Model

```{code-cell}
:id: DTWoaz57SLOt

trainer.save_model(training_args.output_dir)
```

```{code-cell}
:id: uVmqrmU6SXGD

processor.save_pretrained("./path_to_saved_model")
```

+++ {"id": "cOF9myHoSa70"}

#Push model to HF Hub
The training results can now be uploaded to the Hub.

```{code-cell}
:id: qzU2TCL9SYjI

kwargs = {
    "dataset_tags": "tobiolatunji/afrispeech-200",
    "dataset": "AfriSpeech",
    "dataset_args": "config: en, split: test",
    "language": "en",
    "model_name": "Whisper Small En",
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}
```
