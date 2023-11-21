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

+++ {"id": "yzjZqUC1f3Qq"}

# ASR Notebook

+++ {"id": "OI6WYuLoiXrJ"}

### Load the dataset

```{code-cell} ipython3
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

+++ {"id": "XDXjshxZjvnf"}

### Dropping some of the unused columns

```{code-cell} ipython3
# drop unused columns
dataset_cln = dataset.remove_columns(['speaker_id', 'path', 'age_group', 'gender', 'accent', 'domain', 'country', 'duration'])
```

+++ {"id": "d4nt4RllokGM"}

### Prepare Feature Extractor, Tokenizer and Data

- A feature extractor pre-processes the raw audio inputs
- The model performs the sequence-to-sequence mapping
- A tokenizer post-processes the model outputs to text format

+++ {"id": "naH3zHqypmeg"}

load the feature extractor from the pre-trained checkpoint with the default values:

```{code-cell} ipython3
:id: Qu1GUCMjoeup

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
```

+++ {"id": "P69rDmUmpuPd"}

#### load WhisperTokenizer

The Whisper model outputs a sequence of token ids. The tokenizer maps each of these token ids to their corresponding text string. For the English language, we can load the pre-trained tokenizer and use it for fine-tuning without any further modifications. We simply have to specify the target language and the task. These arguments inform the tokenizer to prefix the language and task tokens to the start of encoded label sequences:

```{code-cell} ipython3
:id: 1V9Nmo0xpfFM

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
```

+++ {"id": "qHlO3m8LqF3O"}

#### Combine To create A WhisperProcessor

To simplify using the feature extractor and tokenizer, we can wrap both into a single WhisperProcessor class. This processor object inherits from WhisperFeatureExtractor and WhisperProcessor, and can be used on the audio inputs and model predictions as required. In doing so, we only need to keep track of two objects during training: the processor and the model:

```{code-cell} ipython3
:id: EDtFjKxzqEHc

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
```

+++ {"id": "PAb2Fr5QrJLy"}

### Data Preparation

Since our input audio is sampled at 44100Hz, we need to downsample it to 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model.

We'll set the audio inputs to the correct sampling rate using dataset's cast_column method. This operation does not change the audio in-place, but rather signals to datasets to resample audio samples on the fly the first time that they are loaded:

```{code-cell} ipython3
:id: ls_6eoXmzJWs

from datasets import Audio
dataset_cln = dataset_cln.cast_column("audio", Audio(sampling_rate=16000))
```

+++ {"id": "5ID8X1Gzzseh"}

## A function to prepare our data for the model:

We first provide set constants for the maximum duration of audio in seconds, the maximum input length, and the maximum label length. The maximum input length and label length are important for defining the constraints and processing requirements for our audio data and its corresponding transcriptions within the model.

We then load the resampled audio data by calling batch["audio"].
We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
We encode the transcriptions to label ids through the use of the tokenizer.

```{code-cell} ipython3
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

```{code-cell} ipython3
:id: u9QmkFvma4d_

dataset_cln = dataset_cln.map(prepare_dataset)
dataset_cln = dataset_cln.filter(filter_inputs, input_columns=["input_length"])
dataset_cln = dataset_cln.filter(filter_labels, input_columns=["labels_length"])
dataset_cln = dataset_cln.remove_columns(['labels_length', 'input_length'])
```

```{code-cell} ipython3
:id: kfKc27YLzdQn

val_dataset = dataset_cln['train'].take(3000)
train_dataset = dataset_cln['train'].skip(3000)

dataset_cln['train'] = train_dataset
dataset_cln['validation'] = val_dataset
```

```{code-cell} ipython3
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

```{code-cell} ipython3
:id: GgIWr5bvzdJj

#  initialising the data collator defined above:
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

```{code-cell} ipython3
:id: gAz0yWibzdFn

import evaluate
metric = evaluate.load("wer")
```

```{code-cell} ipython3
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

```{code-cell} ipython3
:id: w0z9a2dYzc-D

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
```

```{code-cell} ipython3
:id: QVM5EKjOzc5A

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
```

```{code-cell} ipython3
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

```{code-cell} ipython3
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

```{code-cell} ipython3
:id: QnIcSltpzMNa

processor.save_pretrained(training_args.output_dir)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 832
id: jb8-UERnI74A
outputId: 6337fefe-39ec-4f43-ac58-32829db58986
---
trainer.train()
```
