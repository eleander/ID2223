from datasets import load_dataset, IterableDatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer

def create_dataset():
    print("Starting creation of dataset")
    common_voice = IterableDatasetDict()

    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "es", split="train", token="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", streaming=True)
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_13_0", "es", split="test", token="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", streaming=True)
    print("Streams created")

    # Removed unecessery columns
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    print("Removed unncessery columns")

    # Normalize the audio to 16kHz
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    print("Normalize the audio to 16kHz")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="Spanish", task="transcribe")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice["train"].column_names)
    print("Applied the mapping")

    return common_voice