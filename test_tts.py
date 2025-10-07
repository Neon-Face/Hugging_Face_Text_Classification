import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

# 1. Load a pre-trained TTS pipeline
# We'll use a model from Microsoft called 'speecht5_tts'
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

# 2. We need 'speaker embeddings' to give the voice a characteristic.
# We can get these from a special dataset.
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# 3. Synthesize speech
text_to_speak = "Hello, I am now ready to work on real-time spoken language models with tool-using capabilities."
speech = synthesiser(text_to_speak, forward_params={"speaker_embeddings": speaker_embedding})

# 4. Save the audio to a file
# The output is a NumPy array, we need to know the sampling rate.
sf.write("output_speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

print("Audio has been generated and saved to output_speech.wav!")