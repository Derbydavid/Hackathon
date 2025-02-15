import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Load the trained model and processor
model_path = r"./results"  # Update if your model is saved elsewhere
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Emotion label mapping (update with your dataset's emotions)
label_map = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fear",
    5: "disgust",
    6: "surprise"
}

# Function to preprocess the audio file
def preprocess_audio(audio_path, processor, max_length=32000):
    speech, sr = librosa.load(audio_path, sr=16000)
    
    # Truncate or pad the audio
    speech = np.pad(speech, (0, max_length - len(speech)), 'constant') if len(speech) < max_length else speech[:max_length]

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return inputs

# Provide your custom audio file path
audio_path = r"C:\Users\Derby\Hackathon\nitesh_happy_wav.wav"  # Replace with your actual file path
inputs = preprocess_audio(audio_path, processor)

# Move inputs to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Perform emotion prediction
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted label
logits = outputs.logits
predicted_id = torch.argmax(logits, dim=-1).item()

# Convert predicted label to actual emotion name
predicted_emotion = label_map.get(predicted_id, "Unknown")

print(f"\nPredicted Emotion: {predicted_emotion}")

# Get probabilities for all emotions
probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
emotion_probabilities = {label_map[i]: prob for i, prob in enumerate(probabilities)}

# Sort and display all emotions with probabilities
sorted_emotions = sorted(emotion_probabilities.items(), key=lambda x: x[1], reverse=True)

print("\nEmotion Probabilities:")
for emotion, prob in sorted_emotions:
    print(f"{emotion}: {prob:.4f}")
