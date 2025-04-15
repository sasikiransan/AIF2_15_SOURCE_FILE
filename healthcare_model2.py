import torch
import whisper
import ollama
import warnings
from transformers import AutoTokenizer, AutoModel
warnings.simplefilter("ignore")
# Load BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

# Load Whisper model once (for efficiency)
whisper_model = whisper.load_model("small")

# Transcribe audio using Whisper
def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error in transcription: {str(e)}"

# Generate BioBERT embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token representation

# Use Mistral for medication recommendation
def get_medication_recommendation(symptoms):
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": f"What are the best medications for {symptoms}?"}]
    )
    return response.get("message", {}).get("content", "No response received.")

# Adjust dosage based on age and weight
def adjust_dosage(age, weight, dosage):
    if age < 12:
        return dosage * 0.5
    elif weight < 50:
        return dosage * 0.75
    return dosage

# Main function
def main():
    print("Personalized Healthcare Model Running...")
    
    # Get user input
    symptoms = input("Enter your symptoms: ")
    age = int(input("Enter your age in years: "))
    weight = float(input("Enter your weight in kg: "))
    temperature = float(input("Enter your body temperature in Celsius: "))
    
    print(f"Symptoms: {symptoms}\nAge: {age} years\nWeight: {weight} kg\nTemperature: {temperature}°C")
    
    # Get BioBERT embedding
    embedding = get_embeddings(symptoms)
    print("BioBERT Embedding:", embedding)
    
    # Get medication recommendation
    medication = get_medication_recommendation(symptoms)
    print("Medication Recommendation:", medication)
    
    # Adjust dosage example
    final_dosage = adjust_dosage(age, weight, 500)
    print("Adjusted Dosage:", final_dosage)

# Run the script
if __name__ == "__main__":
    main()
