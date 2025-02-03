import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import torch
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from transformers import pipeline

# ✅ Load environment variables
dotenv_path = "Envfiles.env"
load_dotenv(dotenv_path)

# ✅ Get API keys
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API Key not found! Set OPENAI_API_KEY in Envfiles.env.")

if not HUGGINGFACE_API_TOKEN:
    raise ValueError("❌ Hugging Face API Key not found! Set HUGGINGFACE_API_TOKEN in Envfiles.env.")

# ✅ Check PyTorch installation
if not torch.cuda.is_available():
    print("⚠️ CUDA is not available. Running on CPU.")

# ✅ Custom UI styling
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #333;
        }
        .upload {
            text-align: center;
            padding: 20px;
        }
        .expander {
            border-radius: 10px;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 10px;
            background-color: #fff;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        audio {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

def progress_bar(duration: int) -> None:
    """Displays a progress bar for a given duration."""
    progress_text = "⏳ Generating content, please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(1, 101):
        time.sleep(duration / 100)
        my_bar.progress(percent_complete, text=progress_text)
    
    my_bar.empty()

def generate_text_from_image(uploaded_file) -> str:
    """Uses BLIP model to generate text from an uploaded image file."""
    try:
        # ✅ Save the uploaded image
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ✅ Open image using PIL
        image = Image.open(file_path).convert("RGB")

        # ✅ Process with BLIP model
        image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        result = image_to_text(image)  # Pass image object, not path
        
        generated_text = result[0]["generated_text"] if result else "No text generated."
        os.remove(file_path)  # ✅ Remove temp file
        
        return generated_text
    except Exception as e:
        return "⚠️ Error generating text from image."

def generate_story_from_text(scenario: str) -> str:
    """Generates a short story using GPT-4o (or GPT-4 fallback)."""
    try:
        prompt_template = """
        You are a talented storyteller. Create a short story (max 50 words) based on the following scenario:
        CONTEXT: {scenario}
        STORY:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["scenario"])
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.9, openai_api_key=OPENAI_API_KEY)
        
        # ✅ Use RunnableLambda to ensure the response is a plain string
        story_llm = RunnableLambda(lambda input: llm.invoke(prompt.format(scenario=input["scenario"])))
        generated_story = story_llm.invoke({"scenario": scenario})

        # ✅ Convert to plain string if needed
        if isinstance(generated_story, list):
            generated_story = " ".join(generated_story)
        elif not isinstance(generated_story, str):
            generated_story = str(generated_story)

        return generated_story.strip()
    except Exception as e:
        return f"⚠️ API Error: {e} - Check OpenAI API key access."

def generate_speech_from_text(message: str) -> str:
    """Generates speech from text using Hugging Face's ESPnet model."""
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": message}

    for attempt in range(3):  # ✅ Try up to 3 times
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            with open("generated_audio.flac", "wb") as file:
                file.write(response.content)
            return "generated_audio.flac"

        elif "currently loading" in response.text:
            estimated_time = response.json().get("estimated_time", 10)  # Default 10 sec
            print(f"🔄 Model is still loading, retrying in {estimated_time} seconds...")
            time.sleep(estimated_time)
        
        else:
            return None

    return "⚠️ Model is still loading, please try again later."

def main() -> None:
    """Main Streamlit application."""
    st.markdown("<h1 class='title'>📸 Image-to-Story Converter</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image (JPG)", type="jpg")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # ✅ Process image
        progress_bar(3)
        scenario = generate_text_from_image(uploaded_file)
        story = generate_story_from_text(scenario)
        audio_file = generate_speech_from_text(story)

        # ✅ Display results
        with st.expander("🖼️ **Generated Image Scenario**", expanded=True):
            st.write(scenario)
        
        with st.expander("📖 **Generated Short Story**", expanded=True):
            st.write(story)

        if audio_file:
            st.audio(audio_file)

if __name__ == "__main__":
    main()
