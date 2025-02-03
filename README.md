# 🎨 Image-to-Speech GenAI Tool Using LLM

## 📚 Overview
The **Image-to-Speech GenAI Tool** is an AI-powered application that transforms an uploaded image into a narrated short story. It leverages **Hugging Face models, OpenAI GPT, and Streamlit** to provide an automated image-to-audio experience.

## 🚀 Features
- **🖼️ Image to Text**: Generates a caption from an uploaded image using the **BLIP Image Captioning model**.
- **✍️ Text to Story**: Converts the caption into a creative short story using **OpenAI's GPT-4o**.
- **🔊 Story to Speech**: Uses **ESPnet VITS** to generate speech audio from the story.
- **🌍 Web Interface**: Built with **Streamlit**, offering a user-friendly experience.
- **⚡ Fully Automated**: No manual intervention required.

## 📌 How It Works
1. **Upload an image** (`.jpg` format only).
2. The tool **generates a caption** using **BLIP**.
3. The caption is **expanded into a short story** using **GPT-4o**.
4. The story is **converted into speech** using **ESPnet VITS**.
5. Users can **listen to or download** the generated audio.

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PujaAmmineni/Image-to-Speech-GenAI-Tool-Using-LLM.git
cd Image-to-Speech-GenAI-Tool-Using-LLM
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys
Create a `.env` file in the project directory and add:
```ini
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_TOKEN=your_huggingface_api_token
```
Replace with your actual API keys.

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

## 🎮 Usage Guide
- Open **http://localhost:8501/** in your browser.
- Upload an image.
- View the **generated caption and story**.
- Click the **Play** button to listen to the story.
- Download the audio if needed.

## 🏗 Technology Stack
- **Python 3.10**
- **Streamlit** (for UI)
- **Hugging Face Transformers** (for BLIP & ESPnet models)
- **OpenAI GPT** (for text generation)
- **PyTorch** (for ML model execution)
- **Requests** (for API calls)

## ❓ Troubleshooting
### 🖥 CUDA Not Available?
If PyTorch runs on CPU instead of GPU:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ⏳ Hugging Face Model Takes Too Long to Load?
Try an alternative captioning model:
```python
from transformers import pipeline
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
```
## 🔮 Future Enhancements
- **Multi-language support** for text-to-speech.
- **Improved captioning accuracy** with enhanced models.
- **Deployment on Streamlit Cloud and Hugging Face Spaces**.


## 🌟 Acknowledgments
Special thanks to:
- **Hugging Face** for AI models.
- **OpenAI** for text generation.
- **Streamlit** for making UI development simple.


🔗 **GitHub Repository**: [Image-to-Speech-GenAI-Tool-Using-LLM](https://github.com/PujaAmmineni/Image-to-Speech-GenAI-Tool-Using-LLM)

