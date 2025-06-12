#####USING EfficientNetV2s CNN classifier for image classification and Llama-2-13b-chat for text generation

import timm
import torch.nn as nn
import gradio as gr
import os
import fitz  # PyMuPDF
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# ========================
# ✅ 1. Setup: Llama 2 13B Chat
# ========================
model_id = "meta-llama/Llama-2-13b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
print("✅ Llama 2 13b chat loading in 4bit  successfully!")
# ========================
# ✅ 2. Setup: Image Classifier
# ========================
class_labels = [
    "Alternaria",
    "Anthracnose",
    "Black Mould Rot",
    "Healthy",
    "Stem and Rot",
    "Mango 1",
    "Mango Red 1",
    "Anwar Ratool",
    "Chaunsa (Black)",
    "Chaunsa (Summer Bahisht)",
    "Chaunsa",
    "Dosehri",
    "Fajri",
    "Langra",
    "Sindhri"
]

# Load the EfficientNetV2-S model from timm
classifier = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=len(class_labels))

# Load your trained weights (.pth file)
state_dict = torch.load("/home/alavia/SkinGPT-4/weights/mango_classifier.pth", map_location="cpu")
classifier.load_state_dict(state_dict)

classifier.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_mango_image(image_pil):
    image = transform(image_pil).unsqueeze(0).to(device)
   # image = image.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = classifier(image)
        _, predicted = outputs.max(1)
        predicted_class = class_labels[predicted.item()]
    return predicted_class
print("✅ EfficientNet loaded and prediction block ran successfully!")
# ========================
# ✅ 3. Setup: PDF Extraction
# ========================
def extract_text_from_pdfs(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            doc = fitz.open(pdf_path)
            for page in doc:
                all_text += page.get_text()
            doc.close()
    return all_text

# 👉 setting path
pdf_folder_path = "/home/alavia/SkinGPT-4/mangodata/"

mango_text_data = extract_text_from_pdfs(pdf_folder_path)
print("✅ Mango PDF data extracted successfully!")
# ========================
# ✅ 4. Text Generation Function (with context)
# ========================
def generate_answer(user_input):
    context = mango_text_data[:1500]  # Use first 1500 characters as context
    prompt = f"{context}\n\n### Human: {user_input}\n### Assistant:"
    output = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, "").strip()
# ======== TEXT ONLY CHAT FUNCTION ========
def mango_text_chat(user_input):
    prompt = f"""
    Question: {user_input}
    Answer:
    """
    output = pipe(prompt, max_new_tokens=300, do_sample=False)
    return output[0]['generated_text']

# ======== IMAGE + CHAT FUNCTIONS (Placeholder - replace with your own logic) ========
def process_image_and_chat(image, user_input):
    # Dummy processing for now; Replace with your model's actual vision-based output
    prompt = f"""
    Question: {user_input}
    Answer:
    """
    output = pipe(prompt, max_new_tokens=300, do_sample=False)
    return output[0]['generated_text']

# ================= GRADIO APP =====================

with gr.Blocks(title="Mango 13b Chat") as demo:
    gr.Markdown("<h1 align='center'>🍋 Mango 13b Chat 🍋</h1>")

    with gr.Tabs():
        # ==== Text Chat Tab ====
        with gr.Tab("📝 Questions based on texts only"):
            gr.Markdown("Chat with Mango 13b Chat about export guides, export inspection & certifications, packaging specifications, preclearance program, cultivation, mango diseases. Answers provided from the official documentation from Pakistan, USA, India & Mexico. ")
            text_input = gr.Textbox(label="Ask something about mango best practices:", placeholder="e.g., What type of hot water treatment is required for mangoes?")
            text_output = gr.Textbox(label="Response")
            text_button = gr.Button("Generate Response")

            text_button.click(mango_text_chat, inputs=text_input, outputs=text_output)

        # ==== Image Upload Chat Tab ====
        with gr.Tab("🖼️ Upload Image & Chat"):
            gr.Markdown("Upload an image of mango fruit and chat about diseases.")
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Mango Image")
                image_text_input = gr.Textbox(label="Ask about this mango image:", placeholder="Describe any symptoms here...")
            image_output = gr.Textbox(label="Response")
            image_button = gr.Button("Analyze Image & Respond")

            image_button.click(process_image_and_chat, inputs=[image_input, image_text_input], outputs=image_output)

demo.launch(server_name="0.0.0.0", server_port=5905, share=True)
