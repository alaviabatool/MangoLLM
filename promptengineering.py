#In this script, I have aligned the image and text. Aligned semantics using prompt engineering

import os
import torch
import gradio as gr
import fitz  # PyMuPDF
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# ======================== SETUP CONFIG ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "meta-llama/Llama-2-13b-chat-hf"

# ======================== 1. Load Llama 2 =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

llama_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

llama_pipe = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=tokenizer,
    device_map="auto"
)

print("âœ… Llama 2 loaded successfully!")

# ======================== 2. Load CNN Classifier =========================
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

classifier = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=len(class_labels))
state_dict = torch.load("/home/alavia/SkinGPT-4/weights/mango_classifier_v2_s.pth", map_location=device)
classifier.load_state_dict(state_dict)
classifier.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_mango(image_pil):
    image = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(image)
        _, predicted = outputs.max(1)
        return class_labels[predicted.item()]

# ======================== 3. Load PDF Text Data =========================
def extract_text_from_pdfs(folder_path):
    text = ""
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf = fitz.open(os.path.join(folder_path, file))
            for page in pdf:
                text += page.get_text()
            pdf.close()
    return text

pdf_folder_path = "/home/alavia/SkinGPT-4/mangodata/"
pdf_knowledge = extract_text_from_pdfs(pdf_folder_path)
print("âœ… PDF text data loaded!")

# ======================== 4. Multimodal QA Function =========================
def mango_advice_pipeline(image_pil, user_question):
    # Step 1: Classify mango
    mango_class = classify_mango(image_pil)

    # Step 2: Prepare prompt for Llama with mango class info + PDF text context
    context = pdf_knowledge[:4000]  # Use a slice if too long for prompt
    prompt = f"""
    Context Information (from mango export PDFs):
    {context}

    Mango Type/Condition Detected: {mango_class}

    User Question: {user_question}

    Give an expert answer on mango handling, disease management, or export processes as per the data.

    Answer:
    """

    response = llama_pipe(prompt, max_new_tokens=400, do_sample=True, temperature=0.7)
    return f"í ½í¿¢ Detected Class: **{mango_class}**\n\ní ½í³œ **Answer:**\n{response[0]['generated_text'].replace(prompt, '').strip()}"

# ======================== 5. Gradio Interface =========================
with gr.Blocks() as demo:
    gr.Markdown("# í ¾íµ­ Mango Multimodal Chat")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Mango Image")
            question_input = gr.Textbox(label="Your Question about this Mango", placeholder="What treatment is recommended for this mango?")

            submit_btn = gr.Button("Analyze Image & Get Answer")

        with gr.Column():
            output = gr.Markdown()

    submit_btn.click(fn=mango_advice_pipeline, inputs=[image_input, question_input], outputs=output)

demo.launch(server_name="0.0.0.0", server_port=5905, share=True)
