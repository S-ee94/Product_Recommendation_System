# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:27:03 2025

@author: seema
"""

# AI-Powered Product Recommendation System (Python - Grok API Edition)
# This version is tailored for the Grok (xAI) API. Works in Colab, Spyder, VS Code, Jupyter, etc.
# Just run this entire script.
# In Colab: paste into a cell and run.
# Locally: save as grok_product_recommender.py and run `python grok_product_recommender.py`

# Install required packages (run once)
# !pip install gradio openai -q    # Uncomment if needed

import gradio as gr
from openai import OpenAI

# ================== PRODUCT DATABASE ==================
# Edit/add products as needed
products = [
    {"name": "iPhone 15 Pro",          "price": 999,  "category": "smartphone", "specs": "A17 Pro chip, titanium body, 48MP camera, excellent video"},
    {"name": "Samsung Galaxy S24 Ultra", "price": 1199, "category": "smartphone", "specs": "S-Pen, 200MP camera, Snapdragon 8 Gen 3, AI features"},
    {"name": "Google Pixel 8 Pro",     "price": 999,  "category": "smartphone", "specs": "Best Android camera, Tensor G3, clean software, 7 years updates"},
    {"name": "OnePlus 12",             "price": 799,  "category": "smartphone", "specs": "Very fast charging, smooth 120Hz display, great performance"},
    {"name": "Nothing Phone (2)",      "price": 599,  "category": "smartphone", "specs": "Unique glyph LED design, clean software, good price"},
    {"name": "Samsung Galaxy A54",     "price": 449,  "category": "smartphone", "specs": "Great mid-range, excellent battery, IP67"},
    {"name": "Moto G Power 5G (2024)",  "price": 299,  "category": "smartphone", "specs": "Huge battery, budget price, wireless charging"},
    {"name": "MacBook Air M3",         "price": 1099, "category": "laptop",     "specs": "Fanless, incredible battery, sharp Retina display"},
    {"name": "Dell XPS 14",            "price": 1499, "category": "laptop",     "specs": "Gorgeous OLED display, premium build, great keyboard"},
    {"name": "Lenovo ThinkPad X1 Carbon", "price": 1399, "category": "laptop",  "specs": "Business legend, best keyboard, very durable"},
    {"name": "Sony WH-1000XM5",        "price": 399,  "category": "headphones","specs": "Industry-leading noise cancelling, 30hr battery"},
    {"name": "AirPods Pro 2",          "price": 249,  "category": "headphones","specs": "Best for iPhone users, spatial audio, ANC"},
    {"name": "Anker Soundcore Liberty 4", "price": 129, "category": "headphones","specs": "Excellent value, good ANC, long battery"},
]

# Product list as string for the prompt
product_list_str = "\n".join([
    f"• {p['name']} - ${p['price']} - {p['category']} - {p['specs']}"
    for p in products
])

def get_recommendations(user_preference, api_key, model_choice):
    if not api_key.strip():
        return "⚠️ Please enter your Grok API key."

    # Initialize OpenAI-compatible client for xAI Grok API
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"  # xAI's base URL
    )

    # Map user choice to actual Grok model name (updated as of Nov 2025)
    model_map = {
        "grok-4 (flagship)": "grok-4",
        "grok-4-fast-reasoning": "grok-4-fast-reasoning",
        "grok-4-1-fast-reasoning (latest)": "grok-4-1-fast-reasoning",
        "grok-code-fast-1": "grok-code-fast-1",
    }
    model = model_map.get(model_choice, "grok-4-1-fast-reasoning")  # Default to latest fast reasoning model

    prompt = f"""
You are an expert product recommendation assistant powered by Grok.
Only recommend products that are in the list below. 
Do NOT recommend anything that is not in this exact list.

Available products:
{product_list_str}

User request: {user_preference}

Return the top 3-5 best matches (or fewer if not many match).
For each recommendation write:
- Product name
- Price
- Why it matches the user's request (2-3 short sentences max)

If no product matches well, say "Sorry, nothing in the current product list matches your criteria very well."

Be friendly, concise, and helpful.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}. Check your API key and model availability at https://x.ai/api."

# ================== GRADIO UI ==================
with gr.Blocks(title="Grok AI Product Recommender") as demo:
    gr.Markdown("# 🎯 Grok-Powered Product Recommendation System")
    gr.Markdown("Enter your preferences (e.g., *'phone under $500 with good camera'* or *'laptop for programming around $1200'*). Grok will recommend from the product list below using the xAI API.")

    with gr.Row():
        with gr.Column(scale=2):
            preference_input = gr.Textbox(
                label="Your preferences / request",
                placeholder="I want a smartphone under $600 with long battery life...",
                lines=3
            )
        with gr.Column(scale=1, min_width=200):
            api_key_input = gr.Textbox(
                label="Grok (xAI) API Key",
                placeholder="xai-... (get at console.x.ai)",
                type="password"
            )
            model_dropdown = gr.Dropdown(
                choices=[
                    "grok-4-1-fast-reasoning (latest)",
                    "grok-4-fast-reasoning",
                    "grok-4 (flagship)",
                    "grok-code-fast-1",
                ],
                value="grok-4-1-fast-reasoning (latest)",
                label="Grok Model"
            )

    gr.Markdown("### Available Products")
    gr.Markdown(product_list_str)

    recommend_btn = gr.Button("Get Grok Recommendations", variant="primary")

    output = gr.Markdown()

    recommend_btn.click(
        fn=get_recommendations,
        inputs=[preference_input, api_key_input, model_dropdown],
        outputs=output
    )

    gr.Markdown("### Get Started")
    gr.Markdown("""
- Sign up for API access: [xAI Console](https://console.x.ai)
- Generate your key: In the console dashboard.
- Pricing: Starts at $0.20/M input tokens (see models above). Details: [xAI Docs](https://docs.x.ai/docs/models)
- Questions? Check [xAI API](https://x.ai/api)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,  # True for public link in Colab
        debug=False,
        server_port=7860
    )

