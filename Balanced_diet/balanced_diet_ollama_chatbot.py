"""
Balanced Diet Chatbot using Ollama + Gradio
------------------------------------------
This educational demo recommends general meal ideas from a local CSV dataset.
It is not a medical diagnostic tool and does not replace advice from a doctor
or registered dietitian.

Requirements:
    pip install ollama gradio pandas

Before running:
    1. Install Ollama from https://ollama.com/download
    2. Open a terminal and download a model, for example:
           ollama pull qwen2.5:0.5b
       If that model is unavailable on your installation, try:
           ollama pull llama3.2:1b
    3. Ensure Ollama is running.
    4. Keep the CSV dataset in the same folder as this Python file.
    5. Run:
           python balanced_diet_ollama_chatbot.py
"""

from pathlib import Path
import os

import gradio as gr
import pandas as pd
import ollama

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "balanced_diet_organic_millets_nuts_fruits_dataset.csv"

# You can override this in your terminal:
# Windows CMD: set OLLAMA_MODEL=qwen2.5:0.5b
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b-instruct")

FALLBACK_DATA = [
    {
        "health_category": "General healthy adult",
        "meal_time": "Breakfast",
        "food_group": "Millets + protein",
        "food_suggestion": "Little millet idli with sambar and cucumber",
        "guidance": "Choose varied foods and adjust portions to your needs",
        "avoid_or_limit": "Frequent sugary drinks and highly processed snacks",
    },
    {
        "health_category": "Type 2 diabetes",
        "meal_time": "Breakfast",
        "food_group": "Millets + protein",
        "food_suggestion": "A moderate portion of ragi dosa with sambar and vegetables",
        "guidance": "Carbohydrate portions and medicines matter; follow your care team's advice",
        "avoid_or_limit": "Sugary drinks and very large grain portions",
    },
    {
        "health_category": "High blood pressure",
        "meal_time": "Snack",
        "food_group": "Fruit + unsalted nuts",
        "food_suggestion": "Guava or orange with a small portion of unsalted nuts",
        "guidance": "Prefer lower-sodium choices; ask a clinician about any potassium restrictions",
        "avoid_or_limit": "Excess salt and salty packaged snacks",
    },
    {
        "health_category": "Thyroid condition (general)",
        "meal_time": "Breakfast",
        "food_group": "Balanced meal",
        "food_suggestion": "Idli or dosa with sambar",
        "guidance": "Thyroid needs vary by diagnosis and medication; follow clinician instructions",
        "avoid_or_limit": "Unprescribed high-dose iodine or kelp supplements",
    },
]


def load_dataset():
    """Load the CSV dataset; use a small fallback if it is missing."""
    if DATASET_PATH.exists():
        try:
            df = pd.read_csv(DATASET_PATH).fillna("")
            return df
        except Exception as exc:
            print(f"Could not read CSV ({exc}); using built-in sample data.")
    else:
        print(f"Dataset not found at {DATASET_PATH}; using built-in sample data.")
    return pd.DataFrame(FALLBACK_DATA)


DATA = load_dataset()


def check_ollama():
    """Return a friendly status message about the local Ollama connection."""
    try:
        response = ollama.list()
        # Current Ollama Python package returns objects with .models; older
        # versions may return dictionaries, so support both.
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])
        names = []
        for model in models or []:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            if name is None and isinstance(model, dict):
                name = model.get("model") or model.get("name")
            if name:
                names.append(name)

        if not names:
            return "Connected to Ollama, but no downloaded models were found. Run `ollama pull qwen2.5:0.5b`."
        if not any(name == MODEL_NAME or name.startswith(MODEL_NAME + ":") for name in names):
            return (
                f"Ollama is running, but model `{MODEL_NAME}` is not installed. "
                f"Available models: {', '.join(names)}. "
                f"Run `ollama pull {MODEL_NAME}` or set OLLAMA_MODEL to an installed model."
            )
        return f"Connected to Ollama. Model `{MODEL_NAME}` is available."
    except Exception as exc:
        return (
            "Cannot connect to Ollama. Start the Ollama app/service, then try again. "
            f"Technical detail: {exc}"
        )


def get_relevant_rows(category: str, meal_time: str, user_message: str) -> str:
    """Filter the dataset and return a compact context for the language model."""
    df = DATA.copy()
    category_col = "health_category" if "health_category" in df.columns else None
    meal_col = "meal_time" if "meal_time" in df.columns else None

    if category_col and category and category != "Any / general":
        mask = df[category_col].astype(str).str.contains(category, case=False, regex=False)
        # A broad match for common category labels.
        if category == "Diabetes":
            mask = df[category_col].astype(str).str.contains("diabetes", case=False, regex=False)
        elif category == "High blood pressure":
            mask = df[category_col].astype(str).str.contains("blood pressure", case=False, regex=False)
        elif category == "Thyroid condition":
            mask = df[category_col].astype(str).str.contains("thyroid", case=False, regex=False)
        elif category == "Healthy person":
            mask = df[category_col].astype(str).str.contains(
                "general healthy|healthy person", case=False, regex=True
            )
        df = df[mask]

    if meal_col and meal_time and meal_time != "Any meal":
        df = df[df[meal_col].astype(str).str.contains(meal_time, case=False, regex=False)]

    # If a filter returns no rows, provide examples from the full dataset instead.
    if df.empty:
        df = DATA.copy()

    # Send a small, relevant context to the model.
    columns = [
        col for col in [
            "health_category", "meal_time", "food_group", "food_suggestion",
            "organic_option", "portion_note", "guidance", "avoid_or_limit"
        ] if col in df.columns
    ]
    return df[columns].head(10).to_string(index=False)


SYSTEM_PROMPT = """
You are a cautious, friendly balanced-diet information assistant.
Use the provided dataset as inspiration for general meal ideas, especially Indian
foods, millets, nuts, fruits, pulses, and vegetables. Explain simply.

Safety rules:
- Do not diagnose, promise a cure, or claim a food treats diabetes, BP, or thyroid disease.
- Give general educational suggestions, not personalized medical prescriptions.
- For diabetes, mention that millet and fruit still contain carbohydrates and portions matter.
- For high blood pressure, emphasize limiting excess sodium; do not suggest potassium salt
  substitutes for people with kidney disease or certain medicines without clinician advice.
- For thyroid conditions, do not recommend stopping medication or taking iodine/kelp supplements.
  Medication timing must follow the prescriber's instructions.
- If the user is pregnant, a child, has kidney disease, food allergies, an eating disorder,
  severe symptoms, or takes relevant medication, recommend professional guidance.
- If asked about symptoms or emergencies, encourage appropriate medical care.
- Organic foods are optional; do not claim organic food automatically has more nutrition.
- Never invent exact nutrition numbers from this dataset.
- Keep the answer practical: offer a balanced meal idea and one short safety note.
"""


def reply(message, history, category, meal_time):
    if not message or not message.strip():
        return "Please type a question about balanced meals or food choices."

    context = get_relevant_rows(category, meal_time, message)
    prompt = (
        f"Selected health category: {category}\n"
        f"Selected meal: {meal_time}\n\n"
        f"Dataset examples:\n{context}\n\n"
        f"User question: {message}\n\n"
        "Answer in clear, simple English. If helpful, provide breakfast/lunch/dinner/snack ideas. "
        "Do not claim the dataset is a medical authority."
    )

    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.4},
        )
        # Current ollama package returns an object; support dictionary responses too.
        answer = getattr(getattr(result, "message", None), "content", None)
        if answer is None and isinstance(result, dict):
            answer = result.get("message", {}).get("content")
        if answer:
            return answer.strip()
        return "Ollama returned an empty response. Please try a shorter question."
    except Exception as exc:
        return (
            f"Unable to use Ollama model `{MODEL_NAME}`.\n\n"
            "Try these steps in Command Prompt:\n"
            "1. Run `ollama list` to see installed models.\n"
            f"2. Run `ollama pull {MODEL_NAME}` if the model is missing.\n"
            "3. Ensure the Ollama app is running.\n"
            "4. If you have another installed model, set OLLAMA_MODEL to that exact model name.\n\n"
            f"Technical detail: {exc}"
        )


with gr.Blocks(title="Balanced Diet Chatbot") as demo:
    gr.Markdown(
        "# 🌿 Balanced Diet Chatbot\n"
        "Ask for general meal ideas using millets, fruits, nuts, pulses, vegetables, "
        "and other foods. Powered by your local Ollama model."
    )
    with gr.Row():
        category_input = gr.Dropdown(
            choices=[
                "Any / general",
                "Healthy person",
                "Diabetes",
                "High blood pressure",
                "Thyroid condition",
                "High cholesterol",
                "Overweight management",
                "General digestive health",
            ],
            value="Any / general",
            label="Choose a health category",
        )
        meal_input = gr.Dropdown(
            choices=["Any meal", "Breakfast", "Lunch", "Dinner", "Snack"],
            value="Any meal",
            label="Meal time",
        )

    gr.Markdown("**Example:** Suggest a balanced breakfast for a person with diabetes.")
    gr.ChatInterface(
        fn=reply,
        additional_inputs=[category_input, meal_input],
        chatbot=gr.Chatbot(height=420),
        textbox=gr.Textbox(
            placeholder="Ask about a balanced meal...",
            lines=2,
            label="Your question",
        ),
    )

    status = gr.Button("Check Ollama connection")
    status_output = gr.Textbox(label="Ollama status", interactive=False)
    status.click(fn=check_ollama, inputs=[], outputs=status_output)

    gr.Markdown(
        "**Health note:** This app provides general educational information only. "
        "It cannot diagnose or treat medical conditions. Ask a qualified healthcare "
        "professional for advice tailored to your health, medication, and lab results."
    )

if __name__ == "__main__":
    print(check_ollama())
    demo.launch()
