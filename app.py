
# ============================================================
# GLOWGUIDE AI - FINAL VERSION
# ============================================================

import os
import joblib
import pandas as pd
import gradio as gr


# ============================================================
# 1. LOAD MODEL AND ENCODERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ml_model = joblib.load(
    os.path.join(BASE_DIR, "skincare_model.pkl")
)

label_encoders = joblib.load(
    os.path.join(BASE_DIR, "label_encoders.pkl")
)


# ============================================================
# 2. GROQ SETUP
# ============================================================

from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Please add your Groq API key in the environment variables."
    )

client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# 3. DEFAULT OPTIONAL VALUES
# ============================================================

DEFAULT_WATER = "1–2 L"
DEFAULT_SLEEP = "6–8 hrs"
DEFAULT_SUNSCREEN = "Sometimes"
DEFAULT_MAKEUP = "Rarely"
DEFAULT_ROUTINE = "Basic"


# ============================================================
# 4. SKINCARE REPORT GENERATOR
# ============================================================

def generate_skincare_report(
    age,
    skin,
    concern,
    budget,
    season,
    water,
    sleep,
    sunscreen,
    makeup,
    routine
):

    try:

        if age is None:
            return """
<div class="error-box">
Please enter your <b>Age</b> to continue.
</div>
"""

        if not skin:
            return """
<div class="error-box">
Please select your <b>Skin Type</b> to continue.
</div>
"""

        if not concern:
            return """
<div class="error-box">
Please select your <b>Skin Concern</b> to continue.
</div>
"""

        if not budget:
            return """
<div class="error-box">
Please select your <b>Budget</b> to continue.
</div>
"""

        if not season:
            return """
<div class="error-box">
Please select your <b>Season</b> to continue.
</div>
"""

        # Automatically derive sensitive skin
        sensitive = "Yes" if skin == "Sensitive" else "No"

        # Optional defaults
        water = water if water else DEFAULT_WATER
        sleep = sleep if sleep else DEFAULT_SLEEP
        sunscreen = sunscreen if sunscreen else DEFAULT_SUNSCREEN
        makeup = makeup if makeup else DEFAULT_MAKEUP
        routine = routine if routine else DEFAULT_ROUTINE

        # Encode inputs
        input_data = {
            "Age": age,

            "Skin_Type":
                label_encoders["Skin_Type"].transform([skin])[0],

            "Skin_Concern":
                label_encoders["Skin_Concern"].transform([concern])[0],

            "Budget":
                label_encoders["Budget"].transform([budget])[0],

            "Season":
                label_encoders["Season"].transform([season])[0],

            "Water_Intake":
                label_encoders["Water_Intake"].transform([water])[0],

            "Sleep_Duration":
                label_encoders["Sleep_Duration"].transform([sleep])[0],

            "Sunscreen_Usage":
                label_encoders["Sunscreen_Usage"].transform(
                    [sunscreen]
                )[0],

            "Makeup_Usage":
                label_encoders["Makeup_Usage"].transform(
                    [makeup]
                )[0],

            "Sensitive_Skin":
                label_encoders["Sensitive_Skin"].transform(
                    [sensitive]
                )[0],

            "Current_Routine":
                label_encoders["Current_Routine"].transform(
                    [routine]
                )[0]
        }

        input_df = pd.DataFrame([input_data])

        # ML prediction
        pred_class = ml_model.predict(input_df)[0]

        probabilities = ml_model.predict_proba(input_df)[0]

        confidence = probabilities.max() * 100

        predicted_routine = label_encoders[
            "Routine_Category"
        ].inverse_transform([pred_class])[0]

        # Groq prompt
        prompt = f"""

You are GlowGuide AI, a personalized skincare assistant.

Based on the user's profile and the Machine Learning
prediction, create a SHORT, practical and visually
easy-to-read skincare recommendation.

USER PROFILE
------------
Age: {age}
Skin Type: {skin}
Skin Concern: {concern}
Budget: {budget}
Season: {season}

Sensitive Skin:
Automatically determined from Skin Type = {sensitive}

OPTIONAL DETAILS
----------------
Water Intake: {water}
Sleep Duration: {sleep}
Sunscreen Usage: {sunscreen}
Makeup Usage: {makeup}
Current Routine: {routine}

MACHINE LEARNING RESULT
-----------------------
Predicted Routine: {predicted_routine}
Confidence: {confidence:.1f}%

Create ONLY these sections:

### 🌸 Why This Routine?
Give 2-3 short bullet points explaining why this
routine category fits the user's profile.

### ☀️ Morning Routine — 5–7 minutes
Give 4-5 simple numbered steps.
Include an approximate time for each step.

### 🌙 Night Routine — 5–7 minutes
Give 4-5 simple numbered steps.
Include an approximate time for each step.

### 🧴 Ingredients to Look For
Give 3-5 useful ingredients with short explanations.

### 🚫 Ingredients / Product Types to Be Careful With
Give 2-4 concise points.

### 🍎 Diet Recommendation
Suggest practical skin-supportive foods.
Mention when they can be consumed.
Do not claim food can cure skin conditions.

### 👩‍⚕️ Dermatologist Consultation
Explain when the user should consider seeing a dermatologist.
Do not diagnose the user.

### 💄 Makeup & Skincare Tips
Give 2-3 concise tips.

### 🌿 Lifestyle & Seasonal Tips
Give 3-4 concise tips.

### ⚠️ Precautions
Give 2-3 short precautions.

Keep the complete response under 550 words.
Use short sentences and bullet points.
Do not diagnose medical conditions.
Do not prescribe medication.
Do not claim to cure diseases.
Mention professional dermatology advice when appropriate.
Keep the tone elegant, friendly and practical.
"""

        completion = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": (
                        "You are GlowGuide AI, a friendly and "
                        "responsible skincare assistant. "
                        "Give concise, practical and general "
                        "skincare information."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.6
        )

        ai_response = completion.choices[0].message.content

        return f"""

<div class="report-title">
🌸 GlowGuide AI
</div>

<div class="result-heading">
✨ Recommended Routine
</div>

<div class="routine-name">
{predicted_routine}
</div>

<div class="confidence">
Machine Learning Confidence: <b>{confidence:.1f}%</b>
</div>

<hr>

{ai_response}

<hr>

<div class="disclaimer">
<b>Disclaimer:</b> GlowGuide AI provides general skincare
information and is not a substitute for professional
medical or dermatological advice.
</div>

"""

    except Exception as e:

        return f"""

<div class="error-box">

<h3>❌ Unable to Generate Report</h3>

Something went wrong while creating your GlowGuide report.

<br><br>

<code>{str(e)}</code>

</div>

"""


# ============================================================
# 5. CHATBOT
# ============================================================

def chat_with_glowguide(message, history):

    try:

        groq_messages = [

            {
                "role": "system",
                "content": (
                    "You are GlowGuide AI, a friendly and "
                    "knowledgeable skincare assistant. "
                    "Answer skincare questions clearly and "
                    "concisely. Give general educational "
                    "skincare information. Do not diagnose "
                    "or treat medical conditions."
                )
            }

        ]

        for item in history:

            if isinstance(item, dict):

                role = item.get("role")
                content = item.get("content")

                if role not in ["user", "assistant"]:
                    continue

                if isinstance(content, str):

                    if content.strip():

                        groq_messages.append({
                            "role": role,
                            "content": content
                        })

                elif isinstance(content, list):

                    text_parts = []

                    for block in content:

                        if isinstance(block, dict):

                            if block.get("type") == "text":

                                text_parts.append(
                                    block.get("text", "")
                                )

                        elif isinstance(block, str):

                            text_parts.append(block)

                    combined_text = "\n".join(text_parts)

                    if combined_text.strip():

                        groq_messages.append({
                            "role": role,
                            "content": combined_text
                        })

            elif isinstance(item, (list, tuple)):

                if len(item) >= 2:

                    user_message = item[0]
                    assistant_message = item[1]

                    if user_message:

                        groq_messages.append({
                            "role": "user",
                            "content": str(user_message)
                        })

                    if assistant_message:

                        groq_messages.append({
                            "role": "assistant",
                            "content": str(assistant_message)
                        })

        groq_messages.append({
            "role": "user",
            "content": message
        })

        completion = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=groq_messages,

            temperature=0.7

        )

        return completion.choices[0].message.content

    except Exception as e:

        return f"❌ Error: {str(e)}"


# ============================================================
# 6. CUSTOM CSS
# ============================================================

custom_css = """
body {
    background:
        linear-gradient(
            135deg,
            #fff7fb 0%,
            #fdf1fa 45%,
            #f7edfb 100%
        ) !important;
}

.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;

    background:
        linear-gradient(
            135deg,
            #fff7fb 0%,
            #fdf1fa 45%,
            #f7edfb 100%
        ) !important;

    color: #293248 !important;
}

.main-header {
    text-align: center;
    padding: 5px 0 18px 0;
}

.main-title {
    font-family: Georgia, serif;
    font-size: 34px;
    font-weight: 700;
    color: #d62d82;
    margin-bottom: 8px;
}

.main-subtitle {
    font-family: Georgia, serif;
    font-size: 19px;
    font-weight: 600;
    color: #293248;
    margin-bottom: 10px;
}

.main-description {
    font-size: 14px;
    color: #555c70;
    line-height: 1.7;
}

.section-title {
    font-family: Georgia, serif;
    font-size: 22px;
    font-weight: 700;
    color: #8d42a7;
    margin: 5px 0 5px 0;
}

.section-description {
    font-size: 13px;
    color: #666b78;
    margin-bottom: 10px;
}

.required-area {
    background:
        linear-gradient(
            135deg,
            rgba(255,247,251,0.82),
            rgba(250,239,250,0.72)
        ) !important;

    border: 1px solid #eadce9 !important;
    border-radius: 14px !important;
    padding: 16px 18px 18px 18px !important;

    box-shadow:
        0 4px 14px rgba(141,66,167,0.06) !important;
}

.gradio-container label span {
    font-family: Georgia, serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #8d42a7 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    font-size: 13px !important;
    border-radius: 10px !important;
    border: 1px solid #e3cfe0 !important;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.68),
            rgba(255,247,251,0.78)
        ) !important;

    color: #293248 !important;
}

.gradio-container select {
    min-height: 40px !important;
}

.optional-area {
    background:
        linear-gradient(
            135deg,
            rgba(255,247,251,0.72),
            rgba(250,239,250,0.62)
        ) !important;

    border: 1px solid #e6dce7 !important;
    border-radius: 14px !important;
    margin-top: 12px !important;
    margin-bottom: 14px !important;
    padding: 8px 12px !important;
}

.optional-text {
    font-size: 13px;
    color: #6a6871;
    padding: 0 0 8px 0;
}

.generate-btn {
    background: #bcdcff !important;
    color: #273248 !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 43px !important;
    font-family: Georgia, serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

.generate-btn:hover {
    background: #add2fa !important;
}

.reset-btn {
    border-radius: 10px !important;
    min-height: 43px !important;
    font-family: Georgia, serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

.output-area {
    background:
        linear-gradient(
            135deg,
            rgba(255,247,251,0.35),
            rgba(250,239,250,0.35)
        ) !important;

    border: none !important;
    padding: 4px 0 15px 0 !important;
    color: #293248 !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
}

.report-title {
    text-align: center;
    font-family: Georgia, serif;
    font-size: 27px;
    font-weight: 700;
    color: #d62d82;
    margin: 5px 0 20px 0;
}

.output-area h3 {
    text-align: center !important;
    font-family: Georgia, serif !important;
    font-size: 19px !important;
    color: #d02f82 !important;
    margin-top: 24px !important;
    margin-bottom: 10px !important;
    padding: 7px 12px !important;
    border-radius: 9px !important;
    background: rgba(255,247,251,0.58) !important;
}

.output-area h2 {
    text-align: center !important;
    font-family: Georgia, serif !important;
    font-size: 20px !important;
    color: #8d42a7 !important;
}

.output-area h4 {
    font-family: Georgia, serif !important;
    color: #a64b88 !important;
    font-size: 16px !important;
}

.output-area p {
    font-size: 13.5px !important;
    line-height: 1.65 !important;
    margin-bottom: 7px !important;
}

.output-area li {
    font-size: 13.5px !important;
    line-height: 1.6 !important;
    margin-bottom: 4px !important;
}

.result-heading {
    text-align: center;
    font-family: Georgia, serif;
    font-size: 21px;
    font-weight: 700;
    color: #8d42a7;
    margin-top: 8px;
}

.routine-name {
    text-align: center;
    font-family: Georgia, serif;
    font-size: 21px;
    font-weight: 700;
    color: #293248;
    margin-top: 8px;
}

.confidence {
    text-align: center;
    font-size: 13px;
    color: #646979;
    margin-top: 7px;
    margin-bottom: 15px;
}

.output-area hr {
    border: none;
    border-top: 1px solid #e2d8e2;
    margin: 18px 0;
}

.disclaimer {
    font-size: 11.5px;
    line-height: 1.5;
    color: #6c6c75;
    text-align: center;
    margin-top: 15px;
}

.error-box {
    color: #8b315e;
    font-size: 14px;
    text-align: center;
    padding: 20px;
}

.chat-area {
    background:
        linear-gradient(
            135deg,
            rgba(255,247,251,0.82),
            rgba(250,239,250,0.72)
        ) !important;

    border: 1px solid #eadce9 !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

.tab-nav button {
    font-family: Georgia, serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #8d42a7 !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 9px 16px !important;
    background: rgba(255,247,251,0.35) !important;
}

.tab-nav button.selected {
    color: #d62d82 !important;
    background: rgba(255,247,251,0.72) !important;
    border-color: #eadce9 !important;
}

footer {
    display: none !important;
}
"""


# ============================================================
# 7. GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="GlowGuide AI") as app:

    gr.HTML(
        """
        <div class="main-header">

            <div class="main-title">
                🌸 GlowGuide AI
            </div>

            <div class="main-subtitle">
                AI Powered Personalized Skincare Recommendation System
            </div>

            <div class="main-description">

                Receive personalized skincare recommendations using:
                <br>

                🤖 Machine Learning &nbsp;&nbsp;
                🧠 Groq AI &nbsp;&nbsp;
                ✨ Personalized AI Guidance

            </div>

        </div>
        """
    )

    with gr.Tabs():

        with gr.Tab("📋 Skincare Report Generator"):

            gr.HTML(
                """
                <div class="section-title">
                    👤 Required Details
                </div>

                <div class="section-description">
                    Fill in these 5 details to receive your
                    personalized skincare recommendation.
                </div>
                """
            )

            with gr.Column(elem_classes=["required-area"]):

                with gr.Row():

                    age = gr.Number(
                        label="Age",
                        minimum=15,
                        maximum=60,
                        precision=0,
                        value=None
                    )

                    skin = gr.Dropdown(
                        choices=[
                            "Combination",
                            "Dry",
                            "Normal",
                            "Oily",
                            "Sensitive"
                        ],
                        label="Skin Type",
                        value=None
                    )

                with gr.Row():

                    concern = gr.Dropdown(
                        choices=[
                            "Acne",
                            "Dryness",
                            "Dullness",
                            "Fine Lines",
                            "Large Pores",
                            "No Major Concern",
                            "Pigmentation",
                            "Uneven Texture"
                        ],
                        label="Skin Concern",
                        value=None
                    )

                    budget = gr.Dropdown(
                        choices=[
                            "Under ₹500",
                            "₹500–1000",
                            "₹1000–2000",
                            "Above ₹2000"
                        ],
                        label="Budget",
                        value=None
                    )

                season = gr.Dropdown(
                    choices=[
                        "Monsoon",
                        "Spring",
                        "Summer",
                        "Winter"
                    ],
                    label="Season",
                    value=None
                )

            with gr.Accordion(
                "＋ Add More Details (Optional)",
                open=False,
                elem_classes=["optional-area"]
            ):

                gr.HTML(
                    """
                    <div class="optional-text">
                        These lifestyle details are optional. You can
                        generate your report without filling them.
                        They simply help GlowGuide make the
                        recommendation more personalized.
                    </div>
                    """
                )

                with gr.Row():

                    water = gr.Dropdown(
                        choices=[
                            "Less than 1 L",
                            "1–2 L",
                            "More than 2 L"
                        ],
                        label="Water Intake",
                        value=None
                    )

                    sleep = gr.Dropdown(
                        choices=[
                            "Less than 6 hrs",
                            "6–8 hrs",
                            "More than 8 hrs"
                        ],
                        label="Sleep Duration",
                        value=None
                    )

                with gr.Row():

                    sunscreen = gr.Dropdown(
                        choices=[
                            "Never",
                            "Sometimes",
                            "Always"
                        ],
                        label="Sunscreen Usage",
                        value=None
                    )

                    makeup = gr.Dropdown(
                        choices=[
                            "Never",
                            "Rarely",
                            "Occasionally",
                            "Daily"
                        ],
                        label="Makeup Usage",
                        value=None
                    )

                routine = gr.Dropdown(
                    choices=[
                        "No Routine",
                        "Basic",
                        "Regular",
                        "Advanced"
                    ],
                    label="Current Routine",
                    value=None
                )

            with gr.Row():

                generate_button = gr.Button(
                    "✨ Generate GlowGuide Report",
                    variant="primary",
                    elem_classes=["generate-btn"],
                    scale=4
                )

                reset_button = gr.Button(
                    "🔄 Reset",
                    variant="secondary",
                    elem_classes=["reset-btn"],
                    scale=1
                )

            gr.HTML(
                """
                <div class="section-title">
                    🌸 Personalized Skincare Report
                </div>
                """
            )

            output = gr.Markdown(
                value="""
<div class="report-title">
🌸 Welcome to GlowGuide AI
</div>

<div style="text-align:center; font-size:13px;">

Fill in the <b>5 required details</b> and click
<b>Generate GlowGuide Report</b>.

<br><br>

Additional lifestyle details are optional.

</div>
                """,
                elem_classes=["output-area"]
            )

            generate_button.click(
                fn=generate_skincare_report,
                inputs=[
                    age,
                    skin,
                    concern,
                    budget,
                    season,
                    water,
                    sleep,
                    sunscreen,
                    makeup,
                    routine
                ],
                outputs=output
            )

            reset_button.click(
                fn=lambda: (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    """
<div class="report-title">
🌸 Welcome to GlowGuide AI
</div>

<div style="text-align:center; font-size:13px;">

Fill in the <b>5 required details</b> and click
<b>Generate GlowGuide Report</b>.

<br><br>

Additional lifestyle details are optional.

</div>
                    """
                ),
                inputs=[],
                outputs=[
                    age,
                    skin,
                    concern,
                    budget,
                    season,
                    water,
                    sleep,
                    sunscreen,
                    makeup,
                    routine,
                    output
                ]
            )

        with gr.Tab("💬 Skincare Assistant Q&A"):

            gr.HTML(
                """
                <div class="section-title">
                    💬 Skincare Assistant
                </div>

                <div class="section-description">
                    Have questions about skincare routines,
                    ingredients, sunscreen or everyday skincare?
                    Ask GlowGuide AI.
                </div>
                """
            )

            with gr.Column(elem_classes=["chat-area"]):

                gr.ChatInterface(
                    fn=chat_with_glowguide,
                    title="🌸 GlowGuide AI Assistant",
                    description="Your personalized AI skincare assistant",
                    textbox=gr.Textbox(
                        placeholder=(
                            "e.g. Can I use Vitamin C "
                            "and Niacinamide together?"
                        )
                    ),
                    examples=[
                        "What skincare routine is good for oily skin?",
                        "What ingredients help with pigmentation?",
                        "How should I build a basic skincare routine?",
                        "Why is sunscreen important?",
                        "Can I use Vitamin C and Niacinamide together?"
                    ]
                )


# ============================================================
# 8. LAUNCH - RENDER COMPATIBLE
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    app.launch(
        server_name="0.0.0.0",
        server_port=port,

        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="pink",
            neutral_hue="slate"
        ),

        css=custom_css
    )
