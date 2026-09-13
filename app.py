
import os
import joblib
import pandas as pd
import gradio as gr
from groq import Groq


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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Please add your Groq API key in the notebook session."
    )

client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# 3. DEFAULT OPTIONAL VALUES
# ============================================================

DEFAULT_SEASON = "Summer"
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
    sensitive,
    season,
    water,
    sleep,
    sunscreen,
    makeup,
    routine
):

    try:

        # ------------------------------------------------------
        # REQUIRED INPUT VALIDATION
        # ------------------------------------------------------

        if age is None:
            return """
# 🌸 GlowGuide AI

Please enter your **Age** to continue.
"""

        if not skin:
            return """
# 🌸 GlowGuide AI

Please select your **Skin Type** to continue.
"""

        if not concern:
            return """
# 🌸 GlowGuide AI

Please select your **Skin Concern** to continue.
"""

        if not budget:
            return """
# 🌸 GlowGuide AI

Please select your **Budget** to continue.
"""

        if not sensitive:
            return """
# 🌸 GlowGuide AI

Please select whether you have **Sensitive Skin**.
"""


        # ------------------------------------------------------
        # OPTIONAL INPUTS
        # ------------------------------------------------------

        season = season if season else DEFAULT_SEASON
        water = water if water else DEFAULT_WATER
        sleep = sleep if sleep else DEFAULT_SLEEP
        sunscreen = sunscreen if sunscreen else DEFAULT_SUNSCREEN
        makeup = makeup if makeup else DEFAULT_MAKEUP
        routine = routine if routine else DEFAULT_ROUTINE


        # ------------------------------------------------------
        # ENCODE INPUTS
        # ------------------------------------------------------

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


        # ------------------------------------------------------
        # MACHINE LEARNING PREDICTION
        # ------------------------------------------------------

        pred_class = ml_model.predict(input_df)[0]

        probabilities = ml_model.predict_proba(input_df)[0]

        confidence = probabilities.max() * 100

        predicted_routine = label_encoders[
            "Routine_Category"
        ].inverse_transform([pred_class])[0]


        # ------------------------------------------------------
        # GROQ PROMPT
        # ------------------------------------------------------

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
Sensitive Skin: {sensitive}

OPTIONAL DETAILS
----------------
Season: {season}
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

### ☀️ Morning Routine
Give 4-5 simple numbered steps.

### 🌙 Night Routine
Give 4-5 simple numbered steps.

### 🧴 Ingredients to Look For
Give 3-5 useful ingredients with very short explanations.

### 🚫 Ingredients / Product Types to Be Careful With
Give 2-4 concise points.

### 💄 Makeup & Skincare Tips
Give 2-3 concise tips if relevant.

### 🌿 Lifestyle & Seasonal Tips
Give 3-4 concise tips based on the user's optional details.

### ⚠️ Precautions
Give 2-3 short precautions.

IMPORTANT:

- Keep the complete response under 450 words.
- Use short sentences.
- Prefer bullet points.
- Do not write long paragraphs.
- Do not repeat the user's profile.
- Do not diagnose medical conditions.
- Do not prescribe medication.
- Do not claim to cure diseases.
- Mention professional dermatology advice when appropriate.
- Keep the tone elegant, friendly and practical.
"""


        # ------------------------------------------------------
        # GROQ RESPONSE
        # ------------------------------------------------------

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


        # ------------------------------------------------------
        # FINAL REPORT
        # ------------------------------------------------------

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

### ❌ Unable to Generate Report

Something went wrong while creating your GlowGuide report.

`{str(e)}`

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


        # ------------------------------------------------------
        # HANDLE GRADIO CHAT HISTORY
        # ------------------------------------------------------

        for item in history:

            # Gradio message format
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


            # Older tuple format
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


        # ------------------------------------------------------
        # CURRENT MESSAGE
        # ------------------------------------------------------

        groq_messages.append({

            "role": "user",
            "content": message

        })


        # ------------------------------------------------------
        # GROQ RESPONSE
        # ------------------------------------------------------

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

/* ============================================================
   MAIN BACKGROUND
   ============================================================ */

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


/* ============================================================
   HEADER
   ============================================================ */

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


/* ============================================================
   SECTION TITLE
   ============================================================ */

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


/* ============================================================
   REQUIRED AREA
   ============================================================ */

.required-area {

    background: rgba(255,255,255,0.40) !important;

    border: 1px solid #eadce9 !important;

    border-radius: 4px !important;

    padding: 10px 14px 12px 14px !important;
}


/* ============================================================
   INPUT LABELS
   ============================================================ */

.gradio-container label span {

    font-size: 13px !important;

    font-weight: 600 !important;

    color: #56647d !important;
}


/* ============================================================
   INPUTS
   ============================================================ */

.gradio-container input {

    font-size: 13px !important;

    border-radius: 4px !important;

    border: 1px solid #dcdce6 !important;

    background: rgba(255,255,255,0.70) !important;
}


.gradio-container .wrap {

    border-radius: 4px !important;

    border-color: #dcdce6 !important;
}


/* ============================================================
   DROPDOWN
   ============================================================ */

.gradio-container select {

    font-size: 13px !important;
}


/* ============================================================
   OPTIONAL AREA
   ============================================================ */

.optional-area {

    background: rgba(255,255,255,0.38) !important;

    border: 1px solid #e6dce7 !important;

    border-radius: 4px !important;

    margin-top: 10px !important;

    margin-bottom: 12px !important;
}


.optional-text {

    font-size: 13px;

    color: #6a6871;

    padding: 0 0 8px 0;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.generate-btn {

    background: #bcdcff !important;

    color: #273248 !important;

    border: none !important;

    border-radius: 4px !important;

    min-height: 43px !important;

    font-family: Georgia, serif !important;

    font-size: 16px !important;

    font-weight: 700 !important;
}


.generate-btn:hover {

    background: #add2fa !important;
}


.reset-btn {

    border-radius: 4px !important;

    min-height: 43px !important;

    font-family: Georgia, serif !important;

    font-size: 15px !important;

    font-weight: 600 !important;
}


/* ============================================================
   REPORT AREA
   ============================================================ */

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


/* ============================================================
   REPORT TITLE
   ============================================================ */

.report-title {

    text-align: center;

    font-family: Georgia, serif;

    font-size: 27px;

    font-weight: 700;

    color: #d62d82;

    margin: 5px 0 20px 0;
}


/* ============================================================
   REPORT HEADINGS
   ============================================================ */

.output-area h3 {

    text-align: center !important;

    font-family: Georgia, serif !important;

    font-size: 19px !important;

    color: #d02f82 !important;

    margin-top: 24px !important;

    margin-bottom: 10px !important;
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


/* ============================================================
   REPORT TEXT
   ============================================================ */

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


/* ============================================================
   ROUTINE RESULT
   ============================================================ */

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


/* ============================================================
   HORIZONTAL LINE
   ============================================================ */

.output-area hr {

    border: none;

    border-top: 1px solid #e2d8e2;

    margin: 18px 0;
}


/* ============================================================
   DISCLAIMER
   ============================================================ */

.disclaimer {

    font-size: 11.5px;

    line-height: 1.5;

    color: #6c6c75;

    text-align: center;

    margin-top: 15px;
}


/* ============================================================
   ERROR
   ============================================================ */

.error-box {

    color: #8b315e;

    font-size: 14px;
}


/* ============================================================
   CHAT
   ============================================================ */

.chat-area {

    background: rgba(255,255,255,0.30) !important;

    border: 1px solid #eadce9 !important;

    border-radius: 6px !important;

    padding: 12px !important;
}


.chat-area textarea {

    font-size: 13px !important;
}


/* ============================================================
   TABS
   ============================================================ */

.tab-nav button {

    font-size: 13px !important;

    font-weight: 600 !important;

    border-radius: 4px !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

footer {

    display: none !important;
}

"""


# ============================================================
# 7. GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="GlowGuide AI"
) as app:


    # ========================================================
    # HEADER
    # ========================================================

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


    # ========================================================
    # TABS
    # ========================================================

    with gr.Tabs():


        # ====================================================
        # REPORT TAB
        # ====================================================

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


            # ------------------------------------------------
            # REQUIRED INPUTS
            # ------------------------------------------------

            with gr.Column(
                elem_classes=["required-area"]
            ):

                with gr.Row():

                    age = gr.Number(
                        label="Age *",
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

                        label="Skin Type *",

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

                        label="Skin Concern *",

                        value=None
                    )


                    budget = gr.Dropdown(

                        choices=[
                            "Under ₹500",
                            "₹500–1000",
                            "₹1000–2000",
                            "Above ₹2000"
                        ],

                        label="Budget *",

                        value=None
                    )


                sensitive = gr.Dropdown(

                    choices=[
                        "No",
                        "Yes"
                    ],

                    label="Sensitive Skin *",

                    value=None
                )


            # ------------------------------------------------
            # OPTIONAL INPUTS
            # ------------------------------------------------

            with gr.Accordion(
                "＋ Add More Details (Optional)",
                open=False,
                elem_classes=["optional-area"]
            ):

                gr.HTML(
                    """
                    <div class="optional-text">
                        These details are optional. You can generate
                        your report without filling them. They simply
                        help GlowGuide make the recommendation more
                        personalized.
                    </div>
                    """
                )


                with gr.Row():

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


                    water = gr.Dropdown(

                        choices=[
                            "Less than 1 L",
                            "1–2 L",
                            "More than 2 L"
                        ],

                        label="Water Intake",

                        value=None
                    )


                with gr.Row():

                    sleep = gr.Dropdown(

                        choices=[
                            "Less than 6 hrs",
                            "6–8 hrs",
                            "More than 8 hrs"
                        ],

                        label="Sleep Duration",

                        value=None
                    )


                    sunscreen = gr.Dropdown(

                        choices=[
                            "Never",
                            "Sometimes",
                            "Always"
                        ],

                        label="Sunscreen Usage",

                        value=None
                    )


                with gr.Row():

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


            # ------------------------------------------------
            # BUTTONS
            # ------------------------------------------------

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


            # ------------------------------------------------
            # OUTPUT HEADING
            # ------------------------------------------------

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


            # ------------------------------------------------
            # GENERATE BUTTON
            # ------------------------------------------------

            generate_button.click(

                fn=generate_skincare_report,

                inputs=[

                    age,
                    skin,
                    concern,
                    budget,
                    sensitive,

                    season,
                    water,
                    sleep,
                    sunscreen,
                    makeup,
                    routine

                ],

                outputs=output
            )


            # ------------------------------------------------
            # RESET BUTTON
            # ------------------------------------------------

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
                    sensitive,

                    season,
                    water,
                    sleep,
                    sunscreen,
                    makeup,
                    routine,

                    output

                ]
            )


        # ====================================================
        # CHATBOT TAB
        # ====================================================

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


            with gr.Column(
                elem_classes=["chat-area"]
            ):

                gr.ChatInterface(

                    fn=chat_with_glowguide,

                    title="🌸 GlowGuide AI Assistant",

                    description=(
                        "Your personalized AI skincare assistant"
                    ),

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
# 8. LAUNCH
# ============================================================

# ============================================================
# 8. LAUNCH
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