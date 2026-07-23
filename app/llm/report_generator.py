import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = """\
You are a medical report assistant. Given a chest X-ray prediction, write a short \
structured report with exactly these sections:

## Findings
Describe the likely radiographic findings for the predicted condition in plain, \
clinical language (2-4 sentences).

## Impression
State the most likely diagnosis based on the prediction and confidence (1-2 sentences).

## Recommendation
Provide 2-3 appropriate next steps or recommendations for the patient/caregiver.

Rules:
- Keep the tone professional but accessible to a layperson.
- Do not invent details beyond what the prediction and confidence provide.
- Always end the report with the following disclaimer on its own line:
"This is an AI-generated report for educational/demonstration purposes only and is \
not a substitute for professional medical diagnosis."
"""


def generate_report(predicted_class: str, confidence: float) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    user_msg = (
        f"Prediction: {predicted_class}\n"
        f"Model confidence: {confidence:.2%}\n\n"
        "Generate the structured report."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()
