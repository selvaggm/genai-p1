import json
import requests
from pypdf import PdfReader


# --------------------------------------------------
# 1. Read PDF
# --------------------------------------------------

pdf_file = "sample_company_leave_policy.pdf"

reader = PdfReader(pdf_file)

pdf_text = ""

for page in reader.pages:
    pdf_text += page.extract_text() + "\n"


# --------------------------------------------------
# 2. Create Prompt
# --------------------------------------------------

prompt = f"""
You are an HR policy document extraction assistant.

Read the following company leave policy.

Extract the leave information and return ONLY valid JSON.

Required JSON structure:

{{
    "company": "company name",
    "leave_policies": [
        {{
            "leave_type": "type of leave",
            "days_per_year": number or null,
            "duration": "duration" or null,
            "carry_forward": true or false or null,
            "maximum_balance": number or null
        }}
    ]
}}

Rules:

1. Do not invent information.
2. If information is not available, return null.
3. Keep the leave type exactly as mentioned in the document.
4. Return only JSON.
5. Do not add explanations outside JSON.

DOCUMENT:

{pdf_text}
"""


# --------------------------------------------------
# 3. Call Ollama LLM API
# --------------------------------------------------

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "format": "json"
    }
)


# --------------------------------------------------
# 4. Get LLM response
# --------------------------------------------------

result = response.json()

llm_output = result["message"]["content"]


# --------------------------------------------------
# 5. Convert JSON string → Python object
# --------------------------------------------------

data = json.loads(llm_output)


# --------------------------------------------------
# 6. Display final result
# --------------------------------------------------

print("\n===== EXTRACTED LEAVE POLICY =====\n")

print(json.dumps(data, indent=4))