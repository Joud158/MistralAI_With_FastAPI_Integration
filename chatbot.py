import requests
from collections import deque
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = "mistral-tiny"

class MistralRequest(BaseModel):
    api_key:str
    question:str

#--- Prompt Engineering ---#
ROLE = "You are a friendly, precise cultural assistant for quick, factual guidance"

TASK = (
    "Answer cultural questions clearly and briefly, Prioritize accuracy, define terms, and add one actionnable tip when helpful"
)

CONSTRAINTS = (
    "Do not fabricate references. If unsure, say so briefly."
    "Avoid policy, medical, or legal advice. Keep answers under 200 words when possible."
)

STYLE = "Tone: warm, concise, non-patrionizing. Use simple sentences and neutral vocabulary."

OUTPUT_FORMAT = (
    "Answer using this structure: \n"
    "1) Direct answer (2-4 sentences)\n"
    "2) Optional bullets (max 3)\n"
    "3) One follow-up question on user intent"
)

SYSTEM_PROMPT = f"{ROLE}\n\nTask: \n{TASK}\n\nConstraints: \n{CONSTRAINTS}\n\nStyle: \n{STYLE}\n\nOutput format: \n{OUTPUT_FORMAT}\n\n"

#--- Context Framwork ---#
memory = [
    "User prefers concise, technical explanations.",
    "User is comfortable with short bullet points."
]

history = deque(maxlen = 6)

#--- Guardrails Function ---#
def enforce_culture_only(user_message):
    culture_keywords = [
        "culture", "art", "music", "literature", "tradition", "heritage",
        "festival", "custom", "society", "language", "dance", "ritual",
        "belief", "musuem", "history", "film", "painting", "architecture",
        "sculpture"
    ]

    if not any (keyword in user_message.lower() for keyword in culture_keywords):
        return (
            "This assistant only answers culture-related questions. "
            "Please rephrase your question to focus on arts, traditions, heritage, "
            "history, or any other cultural topics. "
        )
    
    return None

@app.post("/chat")
def chat_with_mistral(request:MistralRequest):

    guardrails_reply = enforce_culture_only(request.question)
    if guardrails_reply:
        return {"Chatbot: ", guardrails_reply}
    
    messages = [
        {
            "role":"system", "content": SYSTEM_PROMPT
        }
    ]

    if memory:
        mem_text = "Conversation memory: \n-"+"\n".join(memory)
        messages.append({"role":"system","content":mem_text})

    messages.extend(list(history))

    messages.append({"role":"user", "content":request.question})

    data = {
        "model": model,
        "messages": messages
    }

    headers = {
        "Authorization":f"Bearer {request.api_key}",
        "Content_Type":"application/json"
    }

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers = headers,
        json = data
    )

    output = response.json()["choices"][0]["message"]["content"]

    history.append({"role":"user", "content":request.question})
    history.append({"role":"assistant", "content":output})

    return {"Chatbot:",output}