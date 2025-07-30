import uuid, logging, requests
from ..config import settings

logger = logging.getLogger(__name__)

class HyperClovaClient:
    def __init__(self):
        self.host = "https://clovastudio.stream.ntruss.com"
        self.api_key = settings.HYPERCLOVA_API_KEY

    def chat(self, messages, topP=0.8, topK=0, maxTokens=8192,
             temperature=0.5, repetitionPenalty=1.1,
             stop=None, includeAiFilters=True, seed=0):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json; charset=utf-8",
        }
        payload = {
            "messages": messages,
            "topP": topP,
            "topK": topK,
            "maxTokens": maxTokens,
            "temperature": temperature,
            "repetitionPenalty": repetitionPenalty,
            "stop": stop or [],
            "includeAiFilters": includeAiFilters,
            "seed": seed,
            "stream": False,
        }

        resp = requests.post(f"{self.host}/testapp/v3/chat-completions/HCX-005",
                             headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.debug("HyperClova raw response → %s", data)

        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        if "messages" in data:
            return data["messages"][-1].get("content", "")
        if "result" in data and "message" in data["result"]:
            return data["result"]["message"]["content"]

        raise ValueError(f"Unexpected HyperClova response format: {data!r}")

_client = HyperClovaClient()

def ask_hyperclova(question: str) -> str:
    return _client.chat([{"role": "user", "content": question}])
