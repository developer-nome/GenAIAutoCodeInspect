# GenAIAutoCodeInspect
Author: William Horn

Leveraging Generative AI to Automate Source Code Inspection
![alt text](https://github.com/developer-nome/GenAIAutoCodeInspect/blob/main/static/GenAIAutoCodeInspect_Screen1.jpg)
![alt text](https://github.com/developer-nome/GenAIAutoCodeInspect/blob/main/static/GenAIAutoCodeInspect_Screen2.jpg)

# Sample environment file
```
API_KEY = '0'
BASE_URL = 'http://127.0.0.1:11434/v1'
LLM_MODEL = 'phi3.5:3.8b-mini-instruct-q6_K'
LLM_MODEL_DEEP_INSPECTION = 'phi3.5:3.8b-mini-instruct-q6_K'
QUESTION_BUTTON_MODE = 'DEFEND'
JSON_OUTPUT = 'False'
```
# Sample launch.json file
```
{
    "version": "0.2.0",
    "configurations": [
    
        {
            "name": "Python: main",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "main:app",
                "--reload",
                "--port",
                "8000"
            ],
            "jinja": true,
            "justMyCode": true
        },
        {
            "name": "Python: Scanner",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "scanner:app",
                "--reload",
                "--port",
                "8000"
            ],
            "jinja": true,
            "justMyCode": true
        }
    ]
}
```

# Top Tested LLMs
| Model    | Accuracy |
| -------- | ------- |
| Phi-3.5 (3.8B-Mini-Instruct-Q6_K)  | 96% |
| GPT 4o Mini | 94% |
| GLM 4 (9B-Chat-Q5_0)  | 94% |
| SuperNova-Medius (Q4_K_M) | 94% |
| Phi-4 (Q4_K_M) | 92% |
| Llama 3.1 (70B) | 92% |
| Wizard LM2 (7B-Q6_K) | 92% |
| Claud 3 Haiku | 90% |
| Claude 3.5 Sonnet | 88% |
| GPT 4o | 88% |
| Tulu 3 (8B Q4_K_M) | 88% |
| Llama 3.1 (405B) | 85% |
| Mistral Nemo | 85% |
| Llama 3.2 (3B-Instruct-Q6_K) | 83% |
| GPT 4 Turbo| 83% |
| OpenCodeInterpreter DS (6.7B-Q6_K)| 83% |
| Qwen 2.5 Coder (7B-Instruct-Q6_K) | 81% |
| CodeGemma | 79% |
| Llama 3.1 (8B-Instruct-Q6_K) | 79% |
| OpenCoder (8B Q8_0) | 79% |
| EXAONE 3.5 (Q8_0) | 75% |
| Falcon3 (10B Q4_K_M) | 63% |

