# GenAIAutoCodeInspect
Leveraging Generative AI to Automate Source Code Inspection

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
