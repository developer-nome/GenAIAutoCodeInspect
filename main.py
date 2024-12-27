from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from openai import OpenAI
from pydantic import BaseModel

import datetime
import os

load_dotenv()

texts_global = []

class GitRepoToInspect(BaseModel):
    url: str
    numItems: int
    message: str

class InspectionTask(BaseModel):
    iterationNum: int
    metadataSource: str
    messageContent: str
    beginingWith: str
    codeLanguage: str

class RemediateTask(BaseModel):
    iterationNum: int
    selectedText: str
    messageContent: str
    codeLanguage: str

app = FastAPI()

favicon_path = 'favicon.ico'

origins = [
    "*",  # Add the origins that are allowed (you can also use "*" to allow all)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


@app.post("/items/")
async def create_item(gitRepoToInspect: GitRepoToInspect):
    gitRepoToInspect.message = ""
    repo_path = gitRepoToInspect.url

    if repo_path.endswith('.git'):
        repo_path = repo_path[:-4]

    first_characters = gitRepoToInspect.url[:4]
    try:
        if first_characters == "http":
            last_characters = repo_path.rsplit('/', 1)[-1]
            repo_path = "../temp/" + last_characters
            if os.path.exists(repo_path) and os.path.isdir(repo_path):
                current_datetime = datetime.datetime.now()
                formatted_datetime = current_datetime.strftime("%Y%m%d%H%M")
                formatted_datetime = formatted_datetime[2:]
                repo_path = repo_path + "_" + formatted_datetime
            repo = Repo.clone_from(gitRepoToInspect.url, to_path=repo_path)
    except Exception as e:
        gitRepoToInspect.message = "Unabld to clone repo: " + str(e)

    texts_all_types = []
    extensions = ['py', 'js', 'java', 'go', 'cs', 'php', 'ts', 'cpp', 'c', 'swift', '']
    try:
        for ext in extensions:
            if ext == "py":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".py"],
                    exclude=["**/non-utf8-encoding.py"],
                    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
                )
                documents = loader.load()    
                python_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
                )
                texts = python_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "js":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".js"],
                    exclude=["**/non-utf8-encoding.js"],
                    parser=LanguageParser(language=Language.JS, parser_threshold=500),
                )
                documents = loader.load()    
                java_script_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.JS, chunk_size=2000, chunk_overlap=200
                )
                texts = java_script_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "java":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".java"],
                    exclude=["**/non-utf8-encoding.java"],
                    parser=LanguageParser(parser_threshold=500),
                )
                documents = loader.load()    
                java_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.JAVA, chunk_size=2000, chunk_overlap=200
                )
                texts = java_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "go":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".go"],
                    exclude=["**/non-utf8-encoding.go"],
                    parser=LanguageParser(parser_threshold=500),
                )
                documents = loader.load()    
                go_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.GO, chunk_size=2000, chunk_overlap=200
                )
                texts = go_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "cs":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".cs"],
                    exclude=["**/non-utf8-encoding.cs"],
                    parser=LanguageParser(parser_threshold=500),
                )
                documents = loader.load()    
                cs_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.CSHARP, chunk_size=2000, chunk_overlap=200
                )
                texts = cs_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "php":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".php"],
                    exclude=["**/non-utf8-encoding.php"],
                    parser=LanguageParser(parser_threshold=500),
                )
                documents = loader.load()    
                php_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PHP, chunk_size=2000, chunk_overlap=200
                )
                texts = php_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "ts":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".ts"],
                    exclude=["**/non-utf8-encoding.ts"],
                    parser=LanguageParser(language=Language.JS, parser_threshold=500),
                )
                documents = loader.load()    
                type_script_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.TS, chunk_size=2000, chunk_overlap=200
                )
                texts = type_script_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "cpp":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".cpp"],
                    exclude=["**/non-utf8-encoding.cpp"],
                    parser=LanguageParser(language=Language.CPP, parser_threshold=500),
                )
                documents = loader.load()    
                cpp_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.CPP, chunk_size=2000, chunk_overlap=200
                )
                texts = cpp_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "c":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".c"],
                    exclude=["**/non-utf8-encoding.c"],
                    parser=LanguageParser(language=Language.CPP, parser_threshold=500),
                )
                documents = loader.load()    
                c_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.CPP, chunk_size=2000, chunk_overlap=200
                )
                texts = c_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "swift":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/*",
                    suffixes=[".swift"],
                    exclude=["**/non-utf8-encoding.c"],
                    parser=LanguageParser(parser_threshold=500),
                )
                documents = loader.load()    
                swift_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.SWIFT, chunk_size=2000, chunk_overlap=200
                )
                texts = swift_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
            elif ext == "":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/Dockerfile",
                    exclude=["**/non-utf8-encoding.txt"],
                    parser=LanguageParser(parser_threshold=500),
                )
                documents = loader.load()    
                text_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.MARKDOWN, chunk_size=2000, chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                texts_all_types = texts_all_types + texts
    except Exception as e:
        gitRepoToInspect.message = "Unable to load files."

    global texts_global
    texts_global = texts_all_types
    gitRepoToInspect.numItems = len(texts_global)

    if gitRepoToInspect.numItems == 0:
        gitRepoToInspect.message = "Directory does not exist. Or, no " + str(extensions) + " files found"

    return gitRepoToInspect


@app.post("/buildOutContainers/")
async def buildOutContainers(inspectionTask: InspectionTask):
    source_value = texts_global[inspectionTask.iterationNum].metadata['source']
    file_ext = source_value.rsplit('.', 1)[-1]

    code_language = "Python 3"
    if (file_ext == "js"):
        code_language = "JavaScript"
    if (file_ext == "java"):
        code_language = "Java"
    if (file_ext == "go"):
        code_language = "GoLang"
    if (file_ext == "cs"):
        code_language = "C#"
    if (file_ext == "php"):
        code_language = "PHP"
    if (file_ext == "ts"):
        code_language = "TypeScript"
    if (file_ext == "cpp"):
        code_language = "C++"
    if (file_ext == "c"):
        code_language = "C"
    if (file_ext == "swift"):
        code_language = "Swift"
    if ("Dockerfile" in source_value):
        code_language = "Docker"

    inspectionTask.metadataSource = texts_global[inspectionTask.iterationNum].metadata
    inspectionTask.messageContent = ""  # will be filled in with streaming request
    inspectionTask.beginingWith = beginingCodeSnippet(texts_global[inspectionTask.iterationNum].page_content)
    inspectionTask.codeLanguage = code_language
    return inspectionTask


@app.post("/retrieveInspectionsStream/")
async def processInspectionTaskStream(inspectionTask: InspectionTask):
    source_value = texts_global[inspectionTask.iterationNum].metadata['source']
    file_ext = source_value.rsplit('.', 1)[-1]

    code_language = "Python 3"
    if (file_ext == "js"):
        code_language = "JavaScript"
    if (file_ext == "java"):
        code_language = "Java"
    if (file_ext == "go"):
        code_language = "GoLang"
    if (file_ext == "cs"):
        code_language = "C#"
    if (file_ext == "php"):
        code_language = "PHP"
    if (file_ext == "ts"):
        code_language = "TypeScript"
    if (file_ext == "cpp"):
        code_language = "C++"
    if (file_ext == "c"):
        code_language = "C"
    if (file_ext == "swift"):
        code_language = "Swift"
    if ("Dockerfile" in source_value):
        code_language = "Docker"

    client = AsyncOpenAI(
        api_key = os.getenv("API_KEY"),
        base_url = os.getenv("BASE_URL")
    )

    stream = await client.chat.completions.create(
        messages=[
        {
            'role': 'user', 
            'content': 'Give a short and concise response. Analyze the following ' + code_language + ' code for security vulnerabilities using OWASP guidelines and report any issues found in a brief summary. \n\n ' + texts_global[inspectionTask.iterationNum].page_content
        }
    ],
    model = os.getenv("LLM_MODEL"), 
    stream=True
    )
    
    async def generator():
        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    inspectionTask.metadataSource = texts_global[inspectionTask.iterationNum].metadata
    inspectionTask.beginingWith = beginingCodeSnippet(texts_global[inspectionTask.iterationNum].page_content)
    inspectionTask.codeLanguage = code_language
    response_messages = generator()
    return StreamingResponse(response_messages, media_type="text/event-stream")


@app.post("/retrieveRemediationsStream/")
async def processRemediationTask(remediateTask: RemediateTask):
    deep_inspection_prompt_prefix = 'Provide updated ' + remediateTask.codeLanguage + ' code that will address the ' + remediateTask.selectedText + ' issue in the following code: \n\n '
    
    if os.getenv("QUESTION_BUTTON_MODE") == "ATTACK":
        deep_inspection_prompt_prefix = 'How would a white-hat security researcher exploit the ' + remediateTask.selectedText + ' issue in the following code: \n\n '
    
    client = AsyncOpenAI(
        api_key = os.getenv("API_KEY"),
        base_url = os.getenv("BASE_URL")
    )
    stream = await client.chat.completions.create(
    messages=[
        {
            'role': 'user', 
            'content': deep_inspection_prompt_prefix + texts_global[remediateTask.iterationNum].page_content
        }
    ],
    model = os.getenv("LLM_MODEL_DEEP_INSPECTION"),
    stream=True
    )

    async def generator():
        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""
    
    response_messages = generator()
    return StreamingResponse(response_messages, media_type="text/event-stream")


def beginingCodeSnippet(page_content):
    firstFiftyChars = str(page_content)[:50]
    listSplitOnCarriageReturn = firstFiftyChars.splitlines(True)
    # firstLine = listSplitOnCarriageReturn[0]
    return listSplitOnCarriageReturn[0]


def increase_number_suffix(string):
    # Regular expression to match underscore followed by a number at the end of the string
    pattern = r'_(\d+)$'
    match = re.search(pattern, string)
    if match:
        # Extract the number and increment it
        number = int(match.group(1))
        new_number = number + 1
        # Replace the old number with the new one in the string
        return re.sub(pattern, '_' + str(new_number), string)
    else:
        return string





