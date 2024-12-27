from datetime import date
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel

import datetime
import json
import os
import pandas as pd

load_dotenv()

texts_global = []
df = pd.DataFrame(columns=['metadataSource', 'beginingWith', 'codeLanguage', 'vulnerability', 'description', 'confidenceLevel'])

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

class DownloadCSVTask(BaseModel):
    selectedText: str
    messageContent: str

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

@app.get("/")
def read_root():
    model = os.getenv("LLM_MODEL")
    return {model}

@app.post("/items/")
async def create_item(gitRepoToInspect: GitRepoToInspect):
    global df
    df = df.drop(df.index)
    print("Dropped DataFrame")
    
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
    extensions = ['py', 'js', 'java', 'go', 'cs', 'php', 'ts', 'cpp', 'c', 'swift', 'env', '']
    try:
        for ext in extensions:
            if ext == "py":
                texts_all_types += process_files("py", repo_path, Language.PYTHON, True)
            elif ext == "js":
                texts_all_types += process_files("js", repo_path, Language.JS, True)
            elif ext == "java":
                texts_all_types += process_files("java", repo_path, Language.JAVA, False)
            elif ext == "go":
                texts_all_types += process_files("go", repo_path, Language.GO, False)
            elif ext == "cs":
                texts_all_types += process_files("cs", repo_path, Language.CSHARP, False)
            elif ext == "php":
                texts_all_types += process_files("php", repo_path, Language.PHP, False)
            elif ext == "ts":
                texts_all_types += process_files("ts", repo_path, Language.JS, True)
            elif ext == "cpp":
                texts_all_types += process_files("cpp", repo_path, Language.CPP, True)
            elif ext == "c":
                texts_all_types += process_files("c", repo_path, Language.CPP, True)
            elif ext == "swift":
                texts_all_types += process_files("swift", repo_path, Language.SWIFT, False)
            elif ext == "env":
                loader = GenericLoader.from_filesystem(
                    repo_path + "/",
                    glob="**/.env",
                    exclude=["**/non-utf8-encoding.txt"],
                    parser=LanguageParser(parser_threshold=500),
                )
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.MARKDOWN, chunk_size=2000, chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
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

    code_language = extToLanguage(file_ext)

    inspectionTask.metadataSource = texts_global[inspectionTask.iterationNum].metadata
    inspectionTask.messageContent = ""  # will be filled in with request
    inspectionTask.beginingWith = beginingCodeSnippet(texts_global[inspectionTask.iterationNum].page_content)
    inspectionTask.codeLanguage = code_language
    return inspectionTask

@app.post("/retrieveInspectionsStream/")
async def processInspectionTaskStream(inspectionTask: InspectionTask):
    source_value = texts_global[inspectionTask.iterationNum].metadata['source']
    file_ext = source_value.rsplit('.', 1)[-1]

    code_language = extToLanguage(file_ext)

    client = OpenAI(
        api_key = os.getenv("API_KEY"),
        base_url = os.getenv("BASE_URL")
    )

    try:
        system_prompt = """
        Respond with a valid JSON string only, without any additional text or explanations. The output should adhere to the following structure:
        ```
        {
            "security_vulnerabilities": [
                {
                    "vulnerability": "<string>",
                    "description": "<string>",
                    "confidence_level": "<string>"
                }
            ]
        }
        ```
        Do not include any introductory phrases or sentences in your response. Only provide the JSON object.
        """
        # Verbose prompt...
        #'content': 'Give a short and concise response. Analyze the following ' + code_language + ' code for security vulnerabilities using OWASP guidelines and report any issues found in a brief summary. If there are no serious vulnerabilities found, simply say that none were found. Here is the code: \n\n ' + texts_global[inspectionTask.iterationNum].page_content
 
        chat_completion = client.chat.completions.create(
            messages=[
            {
                'role': 'system', 
                'content': system_prompt,
            },
            {
                'role': 'user', 
                'content': 'Your task is to analyze the following ' + code_language + ' code for security vulnerabilities thinking through all lines of code carefully using OWASP guidelines and report any issues found in a brief summary. Ensure that your answers are not false positives. Listing any results that is not clearly a serious code vulnerability will result in a lower score on your assignment. If there are no serious vulnerabilities found, simply say that none were found. Here is the code: \n\n ' + texts_global[inspectionTask.iterationNum].page_content
            }
        ],
        model = os.getenv("LLM_MODEL"),
        stream=False,
        temperature=0.3,
        top_p=0.2,
        response_format={
            "type": "json_object"
        }
        )
    except Exception as e:
        print("Chat Completion Error")

    # TODO: Reomve this print statement
    print("=====")
    print(texts_global[inspectionTask.iterationNum].page_content)
    print("=====")
    print(chat_completion.choices[0].message.content)

    # TODO: Account for chat_completion.choices[0].message.content being None
    inspectionTask.metadataSource = texts_global[inspectionTask.iterationNum].metadata
    inspectionTask.beginingWith = beginingCodeSnippet(texts_global[inspectionTask.iterationNum].page_content)
    inspectionTask.codeLanguage = code_language
    response_messages = chat_completion.choices[0].message.content

    findings_count = appendToDataFrame(inspectionTask.metadataSource, texts_global[inspectionTask.iterationNum].page_content, inspectionTask.codeLanguage, response_messages) #texts_global[inspectionTask.iterationNum].page_content
    #print(df[['vulnerability']].head(10))

    # convert the pandas dataframe column, 'vulnerability', to a json object
    grouped_df = df.groupby('vulnerability').size().reset_index(name='count')
    json_result = grouped_df.to_json(orient='records')
    print(json_result)

    # return StreamingResponse(str(findings_count), media_type="text/event-stream")
    return StreamingResponse(json_result, media_type="text/event-stream")


@app.post("/downloadCSV/")
async def processdownloadCSV(downloadCSVTask: DownloadCSVTask):
    today = date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    file_name = "vulnerability_report_" + formatted_date + ".csv"
    if downloadCSVTask.selectedText != "":
        file_name = downloadCSVTask.selectedText + ".csv"
    global df
    df.to_csv(file_name, index=False)
    downloadCSVTask.messageContent = "Downloaded CSV file named: " + file_name
    return downloadCSVTask


def appendToDataFrame(metadataSource, beginingWith, codeLanguage, json_response) -> int:
    try:
        source_string = ""
        for item in metadataSource:
            if item == "source":
                source_string = metadataSource[item]
        data = json.loads(json_response)
        if data['security_vulnerabilities']:
            new_rows = []
            for item in data['security_vulnerabilities']:
                translated_vulnerability = translateVulnerabilityText(item['vulnerability'])
                new_row ={
                    'metadataSource': source_string, 
                    'beginingWith': beginingWith, 
                    'codeLanguage': codeLanguage, 
                    'vulnerability': translated_vulnerability, 
                    'description': item['description'],
                    'confidenceLevel': item['confidence_level']
                }
                new_rows.append(new_row)
            global df
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            return len(new_rows)
        else:
            print("No vulnerabilities found")
            new_rows = []
            new_row ={
                'metadataSource': source_string, 
                'beginingWith': beginingWith, 
                'codeLanguage': codeLanguage, 
                'vulnerability': "na", 
                'description': "na",
                'confidenceLevel': "na"
            }
            new_rows.append(new_row)
            #global df
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            return 0
    except Exception as e:
        print("Error appending to DataFrame")
        return -1

def translateVulnerabilityText(original_text) -> str:
    translatedText = original_text
    if original_text == "Cross-Site Scripting" or original_text == "XSS":
        translatedText = "Cross-Site Scripting (XSS)"
    if original_text == "Hardcoded Credentials" or original_text == "Hardcoded Password":
        translatedText = "Hardcoded Sensitive Data"

    if "XSS" in original_text:
        translatedText = "Cross-Site Scripting (XSS)"

    validation_phrases = [
        "Input Validation", "Lack of Input Validation", "Improper Input Validation",
        "Insufficient Input Validation", "Insufficient Validation of User Input",
        "Lack of Input Validation and Sanitization", "Use of Unvalidated User Input",
        "User Input Validation", "No Input Validation", "Unvalidated User Input"
    ]    
    if original_text in validation_phrases:
        translatedText = "Unvalidated user input"
    return translatedText

def process_files(ext, repo_path, language, use_parser_language):
    if use_parser_language == True:
        loader = GenericLoader.from_filesystem(
            repo_path + "/",
            glob="**/*",
            suffixes=[f".{ext}"],
            exclude=[f"**/non-utf8-encoding.{ext}"],
            parser=LanguageParser(language=language, parser_threshold=500),
        )
    if use_parser_language == False:
        loader = GenericLoader.from_filesystem(
            repo_path + "/",
            glob="**/*",
            suffixes=[f".{ext}"],
            exclude=[f"**/non-utf8-encoding.{ext}"],
            parser=LanguageParser(parser_threshold=500),
        )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=2000, chunk_overlap=200
    )
    return splitter.split_documents(documents)


def beginingCodeSnippet(page_content):
    firstFiftyChars = str(page_content)[:50]
    listSplitOnCarriageReturn = firstFiftyChars.splitlines(True)
    # firstLine = listSplitOnCarriageReturn[0]
    return listSplitOnCarriageReturn[0]


def extToLanguage(file_ext):
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
    if (file_ext == "env"):
        code_language = "Environment Variables"
    if (file_ext == "Dockerfile"):
        code_language = "Docker"
    return code_language# ...
