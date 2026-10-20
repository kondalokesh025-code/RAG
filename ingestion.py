from langchain_core.documents import Document
import requests 
import json 
import base64
import os 
from dotenv import load_dotenv
from pathlib import Path 
from language_utils import normalize_language 

load_dotenv()
token = os.getenv("GITHUB_TOKEN")
headers = {"Authorization" : f"Bearer {token}"}



FILE_MAP = {
    # Programming languages
    ".py": ("code", "python"),
    ".pyw": ("code", "python"),
    ".pyi": ("code", "python"),
    ".c": ("code", "c"),
    ".h": ("code", "c"),
    ".cpp": ("code", "cpp"),
    ".cc": ("code", "cpp"),
    ".cxx": ("code", "cpp"),
    ".hpp": ("code", "cpp"),
    ".hh": ("code", "cpp"),
    ".go": ("code", "go"),
    ".rs": ("code", "rust"),
    ".rb": ("code", "ruby"),
    ".php": ("code", "php"),
    ".cs": ("code", "csharp"),
    ".swift": ("code", "swift"),
    ".kt": ("code", "kotlin"),
    ".kts": ("code", "kotlin"),
    ".java": ("code", "java"),
    ".js": ("code", "javascript"),
    ".jsx": ("code", "javascript"),
    ".mjs": ("code", "javascript"),
    ".cjs": ("code", "javascript"),
    ".ts": ("code", "typescript"),
    ".tsx": ("code", "typescript"),
    ".scala": ("code", "scala"),
    ".sc": ("code", "scala"),
    ".dart": ("code", "dart"),
    ".r": ("code", "r"),
    ".lua": ("code", "lua"),
    ".pl": ("code", "perl"),
    ".pm": ("code", "perl"),
    ".sh": ("code", "shell"),
    ".bash": ("code", "shell"),
    ".zsh": ("code", "shell"),
    ".fish": ("code", "fish"),
    ".ps1": ("code", "powershell"),
    ".bat": ("code", "batch"),
    ".cmd": ("code", "batch"),
    ".sql": ("code", "sql"),
    ".ex": ("code", "elixir"),
    ".exs": ("code", "elixir"),
    ".erl": ("code", "erlang"),
    ".hrl": ("code", "erlang"),
    ".hs": ("code", "haskell"),
    ".lhs": ("code", "haskell"),
    ".clj": ("code", "clojure"),
    ".cljs": ("code", "clojure"),
    ".cljc": ("code", "clojure"),
    ".elm": ("code", "elm"),
    ".f": ("code", "fortran"),
    ".for": ("code", "fortran"),
    ".f90": ("code", "fortran"),
    ".jl": ("code", "julia"),
    ".ml": ("code", "ocaml"),
    ".mli": ("code", "ocaml"),
    ".plsql": ("code", "sql"),
    ".sol": ("code", "solidity"),
    ".vue": ("code", "vue"),
    ".svelte": ("code", "svelte"),
    ".gradle": ("code", "groovy"),
    ".groovy": ("code", "groovy"),
    ".v": ("code", "verilog"),
    ".sv": ("code", "systemverilog"),
    ".zig": ("code", "zig"),
    ".nim": ("code", "nim"),
    ".vhd": ("code", "vhdl"),
    ".vhdl": ("code", "vhdl"),
    ".css": ("code", "css"),
    ".scss": ("code", "scss"),
    ".sass": ("code", "sass"),
    ".less": ("code", "less"),
    ".graphql": ("code", "graphql"),
    ".gql": ("code", "graphql"),
    ".proto": ("code", "protobuf"),
    ".nix": ("code", "nix"),

    # Markup and documentation
    ".md": ("markup", "markdown"),
    ".mdx": ("markup", "markdown"),
    ".rst": ("markup", "rst"),
    ".html": ("markup", "html"),
    ".htm": ("markup", "html"),
    ".xml": ("markup", "xml"),
    ".svg": ("markup", "xml"),

    # Notebooks
    ".ipynb": ("notebook", None),

    # Configuration and data formats
    ".json": ("config", "json"),
    ".jsonc": ("config", "json"),
    ".json5": ("config", "json"),
    ".yaml": ("config", "yaml"),
    ".yml": ("config", "yaml"),
    ".toml": ("config", "toml"),
    ".ini": ("config", "ini"),
    ".cfg": ("config", "ini"),
    ".conf": ("config", "text"),
    ".properties": ("config", "properties"),
    ".env": ("config", "text"),
    ".txt": ("config", "text"),
    ".csv": ("config", "csv"),
    ".tsv": ("config", "text"),
    ".tf": ("config", "hcl"),
    ".tfvars": ("config", "hcl"),
    ".hcl": ("config", "hcl"),
    ".make": ("config", "makefile"),
    ".mk": ("config", "makefile"),

    # Common extensionless filenames
    "readme": ("markup", "markdown"),
    "license": ("markup", "text"),
    "copying": ("markup", "text"),
    "notice": ("markup", "text"),
    "changelog": ("markup", "markdown"),
    "contributing": ("markup", "markdown"),
    "dockerfile": ("config", "dockerfile"),
    "containerfile": ("config", "dockerfile"),
    "makefile": ("config", "makefile"),
    "gemfile": ("config", "ruby"),
    "rakefile": ("config", "ruby"),
    "procfile": ("config", "text"),
    "justfile": ("config", "text"),
    "taskfile": ("config", "yaml"),
    ".gitignore": ("config", "text"),
    ".gitattributes": ("config", "text"),
    ".dockerignore": ("config", "text"),
}




def get_repo_metadata(owner,repo):

    try : 
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response  = requests.get(url  , headers = headers , timeout = (5  , 5 )) ## timeout( connect , read ) 
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e : 
        return {"error" : str(e)}
    
def get_repo_tree( owner , repo , branch  ): 
    try : 
        
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}" 
        response = requests.get(url , headers = headers , params= {"recursive" : "1"} , timeout = (5 , 5 )) 
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e : 
        return {"error" : str(e)} 


def filter_repo_files (data) :
    files = []
    for file in data["tree"] : 
        if ( file["type"] == "blob"   ):
            files.append(file)

    return files 


def classify_file (file ) : 
    file_path  = file["path"]
    file_name = Path(file_path).name.lower()
    extension = Path(file_path).suffix.lower()


    category , language = FILE_MAP.get(
        file_name , FILE_MAP.get( extension , ("unknown" , None ))
    )

    return {
        "path" : file_path , 
        "category" : category , 
        "language" : language 
                }


def get_file_content(owner , repo , file_path ):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    try :
        response = requests.get(url = url , headers = headers  , timeout = (3 , 5) )
        response.raise_for_status()
        response = response.json()
        content = response["content"]
        encoded_content = content.encode('utf-8')
        bytes = base64.b64decode(encoded_content)
        return bytes 
    except Exception as e : 
        return None 

def process_file(file, raw_content):
    try:
        content = raw_content.decode("utf-8")

        if file["category"] == "notebook":
            return json.loads(content)

        return content

    except (UnicodeDecodeError, json.JSONDecodeError):
        return None 




def ingest_repository( owner , repo ) :
    data = get_repo_metadata(owner, repo)

    if "error" in data:
        print(data["error"])
        raise SystemExit
    
    
    data1 = get_repo_tree(owner , repo , branch = data["default_branch"]) 

    if "error" in data1:
        print(data1["error"])
        raise SystemExit


    data2 = filter_repo_files(data1)
    classifed_files = []
    for file in data2 :
        result = classify_file(file)
        classifed_files.append(result)
    documents =  []
    for file in classifed_files : 
        if ( file["category"] == "unknown") : 
            continue 
        file_bytes  = get_file_content(owner  , repo , file["path"])
        if file_bytes is None : 
            continue
        content = process_file(file , file_bytes )
        if content is None : 
            continue 
      
        if isinstance(content, dict):
            notebook_metadata = content.get("metadata") or {}
            language_info = notebook_metadata.get("language_info") or {}
            kernelspec = notebook_metadata.get("kernelspec") or {}

            notebook_language = normalize_language(
                language_info.get("name") or kernelspec.get("language")
            )

            for cell_index, cell in enumerate(content.get("cells", [])):
                cell_type = cell.get("cell_type")
                source = cell.get("source", "")

                if isinstance(source, list):
                    source = "".join(source)

                if not source.strip():
                    continue

                if cell_type == "code":
                    category = "code"
                    language = notebook_language
                elif cell_type == "markdown":
                    category = "markup"
                    language = "markdown"
                else:
                    continue

                documents.append(Document(
                    page_content=source,
                    metadata={
                        "source": file["path"],
                        "category": category,
                        "language": language,
                        "cell_type": cell_type,
                        "cell_index": cell_index,
                    },
                ))



        if isinstance ( content , str ) : 
            doc = Document(page_content=  content  , metadata = {"source" : file["path"] , "category" : file["category"]  , "language" : file["language"] })
            documents.append( doc )
    return documents 
    


