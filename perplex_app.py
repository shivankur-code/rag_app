import ast
import json
import re
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict
from langchain_community.llms import Ollama

class ErrorStringExtractor(ast.NodeVisitor):
    def __init__(self):
        self.errors = []
        self.current_function = None
        self.variable_values = {}
    
    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.variable_values = {}  # Reset variable values for each function
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if isinstance(node.value, ast.Str):
                self.variable_values[var_name] = node.value.s
            elif isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                # Handle concatenation of string literals
                left = node.value.left.s if isinstance(node.value.left, ast.Str) else self.variable_values.get(node.value.left.id, '')
                right = node.value.right.s if isinstance(node.value.right, ast.Str) else self.variable_values.get(node.value.right.id, '')
                self.variable_values[var_name] = left + right
            elif isinstance(node.value, ast.Call) and getattr(node.value.func, 'attr', '') == 'format':
                # Handle formatted string with placeholders
                if isinstance(node.value.func.value, ast.Str):
                    self.variable_values[var_name] = node.value.func.value.s
        self.generic_visit(node)
    
    def visit_Raise(self, node):
        if isinstance(node.exc, ast.Call) and getattr(node.exc.func, 'id', '') == 'MagnetoInvalidArgumentError':
            error_arg = node.exc.args[0]
            error_message = None
            if isinstance(error_arg, ast.Str):
                error_message = error_arg.s
            elif isinstance(error_arg, ast.Name) and error_arg.id in self.variable_values:
                error_message = self.variable_values[error_arg.id]
            elif isinstance(error_arg, ast.Call) and getattr(error_arg.func, 'attr', '') == 'format':
                if isinstance(error_arg.func.value, ast.Str):
                    error_message = error_arg.func.value.s
            if error_message:
                self.errors.append((self.current_function, error_message))
        self.generic_visit(node)

# def parse_python_file(filename: str) -> List[Dict]:
#     with open(filename, 'r') as file:
#         tree = ast.parse(file.read())
#     extractor = ErrorStringExtractor()
#     extractor.visit(tree)
#     return extractor.errors

def get_or_create_collection(client, collection_name, embedding_fn):
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        collection = client.create_collection(name=collection_name, embedding_function=embedding_fn)
    return collection

def save_chunks_to_vector_store(chunks: List[Dict], collection_name='code_chunks'):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="."))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-MiniLM-L6-v2")
    collection = get_or_create_collection(client, collection_name, embedding_fn)
    
    for i, chunk in enumerate(chunks):
        embedding = embedding_fn(chunk['code'])[0]  # Ensure embedding is a single list
        collection.add(embeddings=[embedding], ids=[str(i)], metadatas=[{'function': chunk['function']}])
    return collection

def find_context_for_error(collection, function_name: str):
    results = collection.query(query_texts=[function_name], n_results=1)
    if results and results['metadatas']:
        return results['metadatas'][0][0]['function']
    return ''

# def update_python_file_with_shorthand(filename: str, replacements: Dict[str, str]):
#     with open(filename, 'r') as file:
#         code = file.read()
#     for original, shorthand in replacements.items():
#         code = code.replace(original, shorthand)
#     with open(filename, 'w') as file:
#         file.write(code)

# def main(filename: str):
#     # Step 1: Parse the Python file
#     errors = parse_python_file(filename)
    
#     # Step 2: Save chunks to vector store
#     chunks = [{'function': func, 'code': code} for func, code in errors]
#     collection = save_chunks_to_vector_store(chunks)
    
#     # Step 3: Process each error message
#     replacements = {}
#     output = {}
#     for func, error_message in errors:
#         context = find_context_for_error(collection, func)
#         shorthand = ask_ollama_for_shorthand(context, error_message)
#         replacements[error_message] = shorthand
#         output[shorthand] = error_message
    
#     rev_map = {v: k for k, v in output.items()}
#     # Step 4: Write to output.json
#     with open('output.json', 'w') as json_file:
#         json.dump(output, json_file, indent=4)
    
#     # Step 5: Update Python file with shorthand error messages
#     # update_python_file_with_shorthand(filename, replacements)
#     replace_error_messages(filename, rev_map)


# Function to ask Ollama for shorthand
def ask_ollama_for_shorthand(context: str, error_message: str) -> str:
    ollama = Ollama(base_url='http://localhost:11434', model="llama3")
    prompt = (
        f"Generate a shorthand for the following error message within the context of '{context}'. "
        f"The shorthand should be descriptive, concise, and use uppercase with underscores. "
        f"Example: for the error message \"'vm_uuid' is a required field for revert of a vm.\", "
        f"the shorthand could be VM_UUID_REQUIRED. "
        f"Error message: '{error_message}'"
    )
    response = ollama.invoke(prompt)
    print(f"Ollama response: {response}")
    shorthand = re.findall(r'\b[A-Z_]{3,}\b', response)
    if shorthand:
        return shorthand[0]
    else:
        print(f"Failed to extract shorthand for: {error_message}")
        return response.strip()

# Custom NodeTransformer to replace error messages
class ErrorStringReplacer(ast.NodeTransformer):
    def __init__(self, rev_map: Dict[str, str]):
        self.rev_map = rev_map

    def visit_Str(self, node):
        if node.s in self.rev_map:
            return ast.copy_location(ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='error_messages', ctx=ast.Load()),
                    attr=self.rev_map[node.s],
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            ), node)
        return node

# Function to parse and replace error messages in the Python file
def parse_and_replace_python_file(filename: str, rev_map: Dict[str, str]):
    with open(filename, 'r') as file:
        tree = ast.parse(file.read())

    replacer = ErrorStringReplacer(rev_map)
    modified_tree = replacer.visit(tree)
    modified_code = ast.unparse(modified_tree)

    with open(filename, 'w') as file:
        file.write(modified_code)

# Function to extract errors from Python file
def extract_errors_from_python_file(filename: str) -> List[Dict[str, str]]:
    with open(filename, 'r') as file:
        tree = ast.parse(file.read())
    extractor = ErrorStringExtractor()
    extractor.visit(tree)
    return extractor.errors
    
# Main function to process the file and update error messages
def main(filename: str):
    errors = extract_errors_from_python_file(filename)
    chunks = [{'function': func, 'code': code} for func, code in errors]
    collection = save_chunks_to_vector_store(chunks)

    replacements = {}
    for func, error_message in errors:
        context = find_context_for_error(collection, func)
        shorthand = ask_ollama_for_shorthand(context, error_message)
        replacements[error_message] = shorthand

    with open('output.json', 'w') as json_file:
        json.dump(replacements, json_file, indent=4)

    update_error_message_lines(filename, replacements)

# Function to update specific error message lines in the Python file
def update_error_message_lines(filename: str, replacements: Dict[str, str]):
    with open(filename, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        original_line = line
        for error_message, shorthand in replacements.items():
            if error_message in line:
                line = line.replace(f'"{error_message}"', "error_messages.{}".format(shorthand))
                line = line.replace(f"'{error_message}'", "error_messages.{}".format(shorthand))
        # original_line = line
        # for error_message, shorthand in replacements.items():
        #     if error_message in line:
        #         line = re.sub(rf'([uU]?)("{error_message}"|\'{error_message}\')', rf'error_messages.{shorthand}', line)
        # new_lines.append(line)

    with open(filename, 'w') as file:
        file.writelines(new_lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
    else:
        main(sys.argv[1])


