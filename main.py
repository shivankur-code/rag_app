import ast, astunparse
import json
import re
import tokenize
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
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
                left = node.value.left.s if isinstance(node.value.left, ast.Str) else self.variable_values.get(node.value.left.id, '')
                right = node.value.right.s if isinstance(node.value.right, ast.Str) else self.variable_values.get(node.value.right.id, '')
                self.variable_values[var_name] = left + right
            elif isinstance(node.value, ast.Call) and getattr(node.value.func, 'attr', '') == 'format':
                if isinstance(node.value.func.value, ast.Str):
                    self.variable_values[var_name] = node.value.func.value.s
        self.generic_visit(node)

    def get_full_string(self, node):
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.get_full_string(node.left) + self.get_full_string(node.right)
        elif isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'format':
            if isinstance(node.func.value, ast.Str):
                return node.func.value.s
        elif isinstance(node, ast.Name) and node.id in self.variable_values:
            return self.variable_values[node.id]
        return None

    def visit_Raise(self, node):
        if isinstance(node.exc, ast.Call) and getattr(node.exc.func, 'id', '') == 'MagnetoInvalidArgumentError':
            error_arg = node.exc.args[0]
            error_message = self.get_full_string(error_arg)
            if error_message:
                self.errors.append((self.current_function, error_message))
        self.generic_visit(node)

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

def ask_ollama_for_shorthand(context: str, error_message: str) -> str:
    ollama = Ollama(base_url='http://localhost:11434', model="mistral")
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

def extract_errors_from_python_file(filename: str) -> List[Dict[str, str]]:
    with open(filename, 'r') as file:
        tree = ast.parse(file.read())
    extractor = ErrorStringExtractor()
    extractor.visit(tree)
    return extractor.errors

class ErrorStringReplacer(ast.NodeTransformer):
    def __init__(self, replacements: Dict[str, str]):
        self.replacements = replacements

    def get_full_string(self, node):
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.get_full_string(node.left) + self.get_full_string(node.right)
        elif isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'format':
            if isinstance(node.func.value, ast.Str):
                return node.func.value.s
        return None

    def visit_Raise(self, node):
        if isinstance(node.exc, ast.Call) and getattr(node.exc.func, 'id', '') == 'MagnetoInvalidArgumentError':
            error_arg = node.exc.args[0]
            error_message = self.get_full_string(error_arg)
            if error_message and error_message in self.replacements:
                new_node = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='error_messages', ctx=ast.Load()),
                        attr=self.replacements[error_message],
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[]
                )
                node.exc.args[0] = new_node
        return self.generic_visit(node)

def update_error_message_ast(filename: str, replacements: Dict[str, str]):
    with open(filename, 'r') as file:
        original_code = file.read()

    tree = ast.parse(original_code)

    replacer = ErrorStringReplacer(replacements)
    modified_tree = replacer.visit(tree)

    modified_code = astunparse.unparse(modified_tree)

    with open(filename, 'w') as file:
        file.write(astunparse.unparse(modified_tree))

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

    update_error_message_ast(filename, replacements)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
    else:
        main(sys.argv[1])
