import ast
import builtins

class VariableNormalizer(ast.NodeTransformer):
    def __init__(self):
        self.name_map = {}
        self.counter = 0
        # Get all built-in names that should NOT be renamed
        self.protected_names = set(dir(builtins)) | {'self', 'cls', 'Solution'}
        
    def _get_normalized_name(self, name):
        # Don't rename built-ins, special names, or already protected names
        if name in self.protected_names or name.startswith('_'):
            return name
        if name not in self.name_map:
            self.name_map[name] = f"var_{self.counter}"
            self.counter += 1
        return self.name_map[name]
    
    def visit_Name(self, node):
        # Only rename user-defined variables, not built-in references
        if isinstance(node.ctx, (ast.Store, ast.Load)) and node.id not in self.protected_names:
            node.id = self._get_normalized_name(node.id)
        return node
    
    def visit_arg(self, node):
        node.arg = self._get_normalized_name(node.arg)
        return node
    
    def visit_FunctionDef(self, node):
        node.name = self._get_normalized_name(node.name)
        self.generic_visit(node)
        return node
    
    def visit_ClassDef(self, node):
        node.name = self._get_normalized_name(node.name)
        self.generic_visit(node)
        return node

def normalize_whitespace(code_str):
    # Strip leading/trailing whitespace, normalize line endings, remove empty lines
    try:
        lines = code_str.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        non_empty_lines = [line.rstrip() for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)
    except Exception as e:
        return code_str.strip()

def strip_comments(code_str):
    # Remove comments and docstrings while preserving code structure
    try:
        tree = ast.parse(code_str)
        
        # Remove docstrings from functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                    node.body = node.body[1:] if len(node.body) > 1 else [ast.Pass()]
        
        # Remove module-level docstrings
        if (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, (ast.Constant, ast.Str))):
            tree.body = tree.body[1:]
        
        ast.fix_missing_locations(tree)
        code_no_docstrings = ast.unparse(tree)
        
        # Remove inline comments
        lines = code_no_docstrings.split('\n')
        lines_no_comments = []
        for line in lines:
            if '#' in line:
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                    elif char == '#' and not in_string:
                        line = line[:i].rstrip()
                        break
            lines_no_comments.append(line)
        
        return '\n'.join(lines_no_comments)
    except Exception as e:
        return code_str

def format_normalize(code_str):
    # Normalize basic formatting: indentation and spacing
    try:
        tree = ast.parse(code_str)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        return code_str.strip()

def canonicalize_code(code_str):
    try:
        tree = ast.parse(code_str)
        
        # Remove docstrings from functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                    node.body = node.body[1:] if len(node.body) > 1 else [ast.Pass()]
        
        # Remove module-level docstrings
        if (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, (ast.Constant, ast.Str))):
            tree.body = tree.body[1:]
        
        # Normalize variable names
        normalizer = VariableNormalizer()
        tree = normalizer.visit(tree)
        
        # Fix missing locations after transformation
        ast.fix_missing_locations(tree)
        
        # Convert back to code
        canonical = ast.unparse(tree)
        
        return canonical.strip()
    except Exception as e:
        return code_str.strip()

