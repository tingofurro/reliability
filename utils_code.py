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

