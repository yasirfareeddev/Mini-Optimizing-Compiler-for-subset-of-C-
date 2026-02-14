from flask import Flask, request, jsonify
from flask_cors import CORS
import compiler  # your compiler.py file

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

def ast_to_dict(node):
    """
    Serialize AST to dict recursively
    """
    if node is None:
        return None
    d = {"type": type(node).__name__}
    if isinstance(node, compiler.Program):
        d["statements"] = [ast_to_dict(s) for s in node.statements]
    elif isinstance(node, compiler.VarDecl):
        d["var_type"] = node.var_type
        d["name"] = node.name
        d["init_expr"] = ast_to_dict(node.init_expr)
    elif isinstance(node, compiler.Assign):
        d["name"] = node.name
        d["expr"] = ast_to_dict(node.expr)
    elif isinstance(node, compiler.Print):
        d["parts"] = [ast_to_dict(p) for p in node.parts]
    elif isinstance(node, compiler.If):
        d["cond"] = ast_to_dict(node.cond)
        d["then_block"] = ast_to_dict(node.then_block)
        d["else_block"] = ast_to_dict(node.else_block)
    elif isinstance(node, compiler.While):
        d["cond"] = ast_to_dict(node.cond)
        d["body"] = ast_to_dict(node.body)
    elif isinstance(node, compiler.Block):
        d["statements"] = [ast_to_dict(s) for s in node.statements]
    elif isinstance(node, compiler.BinaryOp):
        d["op"] = node.op
        d["left"] = ast_to_dict(node.left)
        d["right"] = ast_to_dict(node.right)
    elif isinstance(node, compiler.UnaryOp):
        d["op"] = node.op
        d["operand"] = ast_to_dict(node.operand)
    elif isinstance(node, compiler.Literal):
        d["value"] = node.value
        d["typ"] = node.typ
    elif isinstance(node, compiler.Variable):
        d["name"] = node.name
    return d

@app.route("/compile", methods=["POST"])
def compile_code():
    data = request.get_json()
    code = data.get("code", "")
    try:
        result = compiler.compile_source(code, verbose=False)
        
        # Process tokens to match terminal format
        processed_tokens = []
        for token in result['tokens']:
            if token.type != 'EOF':
                processed_tokens.append({
                    "type": token.type,
                    "value": token.value,
                    "lineno": token.lineno
                })
        
        # Process AST
        ast_dict = ast_to_dict(result['ast']) if result['ast'] else {}
        
        # Process TAC
        tac_list = [repr(t) for t in result['tac']] if result['tac'] else []
        optimized_tac_list = [repr(t) for t in result['optimized_tac']] if result['optimized_tac'] else []
        
        # Process assembly
        assembly_list = result['asm'] if result['asm'] else []
        
        # Process output
        output_list = result['output'] if result['output'] else []
        
        # Prepare response
        response = {
            "tokens": processed_tokens,
            "ast": ast_dict,
            "tac": tac_list,
            "optimized_tac": optimized_tac_list,
            "assembly": assembly_list,
            "output": output_list,
            "errors": result['errors'],
            "symbol_table": result['symbol_table'],
            "memory": result['memory']
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "tokens": [],
            "tac": [],
            "optimized_tac": [],
            "assembly": [],
            "ast": {},
            "output": [],
            "errors": [f"Unexpected error: {str(e)}"],
            "symbol_table": {},
            "memory": {}
        }), 500

if __name__ == "__main__":
    app.run(debug=True)