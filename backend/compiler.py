#!/usr/bin/env python3
"""
mini_compiler_fixed.py
Single-file educational compiler pipeline (lexer → LL(1) parser → semantic analysis
→ TAC IR → simple optimizations → assembly + interpreter).

Fix: parser now recognizes function headers like `int main() { ... }` and won't
mistake them for variable declarations (no bogus "missing semicolon" error).
"""

import re
import sys
from collections import namedtuple, defaultdict, deque
from copy import deepcopy

# =====================================================
# GLOBAL HELPERS (temps, labels, errors)
# =====================================================
temp_count = 0
label_count = 0
errors = []

def error(phase, msg, lineno=None):
    if lineno is not None:
        errors.append(f"{phase} error (line {lineno}): {msg}")
    else:
        errors.append(f"{phase} error: {msg}")

def new_temp():
    global temp_count
    temp_count += 1
    return f"t{temp_count}"

def new_label():
    global label_count
    label_count += 1
    return f"L{label_count}"

# =====================================================
# LEXER
# =====================================================
Token = namedtuple('Token', ['type', 'value', 'lineno'])

class Lexer:
    KEYWORDS = {'int','float','char','string','auto','if','else','while','return','cout','cin','for','break','continue','true','false'}
    token_specification = [
        ("COMMENT",   r'//.*'),
        ("NUMBER",    r'\d+\.\d+|\d+'),           # float or int
        ("STRING",    r'"([^"\\]|\\.)*"'),
        ("CHAR",      r'\'([^\'\\]|\\.)\''),
        ("SHIFT",     r'<<'),
        ("ID",        r'[A-Za-z_]\w*'),
        ("OP",        r'\+|\-|\*|/|%|==|!=|<=|>=|<|>'),
        ("ASSIGN",    r'='),
        ("END",       r';'),
        ("LPAREN",    r'\('),
        ("RPAREN",    r'\)'),
        ("LBRACE",    r'\{'),
        ("RBRACE",    r'\}'),
        ("COMMA",     r','),
        ("SKIP",      r'[ \t]+'),
        ("NEWLINE",   r'\n'),
        ("MISMATCH",  r'.'),
    ]
    tok_regex = '|'.join(f'(?P<{n}>{p})' for n,p in token_specification)
    master_re = re.compile(tok_regex)

    def __init__(self, code):
        self.code = code
        self.lineno = 1
        self.tokens = []
        self._tokenize()

    def _tokenize(self):
        for mo in self.master_re.finditer(self.code):
            kind = mo.lastgroup
            val = mo.group()
            if kind == "NUMBER":
                self.tokens.append(Token('NUMBER', val, self.lineno))
            elif kind == "STRING":
                self.tokens.append(Token('STRING', val, self.lineno))
            elif kind == "CHAR":
                self.tokens.append(Token('CHAR', val, self.lineno))
            elif kind == "ID":
                if val in Lexer.KEYWORDS:
                    self.tokens.append(Token(val.upper(), val, self.lineno))
                else:
                    self.tokens.append(Token('ID', val, self.lineno))
            elif kind == "SHIFT":
                self.tokens.append(Token('SHIFT', val, self.lineno))
            elif kind == "OP":
                self.tokens.append(Token('OP', val, self.lineno))
            elif kind == "ASSIGN":
                self.tokens.append(Token('ASSIGN', val, self.lineno))
            elif kind == "END":
                self.tokens.append(Token('END', val, self.lineno))
            elif kind == "LPAREN":
                self.tokens.append(Token('LPAREN', val, self.lineno))
            elif kind == "RPAREN":
                self.tokens.append(Token('RPAREN', val, self.lineno))
            elif kind == "LBRACE":
                self.tokens.append(Token('LBRACE', val, self.lineno))
            elif kind == "RBRACE":
                self.tokens.append(Token('RBRACE', val, self.lineno))
            elif kind == "COMMA":
                self.tokens.append(Token('COMMA', val, self.lineno))
            elif kind == "NEWLINE":
                self.lineno += 1
            elif kind == "SKIP" or kind == "COMMENT":
                pass
            elif kind == "MISMATCH":
                error("Lexical", f"Unexpected character {val!r}", self.lineno)
            else:
                self.tokens.append(Token(kind, val, self.lineno))

    def peek_all(self):
        return list(self.tokens)

# =====================================================
# AST NODES
# =====================================================
class Node: pass

class Program(Node):
    def __init__(self, statements):
        self.statements = statements

class VarDecl(Node):
    def __init__(self, var_type, name, init_expr=None, lineno=None):
        self.var_type = var_type  # 'int' | 'float' | 'char' | 'string' | 'auto'
        self.name = name
        self.init_expr = init_expr
        self.lineno = lineno

class Assign(Node):
    def __init__(self, name, expr, lineno=None):
        self.name = name
        self.expr = expr
        self.lineno = lineno

class Print(Node):  # cout << ... ;
    def __init__(self, parts, lineno=None):
        self.parts = parts
        self.lineno = lineno

class If(Node):
    def __init__(self, cond, then_block, else_block=None, lineno=None):
        self.cond = cond
        self.then_block = then_block
        self.else_block = else_block
        self.lineno = lineno

class While(Node):
    def __init__(self, cond, body, lineno=None):
        self.cond = cond
        self.body = body
        self.lineno = lineno

class Block(Node):
    def __init__(self, statements):
        self.statements = statements

class BinaryOp(Node):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class UnaryOp(Node):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

class Literal(Node):
    def __init__(self, value, typ):
        self.value = value
        self.typ = typ  # 'int', 'float', 'string', 'char', 'bool'

class Variable(Node):
    def __init__(self, name):
        self.name = name

# =====================================================
# PARSER (recursive-descent, LL(1) style)
# =====================================================
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token('EOF','',None)

    def peek_n(self, n):
        idx = self.pos + n
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token('EOF','',None)

    def advance(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, ttype, msg=None):
        tok = self.peek()
        if tok.type == ttype:
            return self.advance()
        else:
            m = msg or f"Expected token {ttype} but got {tok.type}"
            error("Syntax", m, tok.lineno)
            raise SyntaxError(m)

    def parse(self):
        stmts = []
        while self.peek().type != 'EOF':
            s = self.statement()
            if s is not None:
                # If s is a Block that came from a function definition, expand it
                if isinstance(s, Block):
                    # Put the statements inside the block into top-level
                    stmts.extend(s.statements)
                else:
                    stmts.append(s)
        return Program(stmts)

    def statement(self):
        tok = self.peek()
        if tok.type in ('INT','FLOAT','CHAR','STRING','AUTO'):
            return self.var_decl_or_function()
        if tok.type == 'ID':
            return self.assignment()
        if tok.type == 'COUT':
            return self.cout_statement()
        if tok.type == 'IF':
            return self.if_statement()
        if tok.type == 'WHILE':
            return self.while_statement()
        if tok.type == 'LBRACE':
            return self.block()
        # fallback: consume token to avoid infinite loop
        error("Syntax", f"unexpected token {tok.type}", tok.lineno)
        self.advance()
        return None

    def var_decl_or_function(self):
        t = self.advance()  # type token
        var_type = t.value
        lineno = t.lineno

        next_tok = self.peek()
        # detect function header: type ID '(' ... ')'
        if next_tok.type == 'ID' and self.peek_n(1).type == 'LPAREN':
            # function definition (we'll accept only parameterless functions like main())
            id_tok = self.advance()  # function name
            # consume '(' ')'
            self.expect('LPAREN', "expected '(' after function name")
            # currently we don't support parameters: require immediate ')'
            self.expect('RPAREN', "expected ')' after function parameters")
            # next must be a block
            if self.peek().type == 'LBRACE':
                func_block = self.block()
                # return the block (so its statements join the program)
                return func_block
            else:
                error("Syntax", "expected function body '{...}'", lineno)
                return None

        # otherwise it's a variable declaration
        id_tok = self.expect('ID', "expected identifier after type")
        name = id_tok.value
        init_expr = None
        if self.peek().type == 'ASSIGN':
            self.advance()
            init_expr = self.expression()
        self.expect('END', "missing semicolon after declaration")
        return VarDecl(var_type, name, init_expr, lineno)

    def assignment(self):
        idtok = self.advance()
        name = idtok.value
        lineno = idtok.lineno
        self.expect('ASSIGN', "expected '=' for assignment")
        expr = self.expression()
        self.expect('END', "missing semicolon after assignment")
        return Assign(name, expr, lineno)

    def cout_statement(self):
        tok = self.advance()  # COUT
        lineno = tok.lineno
        parts = []
        # accept sequences like: SHIFT expr SHIFT expr ... END
        while True:
            if self.peek().type == 'SHIFT':
                self.advance()
                # next is expression
                part = self.expression()
                parts.append(part)
                continue
            elif self.peek().type == 'END':
                self.advance()
                break
            else:
                # try to recover
                if self.peek().type == 'EOF':
                    break
                error("Syntax", f"unexpected token in cout: {self.peek().type}", self.peek().lineno)
                self.advance()
                if self.peek().type == 'END':
                    self.advance(); break
        return Print(parts, lineno)

    def if_statement(self):
        tok = self.advance()  # IF
        lineno = tok.lineno
        self.expect('LPAREN', "expected '(' after if")
        cond = self.expression()
        self.expect('RPAREN', "expected ')' after if condition")
        then_block = self.block()
        else_block = None
        if self.peek().type == 'ELSE':
            self.advance()
            if self.peek().type == 'LBRACE':
                else_block = self.block()
            else:
                else_block = Block([self.statement()])
        return If(cond, then_block, else_block, lineno)

    def while_statement(self):
        tok = self.advance()  # WHILE
        lineno = tok.lineno
        self.expect('LPAREN', "expected '(' after while")
        cond = self.expression()
        self.expect('RPAREN', "expected ')' after while condition")
        body = self.block()
        return While(cond, body, lineno)

    def block(self):
        self.expect('LBRACE', "expected '{' for block")
        stmts = []
        while self.peek().type != 'RBRACE' and self.peek().type != 'EOF':
            s = self.statement()
            if s is not None:
                # if a nested block is returned, keep as Block node
                stmts.append(s)
        self.expect('RBRACE', "expected '}' after block")
        return Block(stmts)

    # Expressions: precedence climbing via separate functions
    def expression(self):
        return self.equality()

    def equality(self):
        node = self.relation()
        while self.peek().type == 'OP' and self.peek().value in ('==','!='):
            op = self.advance().value
            right = self.relation()
            node = BinaryOp(op, node, right)
        return node

    def relation(self):
        node = self.additive()
        while self.peek().type == 'OP' and self.peek().value in ('<','>','<=','>='):
            op = self.advance().value
            right = self.additive()
            node = BinaryOp(op, node, right)
        return node

    def additive(self):
        node = self.multiplicative()
        while self.peek().type == 'OP' and self.peek().value in ('+','-'):
            op = self.advance().value
            right = self.multiplicative()
            node = BinaryOp(op, node, right)
        return node

    def multiplicative(self):
        node = self.unary()
        while self.peek().type == 'OP' and self.peek().value in ('*','/','%'):
            op = self.advance().value
            right = self.unary()
            node = BinaryOp(op, node, right)
        return node

    def unary(self):
        if self.peek().type == 'OP' and self.peek().value in ('-','+','!'):
            op = self.advance().value
            operand = self.unary()
            return UnaryOp(op, operand)
        return self.primary()

    def primary(self):
        tok = self.peek()
        if tok.type == 'NUMBER':
            self.advance()
            if '.' in tok.value:
                return Literal(float(tok.value), 'float')
            else:
                return Literal(int(tok.value), 'int')
        if tok.type == 'STRING':
            self.advance()
            s = tok.value[1:-1].encode('utf-8').decode('unicode_escape')
            return Literal(s, 'string')
        if tok.type == 'CHAR':
            self.advance()
            ch = tok.value[1:-1].encode('utf-8').decode('unicode_escape')
            return Literal(ch, 'char')
        if tok.type == 'ID':
            self.advance()
            return Variable(tok.value)
        if tok.type == 'LPAREN':
            self.advance()
            node = self.expression()
            self.expect('RPAREN', "missing closing parenthesis")
            return node
        error("Syntax", f"unexpected token {tok.type} in expression", tok.lineno)
        self.advance()
        return Literal(0,'int')

# =====================================================
# SEMANTIC ANALYZER + TYPE INFERENCE
# =====================================================
class SemanticAnalyzer:
    def __init__(self):
        # symbol_table: name -> type (string)
        self.symbols = {}
        # memory (runtime) initial values (None means uninitialized)
        self.memory = {}

    def analyze(self, node):
        if isinstance(node, Program):
            for s in node.statements:
                self.analyze(s)
        elif isinstance(node, VarDecl):
            if not re.match(r'^[A-Za-z_]\w*$', node.name):
                error("Lexical", f"invalid identifier '{node.name}'", node.lineno)
                return
            if node.name in self.symbols:
                error("Semantic", f"variable '{node.name}' already declared", node.lineno)
                return
            if node.var_type.lower() == 'auto':
                if node.init_expr is None:
                    error("Semantic", f"cannot infer type for '{node.name}' without initializer", node.lineno)
                    self.symbols[node.name] = 'int'
                    self.memory[node.name] = None
                else:
                    typ = self.infer_expr_type(node.init_expr)
                    self.symbols[node.name] = typ
                    self.memory[node.name] = None
            else:
                self.symbols[node.name] = node.var_type.lower()
                self.memory[node.name] = None
                if node.init_expr is not None:
                    rtype = self.infer_expr_type(node.init_expr)
                    if not type_compatible(self.symbols[node.name], rtype):
                        error("Semantic", f"type mismatch: cannot assign {rtype} to {self.symbols[node.name]}", node.lineno)
        elif isinstance(node, Assign):
            if node.name not in self.symbols:
                error("Semantic", f"undeclared variable '{node.name}'", node.lineno)
                return
            rtype = self.infer_expr_type(node.expr)
            lhs_type = self.symbols[node.name]
            if not type_compatible(lhs_type, rtype):
                error("Semantic", f"type mismatch: cannot assign {rtype} to {lhs_type}", node.lineno)
        elif isinstance(node, Print):
            for p in node.parts:
                self.infer_expr_type(p)
        elif isinstance(node, If):
            ct = self.infer_expr_type(node.cond)
            if ct not in ('int','float','bool'):
                error("Semantic", f"condition should be numeric/bool, got {ct}", node.lineno)
            self.analyze(node.then_block)
            if node.else_block:
                self.analyze(node.else_block)
        elif isinstance(node, While):
            ct = self.infer_expr_type(node.cond)
            if ct not in ('int','float','bool'):
                error("Semantic", f"condition should be numeric/bool, got {ct}", node.lineno)
            self.analyze(node.body)
        elif isinstance(node, Block):
            for s in node.statements:
                self.analyze(s)
        else:
            pass

    def infer_expr_type(self, expr):
        if isinstance(expr, Literal):
            return expr.typ
        if isinstance(expr, Variable):
            if expr.name not in self.symbols:
                error("Semantic", f"undeclared variable '{expr.name}'", None)
                return 'int'
            return self.symbols[expr.name]
        if isinstance(expr, UnaryOp):
            t = self.infer_expr_type(expr.operand)
            if expr.op == '!':
                return 'int'
            return t
        if isinstance(expr, BinaryOp):
            lt = self.infer_expr_type(expr.left)
            rt = self.infer_expr_type(expr.right)
            if expr.op in ('+','-','*','/','%'):
                if lt == 'string' or rt == 'string':
                    if expr.op == '+':
                        return 'string'
                    else:
                        error("Semantic", f"invalid operation {expr.op} on string", None)
                        return 'int'
                if lt == 'float' or rt == 'float':
                    return 'float'
                return 'int'
            if expr.op in ('==','!=','<','>','<=','>='):
                return 'int'
            return 'int'
        return 'int'

def type_compatible(lhs, rhs):
    lhs = lhs.lower()
    rhs = rhs.lower()
    if lhs == rhs:
        return True
    if lhs == 'float' and rhs == 'int':
        return True
    return False

# =====================================================
# IR (TAC) GENERATION
# =====================================================
class TACInstruction:
    def __init__(self, op, dest=None, arg1=None, arg2=None, comment=None):
        self.op = op
        self.dest = dest
        self.arg1 = arg1
        self.arg2 = arg2
        self.comment = comment

    def __repr__(self):
        if self.op == 'label':
            return f"{self.dest}:"
        if self.op == 'jmp':
            return f"jmp {self.dest}"
        if self.op == 'jz':
            return f"jz {self.arg1} {self.dest}"
        if self.op == 'print':
            return f'print {self.arg1}'
        if self.op == 'assign':
            return f"{self.dest} = {self.arg1}"
        if self.op in ('add','sub','mul','div','mod','eq','ne','lt','le','gt','ge'):
            return f"{self.dest} = {self.arg1} {self.op} {self.arg2}"
        return f"{self.op} {self.dest} {self.arg1} {self.arg2}"

class IRGenerator:
    def __init__(self):
        self.tac = []

    def gen(self, node):
        if isinstance(node, Program):
            for s in node.statements:
                self.gen(s)
            return self.tac
        if isinstance(node, VarDecl):
            if node.init_expr:
                t = self.gen_expr(node.init_expr)
                self.tac.append(TACInstruction('assign', dest=node.name, arg1=t))
            else:
                self.tac.append(TACInstruction('assign', dest=node.name, arg1='None'))
            return
        if isinstance(node, Assign):
            t = self.gen_expr(node.expr)
            self.tac.append(TACInstruction('assign', dest=node.name, arg1=t))
            return
        if isinstance(node, Print):
            parts = []
            for p in node.parts:
                t = self.gen_expr(p)
                parts.append(t)
            for p in parts:
                self.tac.append(TACInstruction('print', arg1=p))
            return
        if isinstance(node, If):
            tcond = self.gen_expr(node.cond)
            l_else = new_label()
            l_end = new_label()
            self.tac.append(TACInstruction('jz', dest=l_else, arg1=tcond))
            self.gen(node.then_block)
            self.tac.append(TACInstruction('jmp', dest=l_end))
            self.tac.append(TACInstruction('label', dest=l_else))
            if node.else_block:
                self.gen(node.else_block)
            self.tac.append(TACInstruction('label', dest=l_end))
            return
        if isinstance(node, While):
            l_start = new_label()
            l_end = new_label()
            self.tac.append(TACInstruction('label', dest=l_start))
            tcond = self.gen_expr(node.cond)
            self.tac.append(TACInstruction('jz', dest=l_end, arg1=tcond))
            self.gen(node.body)
            self.tac.append(TACInstruction('jmp', dest=l_start))
            self.tac.append(TACInstruction('label', dest=l_end))
            return
        if isinstance(node, Block):
            for s in node.statements:
                self.gen(s)
            return

    def gen_expr(self, expr):
        if isinstance(expr, Literal):
            if expr.typ == 'string':
                return f'"{expr.value}"'
            if expr.typ == 'char':
                return f"'{expr.value}'"
            if expr.typ == 'float':
                return str(float(expr.value))
            if expr.typ == 'int':
                return str(int(expr.value))
            return str(expr.value)
        if isinstance(expr, Variable):
            return expr.name
        if isinstance(expr, UnaryOp):
            t = self.gen_expr(expr.operand)
            dest = new_temp()
            if expr.op == '-':
                self.tac.append(TACInstruction('mul', dest=dest, arg1='-1', arg2=t))
            elif expr.op == '+':
                self.tac.append(TACInstruction('assign', dest=dest, arg1=t))
            elif expr.op == '!':
                tmp = new_temp()
                self.tac.append(TACInstruction('eq', dest=tmp, arg1=t, arg2='0'))
                return tmp
            return dest
        if isinstance(expr, BinaryOp):
            a = self.gen_expr(expr.left)
            b = self.gen_expr(expr.right)
            dest = new_temp()
            opmap = {'+':'add','-':'sub','*':'mul','/':'div','%':'mod',
                     '==':'eq','!=':'ne','<':'lt','<=':'le','>':'gt','>=':'ge'}
            op = opmap.get(expr.op, None)
            if op is None:
                error("IRGen", f"unknown binary op {expr.op}")
                self.tac.append(TACInstruction('assign', dest=dest, arg1='0'))
            else:
                self.tac.append(TACInstruction(op, dest=dest, arg1=a, arg2=b))
            return dest
        tmp = new_temp()
        self.tac.append(TACInstruction('assign', dest=tmp, arg1='0'))
        return tmp

# =====================================================
# OPTIMIZER: Constant Folding + Dead Code Elimination (simple)
# =====================================================
def is_constant(token):
    if token is None:
        return True
    if re.fullmatch(r'-?\d+', str(token)):
        return True
    if re.fullmatch(r'-?\d+\.\d+', str(token)):
        return True
    if (isinstance(token, str) and token.startswith('"') and token.endswith('"')):
        return True
    if (isinstance(token, str) and token.startswith("'") and token.endswith("'")):
        return True
    return False

def fold_constants(tac):
    newtac = []
    for instr in tac:
        if instr.op in ('add','sub','mul','div','mod','eq','ne','lt','le','gt','ge'):
            a = instr.arg1
            b = instr.arg2
            if is_constant(a) and is_constant(b):
                try:
                    def parse_const(x):
                        if isinstance(x, str) and x.startswith('"') and x.endswith('"'):
                            return x[1:-1]
                        if isinstance(x, str) and x.startswith("'") and x.endswith("'"):
                            return x[1:-1]
                        if '.' in str(x):
                            return float(x)
                        return int(x)
                    va = parse_const(a)
                    vb = parse_const(b)
                    mapping = {
                        'add': lambda x,y: x+y,
                        'sub': lambda x,y: x-y,
                        'mul': lambda x,y: x*y,
                        'div': lambda x,y: x/y if y != 0 else 0,
                        'mod': lambda x,y: x%y if y != 0 else 0,
                        'eq': lambda x,y: 1 if x==y else 0,
                        'ne': lambda x,y: 1 if x!=y else 0,
                        'lt': lambda x,y: 1 if x<y else 0,
                        'le': lambda x,y: 1 if x<=y else 0,
                        'gt': lambda x,y: 1 if x>y else 0,
                        'ge': lambda x,y: 1 if x>=y else 0,
                    }
                    val = mapping[instr.op](va,vb)
                    if isinstance(val, str):
                        lit = f'"{val}"'
                    else:
                        if isinstance(val, float) and val.is_integer():
                            lit = str(int(val))
                        else:
                            lit = str(val)
                    newtac.append(TACInstruction('assign', dest=instr.dest, arg1=lit))
                    continue
                except Exception:
                    pass
        newtac.append(instr)
    return newtac

def dead_code_elimination(tac):
    uses = set()
    for instr in tac:
        if instr.op in ('assign','add','sub','mul','div','mod','eq','ne','lt','le','gt','ge'):
            if instr.arg1 is not None and not is_constant(instr.arg1):
                uses.add(instr.arg1)
            if instr.arg2 is not None and not is_constant(instr.arg2):
                uses.add(instr.arg2)
        elif instr.op == 'jz':
            if instr.arg1 is not None and not is_constant(instr.arg1):
                uses.add(instr.arg1)
        elif instr.op == 'print':
            if instr.arg1 is not None and not is_constant(instr.arg1):
                uses.add(instr.arg1)
    result = []
    for instr in reversed(tac):
        keep = True
        if instr.op == 'assign':
            dest = instr.dest
            if re.fullmatch(r't\d+', str(dest)) and dest not in uses:
                keep = False
            else:
                if instr.arg1 and not is_constant(instr.arg1):
                    uses.add(instr.arg1)
        elif instr.op in ('add','sub','mul','div','mod','eq','ne','lt','le','gt','ge'):
            dest = instr.dest
            if re.fullmatch(r't\d+', str(dest)) and dest not in uses:
                keep = False
            else:
                if instr.arg1 and not is_constant(instr.arg1):
                    uses.add(instr.arg1)
                if instr.arg2 and not is_constant(instr.arg2):
                    uses.add(instr.arg2)
        elif instr.op == 'print':
            if instr.arg1 and not is_constant(instr.arg1):
                uses.add(instr.arg1)
        elif instr.op == 'jz':
            if instr.arg1 and not is_constant(instr.arg1):
                uses.add(instr.arg1)
        if keep:
            result.append(instr)
    result.reverse()
    return result

def optimize_tac(tac):
    old = deepcopy(tac)
    after_fold = fold_constants(old)
    after_dce = dead_code_elimination(after_fold)
    return after_dce

# =====================================================
# TAC Interpreter / Simple Assembly Emission
# =====================================================
def tac_to_assembly(tac):
    asm = []
    for instr in tac:
        if instr.op == 'label':
            asm.append(f"{instr.dest}:")
        elif instr.op == 'jmp':
            asm.append(f"JMP {instr.dest}")
        elif instr.op == 'jz':
            asm.append(f"JZ {instr.arg1} -> {instr.dest}")
        elif instr.op == 'print':
            asm.append(f"PRINT {instr.arg1}")
        elif instr.op == 'assign':
            asm.append(f"MOV {instr.dest}, {instr.arg1}")
        elif instr.op in ('add','sub','mul','div','mod','eq','ne','lt','le','gt','ge'):
            asm.append(f"{instr.op.upper()} {instr.dest}, {instr.arg1}, {instr.arg2}")
        else:
            asm.append(f"{instr.op} {instr.dest} {instr.arg1} {instr.arg2}")
    return asm

def execute_tac(tac, symbol_table):
    mem = dict(symbol_table)
    outputs = []
    labels = {}
    for i,instr in enumerate(tac):
        if instr.op == 'label':
            labels[instr.dest] = i
    pc = 0
    def get_val(x):
        if x is None:
            return 0
        x = str(x)
        if re.fullmatch(r'-?\d+', x):
            return int(x)
        if re.fullmatch(r'-?\d+\.\d+', x):
            return float(x)
        if x.startswith('"') and x.endswith('"'):
            return x[1:-1]
        if x.startswith("'") and x.endswith("'"):
            return x[1:-1]
        return mem.get(x, 0)
    while pc < len(tac):
        instr = tac[pc]
        if instr.op == 'label':
            pc += 1; continue
        if instr.op == 'assign':
            val = get_val(instr.arg1)
            mem[instr.dest] = val
            pc += 1; continue
        if instr.op == 'print':
            val = get_val(instr.arg1)
            outputs.append(str(val))
            pc += 1; continue
        if instr.op in ('add','sub','mul','div','mod','eq','ne','lt','le','gt','ge'):
            a = get_val(instr.arg1)
            b = get_val(instr.arg2)
            res = 0
            try:
                if instr.op == 'add':
                    res = a + b
                elif instr.op == 'sub':
                    res = a - b
                elif instr.op == 'mul':
                    res = a * b
                elif instr.op == 'div':
                    res = a / b if b != 0 else 0
                elif instr.op == 'mod':
                    res = a % b if b != 0 else 0
                elif instr.op == 'eq':
                    res = 1 if a == b else 0
                elif instr.op == 'ne':
                    res = 1 if a != b else 0
                elif instr.op == 'lt':
                    res = 1 if a < b else 0
                elif instr.op == 'le':
                    res = 1 if a <= b else 0
                elif instr.op == 'gt':
                    res = 1 if a > b else 0
                elif instr.op == 'ge':
                    res = 1 if a >= b else 0
            except Exception:
                res = 0
            mem[instr.dest] = res
            pc += 1; continue
        if instr.op == 'jz':
            cond = get_val(instr.arg1)
            if cond == 0 or cond == False:
                dest_index = labels.get(instr.dest, None)
                if dest_index is None:
                    pc += 1
                else:
                    pc = dest_index + 1
            else:
                pc += 1
            continue
        if instr.op == 'jmp':
            dest_index = labels.get(instr.dest, None)
            if dest_index is None:
                pc += 1
            else:
                pc = dest_index + 1
            continue
        pc += 1
    return outputs

# =====================================================
# COMPILER DRIVER
# =====================================================
def compile_source(code, verbose=True):
    global temp_count, label_count, errors
    temp_count = 0; label_count = 0; errors = []

    # Initialize result with empty values
    result = {
        'tokens': [],
        'ast': None,
        'tac': [],
        'optimized_tac': [],
        'asm': [],
        'output': [],
        'errors': [],
        'symbol_table': {},
        'memory': {}
    }

    lex = Lexer(code)
    toks = lex.peek_all()
    result['tokens'] = toks
    
    # Check for lexical errors
    if errors:
        result['errors'] = errors.copy()
        return result

    toks.append(Token('EOF','',None))
    
    # Parse the code
    parser = Parser(toks)
    try:
        ast = parser.parse()
        result['ast'] = ast
    except SyntaxError:
        if errors:
            result['errors'] = errors.copy()
        else:
            result['errors'] = ["Syntax error during parse"]
        return result

    # Analyze semantics
    sem = SemanticAnalyzer()
    sem.analyze(ast)
    if errors:
        result['errors'] = errors.copy()
        result['symbol_table'] = sem.symbols
        result['memory'] = sem.memory
        return result

    # Generate IR
    irgen = IRGenerator()
    tac = irgen.gen(ast) or irgen.tac
    result['tac'] = tac

    # Optimize IR
    optimized = optimize_tac(tac)
    result['optimized_tac'] = optimized

    # Generate assembly
    asm = tac_to_assembly(optimized)
    result['asm'] = asm

    # Execute
    outputs = execute_tac(optimized, sem.memory)
    result['output'] = outputs
    result['symbol_table'] = sem.symbols
    result['memory'] = sem.memory

    # Return the result
    return result

# =====================================================
# TEST PROGRAM
# =====================================================
TEST_PROGRAM = r'''
// sample program
int main() {
    int x = 10;
    int y = "Hello";
    int z = (x + y) * 2;
    cout << z;
}
'''

if __name__ == '__main__':
    compile_source(TEST_PROGRAM)