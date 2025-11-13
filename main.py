import sys
from dataclasses import dataclass
from typing import List, Any, Optional
import os

# ==========================
#   ЛЕКСЕР
# ==========================

class TokenType:
    # однобуквенные
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    SEMI = "SEMI"
    COMMA = "COMMA"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    PERCENT = "PERCENT"
    ASSIGN = "ASSIGN"

    # сравнения
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    LTE = "LTE"
    GT = "GT"
    GTE = "GTE"

    # ключевые слова / типы
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOL = "BOOL"

    KW_ANTON = "ANTON"
    KW_LET = "LET"
    KW_IF = "IF"
    KW_ELSE = "ELSE"
    KW_WHILE = "WHILE"
    KW_INPUT = "INPUT"
    KW_PRINT = "PRINT"
    KW_AND = "AND"
    KW_OR = "OR"
    KW_NOT = "NOT"

    EOF = "EOF"


KEYWORDS = {
    "Anton": TokenType.KW_ANTON,
    "let": TokenType.KW_LET,
    "if": TokenType.KW_IF,
    "else": TokenType.KW_ELSE,
    "while": TokenType.KW_WHILE,
    "input": TokenType.KW_INPUT,
    "print": TokenType.KW_PRINT,
    "and": TokenType.KW_AND,
    "or": TokenType.KW_OR,
    "not": TokenType.KW_NOT,
    "true": TokenType.BOOL,
    "false": TokenType.BOOL,
}


@dataclass
class Token:
    type: str
    value: Any
    pos: int


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def peek(self) -> str:
        if self.pos >= len(self.text):
            return "\0"
        return self.text[self.pos]

    def advance(self):
        self.pos += 1

    def match(self, expected: str) -> bool:
        if self.peek() == expected:
            self.advance()
            return True
        return False

    def skip_whitespace_and_comments(self):
        while True:
            c = self.peek()
            if c in " \t\r\n":
                self.advance()
            elif c == "/" and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == "/":
                # комментарий до конца строки
                while self.peek() not in ("\n", "\0"):
                    self.advance()
            else:
                break

    def identifier_or_keyword(self) -> Token:
        start = self.pos
        self.advance()  # первая буква / _
        while self.peek().isalnum() or self.peek() == "_":
            self.advance()
        text = self.text[start:self.pos]
        ttype = KEYWORDS.get(text, TokenType.IDENT)
        if ttype == TokenType.BOOL:
            value = True if text == "true" else False
            return Token(TokenType.BOOL, value, start)
        return Token(ttype, text, start)

    def number(self) -> Token:
        start = self.pos
        while self.peek().isdigit():
            self.advance()
        text = self.text[start:self.pos]
        return Token(TokenType.NUMBER, int(text), start)

    def string(self) -> Token:
        start = self.pos
        self.advance()  # opening "
        chars = []
        while True:
            c = self.peek()
            if c == "\0":
                raise SyntaxError(f"Unterminated string at {start}")
            if c == '"':
                self.advance()
                break
            if c == "\\":
                # простые экранирования: \" \\ \n
                self.advance()
                esc = self.peek()
                if esc == "n":
                    chars.append("\n")
                elif esc == '"':
                    chars.append('"')
                elif esc == "\\":
                    chars.append("\\")
                else:
                    chars.append(esc)
                self.advance()
            else:
                chars.append(c)
                self.advance()
        return Token(TokenType.STRING, "".join(chars), start)

    def next_token(self) -> Token:
        self.skip_whitespace_and_comments()
        start = self.pos
        c = self.peek()

        if c == "\0":
            return Token(TokenType.EOF, None, self.pos)

        # идентификатор или ключевое слово
        if c.isalpha() or c == "_":
            return self.identifier_or_keyword()

        # число
        if c.isdigit():
            return self.number()

        # строка
        if c == '"':
            return self.string()

        # операторы и разделители
        if c == "{":
            self.advance()
            return Token(TokenType.LBRACE, "{", start)
        if c == "}":
            self.advance()
            return Token(TokenType.RBRACE, "}", start)
        if c == "(":
            self.advance()
            return Token(TokenType.LPAREN, "(", start)
        if c == ")":
            self.advance()
            return Token(TokenType.RPAREN, ")", start)
        if c == ";":
            self.advance()
            return Token(TokenType.SEMI, ";", start)
        if c == ",":
            self.advance()
            return Token(TokenType.COMMA, ",", start)
        if c == "+":
            self.advance()
            return Token(TokenType.PLUS, "+", start)
        if c == "-":
            self.advance()
            return Token(TokenType.MINUS, "-", start)
        if c == "*":
            self.advance()
            return Token(TokenType.STAR, "*", start)
        if c == "/":
            self.advance()
            return Token(TokenType.SLASH, "/", start)
        if c == "%":
            self.advance()
            return Token(TokenType.PERCENT, "%", start)
        if c == "=":
            self.advance()
            if self.match("="):
                return Token(TokenType.EQ, "==", start)
            return Token(TokenType.ASSIGN, "=", start)
        if c == "!":
            self.advance()
            if self.match("="):
                return Token(TokenType.NEQ, "!=", start)
            raise SyntaxError(f"Unexpected character ! at {start}")
        if c == "<":
            self.advance()
            if self.match("="):
                return Token(TokenType.LTE, "<=", start)
            return Token(TokenType.LT, "<", start)
        if c == ">":
            self.advance()
            if self.match("="):
                return Token(TokenType.GTE, ">=", start)
            return Token(TokenType.GT, ">", start)

        raise SyntaxError(f"Unexpected character {c!r} at {start}")


# ==========================
#   AST
# ==========================

# Выражения
class Expr: ...


@dataclass
class NumberExpr(Expr):
    value: int


@dataclass
class StringExpr(Expr):
    value: str


@dataclass
class BoolExpr(Expr):
    value: bool


@dataclass
class VarExpr(Expr):
    name: str


@dataclass
class BinaryExpr(Expr):
    left: Expr
    op: str  # TokenType
    right: Expr


@dataclass
class UnaryExpr(Expr):
    op: str
    expr: Expr


# Стейтменты
class Stmt: ...


@dataclass
class VarDeclStmt(Stmt):
    name: str
    init: Optional[Expr]


@dataclass
class AssignStmt(Stmt):
    name: str
    expr: Expr


@dataclass
class InputStmt(Stmt):
    name: str


@dataclass
class PrintStmt(Stmt):
    args: List[Expr]


@dataclass
class IfStmt(Stmt):
    cond: Expr  # LogicExpr, но для простоты Expr
    then_branch: List[Stmt]
    else_branch: Optional[List[Stmt]]


@dataclass
class WhileStmt(Stmt):
    cond: Expr
    body: List[Stmt]


# ==========================
#   ПАРСЕР
# ==========================

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current = self.lexer.next_token()

    def eat(self, token_type: str) -> Token:
        if self.current.type == token_type:
            tok = self.current
            self.current = self.lexer.next_token()
            return tok
        raise SyntaxError(f"Expected {token_type}, got {self.current.type} at pos {self.current.pos}")

    def match(self, *types) -> bool:
        if self.current.type in types:
            self.current = self.lexer.next_token()
            return True
        return False

    # ----- Верхний уровень -----

    def parse_program(self) -> List[Stmt]:
        self.eat(TokenType.KW_ANTON)
        stmts = self.parse_block()
        if self.current.type != TokenType.EOF:
            raise SyntaxError("Extra input after program end")
        return stmts

    def parse_block(self) -> List[Stmt]:
        self.eat(TokenType.LBRACE)
        stmts: List[Stmt] = []
        while self.current.type != TokenType.RBRACE:
            stmts.append(self.parse_statement())
        self.eat(TokenType.RBRACE)
        return stmts

    # ----- Стейтменты -----

    def parse_statement(self) -> Stmt:
        if self.current.type == TokenType.KW_LET:
            return self.parse_var_decl()
        if self.current.type == TokenType.KW_INPUT:
            return self.parse_input()
        if self.current.type == TokenType.KW_PRINT:
            return self.parse_print()
        if self.current.type == TokenType.KW_IF:
            return self.parse_if()
        if self.current.type == TokenType.KW_WHILE:
            return self.parse_while()
        if self.current.type == TokenType.IDENT:
            # присваивание
            name = self.current.value
            self.eat(TokenType.IDENT)
            self.eat(TokenType.ASSIGN)
            expr = self.parse_expression()
            self.eat(TokenType.SEMI)
            return AssignStmt(name, expr)
        raise SyntaxError(f"Unexpected token {self.current.type} in statement at {self.current.pos}")

    def parse_var_decl(self) -> Stmt:
        self.eat(TokenType.KW_LET)
        name = self.eat(TokenType.IDENT).value
        init = None
        if self.match(TokenType.ASSIGN):
            init = self.parse_expression()
        self.eat(TokenType.SEMI)
        return VarDeclStmt(name, init)


    def parse_input(self) -> Stmt:
        self.eat(TokenType.KW_INPUT)
        self.eat(TokenType.LPAREN)
        name = self.eat(TokenType.IDENT).value
        self.eat(TokenType.RPAREN)
        self.eat(TokenType.SEMI)
        return InputStmt(name)

    def parse_print(self) -> Stmt:
        self.eat(TokenType.KW_PRINT)
        self.eat(TokenType.LPAREN)
        args: List[Expr] = []
        if self.current.type != TokenType.RPAREN:
            args.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expression())
        self.eat(TokenType.RPAREN)
        self.eat(TokenType.SEMI)
        return PrintStmt(args)

    def parse_if(self) -> Stmt:
        self.eat(TokenType.KW_IF)
        self.eat(TokenType.LPAREN)
        cond = self.parse_logic_expr()
        self.eat(TokenType.RPAREN)
        then_branch = self.parse_block()
        else_branch = None
        if self.match(TokenType.KW_ELSE):
            else_branch = self.parse_block()
        return IfStmt(cond, then_branch, else_branch)

    def parse_while(self) -> Stmt:
        self.eat(TokenType.KW_WHILE)
        self.eat(TokenType.LPAREN)
        cond = self.parse_logic_expr()
        self.eat(TokenType.RPAREN)
        body = self.parse_block()
        return WhileStmt(cond, body)

    # ----- Логические выражения -----

    def parse_logic_expr(self) -> Expr:
        # OrChain = AndChain { 'or' AndChain }
        expr = self.parse_and_chain()
        while self.current.type == TokenType.KW_OR:
            op = self.current.type
            self.eat(TokenType.KW_OR)
            right = self.parse_and_chain()
            expr = BinaryExpr(expr, op, right)
        return expr

    def parse_and_chain(self) -> Expr:
        expr = self.parse_not_chain()
        while self.current.type == TokenType.KW_AND:
            op = self.current.type
            self.eat(TokenType.KW_AND)
            right = self.parse_not_chain()
            expr = BinaryExpr(expr, op, right)
        return expr

    def parse_not_chain(self) -> Expr:
        if self.current.type == TokenType.KW_NOT:
            op = self.current.type
            self.eat(TokenType.KW_NOT)
            inner = self.parse_not_chain()  # рекурсивно, как обсуждали
            return UnaryExpr(op, inner)
        return self.parse_compare_expr()

    def parse_compare_expr(self) -> Expr:
        # Expression [ CompareOp Expression ]
        expr = self.parse_expression()
        if self.current.type in (TokenType.EQ, TokenType.NEQ,
                                 TokenType.LT, TokenType.LTE,
                                 TokenType.GT, TokenType.GTE):
            op = self.current.type
            self.eat(self.current.type)
            right = self.parse_expression()
            expr = BinaryExpr(expr, op, right)
        return expr

    # ----- Арифметика / логика в выражениях -----

    def parse_expression(self) -> Expr:
        # Expression = Term { ("+" | "-") Term }
        expr = self.parse_term()
        while self.current.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current.type
            self.eat(self.current.type)
            right = self.parse_term()
            expr = BinaryExpr(expr, op, right)
        return expr

    def parse_term(self) -> Expr:
        # Term = Factor | MulDivTerm
        # здесь реализуем как обычный приоритет: *, /, %
        expr = self.parse_factor()
        # если дальше * / % — это MulDivTerm
        while self.current.type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.current.type
            self.eat(self.current.type)
            right = self.parse_factor_numeric()
            expr = BinaryExpr(expr, op, right)
        return expr

    def parse_factor(self) -> Expr:
        tok = self.current
        if tok.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return NumberExpr(tok.value)
        if tok.type == TokenType.STRING:
            self.eat(TokenType.STRING)
            return StringExpr(tok.value)
        if tok.type == TokenType.BOOL:
            self.eat(TokenType.BOOL)
            return BoolExpr(tok.value)
        if tok.type == TokenType.IDENT:
            self.eat(TokenType.IDENT)
            return VarExpr(tok.value)
        if tok.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            # БЫЛО: expr = self.parse_expression()
            # ДОЛЖНО БЫТЬ:
            expr = self.parse_logic_expr()
            self.eat(TokenType.RPAREN)
            return expr
        raise SyntaxError(f"Unexpected token {tok.type} in expression at {tok.pos}")

    def parse_factor_numeric(self) -> Expr:
        """NumericFactor: Number | Identifier | '(' Expression ')'"""
        tok = self.current
        if tok.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return NumberExpr(tok.value)
        if tok.type == TokenType.IDENT:
            self.eat(TokenType.IDENT)
            return VarExpr(tok.value)
        if tok.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            expr = self.parse_expression()
            self.eat(TokenType.RPAREN)
            return expr
        raise SyntaxError(f"Numeric factor expected at {tok.pos}")


# ==========================
#   ИНТЕРПРЕТАТОР
# ==========================

class Environment:
    def __init__(self):
        self.vars = {}

    def declare(self, name: str, value: Any = None):
        if name in self.vars:
            raise RuntimeError(f"Variable '{name}' already declared")
        self.vars[name] = value

    def set(self, name: str, value: Any):
        if name not in self.vars:
            raise RuntimeError(f"Undeclared variable '{name}'")
        self.vars[name] = value

    def get(self, name: str) -> Any:
        if name not in self.vars:
            raise RuntimeError(f"Undeclared variable '{name}'")
        return self.vars[name]


class Interpreter:
    def __init__(self):
        self.env = Environment()

    # ---- выражения ----

    def eval_expr(self, expr: Expr) -> Any:
        if isinstance(expr, NumberExpr):
            return expr.value
        if isinstance(expr, StringExpr):
            return expr.value
        if isinstance(expr, BoolExpr):
            return expr.value
        if isinstance(expr, VarExpr):
            return self.env.get(expr.name)
        if isinstance(expr, UnaryExpr):
            val = self.eval_expr(expr.expr)
            if expr.op == TokenType.KW_NOT:
                return not bool(val)
            else:
                raise RuntimeError("Unknown unary operator")
        if isinstance(expr, BinaryExpr):
            left = self.eval_expr(expr.left)
            right = self.eval_expr(expr.right)
            op = expr.op

            # логика
            if op == TokenType.KW_AND:
                return bool(left) and bool(right)
            if op == TokenType.KW_OR:
                return bool(left) or bool(right)

            # сравнения
            if op == TokenType.EQ:
                return left == right
            if op == TokenType.NEQ:
                return left != right
            if op == TokenType.LT:
                return left < right
            if op == TokenType.LTE:
                return left <= right
            if op == TokenType.GT:
                return left > right
            if op == TokenType.GTE:
                return left >= right

            # арифметика / конкатенация
            if op == TokenType.PLUS:
                # разрешаем число+число или строка+строка
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left + right
                raise RuntimeError("Operator + supports int+int or string+string")
            if op == TokenType.MINUS:
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left - right
                raise RuntimeError("Operator - only for numbers")
            if op == TokenType.STAR:
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left * right
                # запрещаем строку * число (по заданию)
                raise RuntimeError("Operator * only for numbers (no string repetition)")
            if op == TokenType.SLASH:
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    if right == 0:
                        raise RuntimeError("Division by zero")
                    return left / right
                raise RuntimeError("Operator / only for numbers")
            if op == TokenType.PERCENT:
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    if right == 0:
                        raise RuntimeError("Modulo by zero")
                    return left % right
                raise RuntimeError("Operator % only for numbers")

            raise RuntimeError(f"Unknown binary operator {op}")

        raise RuntimeError("Unknown expression type")

    # ---- стейтменты ----

    def exec_stmt(self, stmt: Stmt):
        if isinstance(stmt, VarDeclStmt):
            value = self.eval_expr(stmt.init) if stmt.init is not None else None
            self.env.declare(stmt.name, value)
        elif isinstance(stmt, AssignStmt):
            value = self.eval_expr(stmt.expr)
            self.env.set(stmt.name, value)
        elif isinstance(stmt, InputStmt):
            user = input(f"{stmt.name}> ")
            # можно попытаться привести к int
            if user.isdigit():
                val: Any = int(user)
            else:
                val = user
            if stmt.name not in self.env.vars:
                self.env.declare(stmt.name, val)
            else:
                self.env.set(stmt.name, val)
        elif isinstance(stmt, PrintStmt):
            values = [self.eval_expr(e) for e in stmt.args]
            print(*values)
        elif isinstance(stmt, IfStmt):
            cond = self.eval_expr(stmt.cond)
            if cond:
                for s in stmt.then_branch:
                    self.exec_stmt(s)
            elif stmt.else_branch is not None:
                for s in stmt.else_branch:
                    self.exec_stmt(s)
        elif isinstance(stmt, WhileStmt):
            while self.eval_expr(stmt.cond):
                for s in stmt.body:
                    self.exec_stmt(s)
        else:
            raise RuntimeError("Unknown statement type")

    def exec_program(self, stmts: List[Stmt]):
        for s in stmts:
            self.exec_stmt(s)


# ==========================
#   ЗАПУСК
# ==========================

def run_anton(source: str):
    lexer = Lexer(source)
    parser = Parser(lexer)
    program = parser.parse_program()
    interp = Interpreter()
    interp.exec_program(program)

def run_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        print(f"--- Запуск файла: {path} ---")
        run_anton(source)
        print(f"--- Завершено: {path} ---\n")
    except FileNotFoundError:
        print(f"Ошибка: файл '{path}' не найден.")
    except SyntaxError as e:
        print(f"[СИНТАКСИЧЕСКАЯ ОШИБКА] {e}")
    except RuntimeError as e:
        print(f"[ОШИБКА ВЫПОЛНЕНИЯ] {e}")
    except Exception as e:
        print(f"[НЕИЗВЕСТНАЯ ОШИБКА] {e}")

def run_with_menu(directory="."):
    files = [f for f in os.listdir(directory) if f.endswith(".anton")]

    if not files:
        print("В этой папке нет .anton файлов.")
        return

    print("Доступные файлы для запуска:\n")
    for i, name in enumerate(files, 1):
        print(f"{i}) {name}")

    print("\n0) Выход")

    while True:
        choice = input("\nВведите номер файла: ")

        if not choice.isdigit():
            print("Введите номер.")
            continue

        choice = int(choice)

        if choice == 0:
            print("Выход.")
            break

        if 1 <= choice <= len(files):
            filename = os.path.join(directory, files[choice - 1])
            run_file(filename)
        else:
            print("Нет такого номера.")

if __name__ == "__main__":
    run_with_menu(".")
