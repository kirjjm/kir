import re
import argparse
import sys
from dataclasses import dataclass


@dataclass
class Token:
    type: str
    value: str


def remove_comments(text: str) -> str:
    text = re.sub(r"\|\|.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\{.*?\}", "", text, flags=re.DOTALL)
    return text


_number_re = re.compile(r"^\d+\.\d*|\d+$")


def is_number(s: str) -> bool:
    return bool(_number_re.fullmatch(s))


def parse_number(s: str):
    return float(s) if "." in s else int(s)


def eval_const_expr(expr_text: str, consts: dict):
    if not (expr_text.startswith("^[") and expr_text.endswith("]")):
        raise SyntaxError(f"Некорректное константное выражение: {expr_text}")

    inner = expr_text[2:-1].strip()
    if not inner:
        raise SyntaxError("Пустое константное выражение")

    stack = []
    for tok in inner.split():
        if is_number(tok):
            stack.append(parse_number(tok))
        elif tok in consts:
            stack.append(consts[tok])
        elif tok == "+":
            if len(stack) < 2:
                raise SyntaxError("Недостаточно операндов для +")
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif tok == "pow()":
            if len(stack) < 2:
                raise SyntaxError("Недостаточно операндов для pow()")
            b = stack.pop()
            a = stack.pop()
            stack.append(pow(a, b))
        elif tok == "print()":
            if not stack:
                raise SyntaxError("Нет значения для print()")
            value = stack[-1]
            print(value)
        else:
            raise SyntaxError(f"Неизвестный токен в константном выражении: {tok}")

    if len(stack) != 1:
        raise SyntaxError("Лишние значения в стеке константного выражения")
    return stack[0]


def extract_consts(text: str):
    consts = {}
    other_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("var "):
            parts = stripped.split(None, 3)
            if len(parts) < 3:
                raise SyntaxError(f"Неверная строка константы: {line}")
            _, name, value_str = parts[0], parts[1], " ".join(parts[2:])

            if is_number(value_str):
                value = parse_number(value_str)
            elif re.fullmatch(r"'[^']*'", value_str):
                value = value_str[1:-1]
            elif value_str.startswith("^[") and value_str.endswith("]"):
                value = eval_const_expr(value_str, consts)
            elif value_str in consts:
                value = consts[value_str]
            else:
                raise SyntaxError(f"Неподдерживаемое значение константы: {value_str}")

            consts[name] = value
        else:
            other_lines.append(line)

    new_text = "\n".join(other_lines)
    return new_text, consts


TOKEN_RE = re.compile(
    r"(?P<WS>\s+)|(?P<ASSIGN>:=)|(?P<SEMICOLON>;)|(?P<LPAREN>\()|(?P<RPAREN>\))|(?P<BEGIN>\bbegin\b)|(?P<END>\bend\b)|(?P<LIST>\blist\b)|(?P<CONST_EXPR>\^\[.*?\])|(?P<STRING>'[^']*')|(?P<NUMBER>\d+\.\d*|\d+)|(?P<IDENT>[a-zA-Z]+)"
)


def tokenize(text: str):
    tokens = []
    pos = 0
    while pos < len(text):
        match = TOKEN_RE.match(text, pos)
        if not match:
            raise SyntaxError(f"Неожиданный символ {text[pos]!r} на позиции {pos}")
        kind = match.lastgroup
        value = match.group(kind)
        pos = match.end()

        if kind == "WS":
            continue
        tokens.append(Token(kind, value))

    tokens.append(Token("EOF", ""))
    return tokens


class Parser:
    def __init__(self, tokens, consts):
        self.tokens = tokens
        self.pos = 0
        self.consts = consts

    @property
    def current(self):
        return self.tokens[self.pos]

    def eat(self, expected_type: str):
        if self.current.type != expected_type:
            raise SyntaxError(
                f"Ожидался {expected_type}, а встретился {self.current.type} ({self.current.value})"
            )
        self.pos += 1

    def parse(self):
        return self.parse_dict()

    def parse_dict(self):
        result = {}
        self.eat("BEGIN")
        while self.current.type != "END":
            if self.current.type != "IDENT":
                raise SyntaxError("Ожидалось имя поля в словаре")
            name = self.current.value
            self.eat("IDENT")
            self.eat("ASSIGN")
            value = self.parse_value()
            self.eat("SEMICOLON")
            result[name] = value
        self.eat("END")
        return result

    def parse_value(self):
        tok = self.current

        if tok.type == "NUMBER":
            self.eat("NUMBER")
            return parse_number(tok.value)

        if tok.type == "STRING":
            self.eat("STRING")
            return tok.value[1:-1]

        if tok.type == "LPAREN":
            return self.parse_array()

        if tok.type == "BEGIN":
            return self.parse_dict()

        if tok.type == "CONST_EXPR":
            self.eat("CONST_EXPR")
            return eval_const_expr(tok.value, self.consts)

        if tok.type == "IDENT":
            name = tok.value
            self.eat("IDENT")
            if name in self.consts:
                return self.consts[name]
            raise SyntaxError(f"Неизвестная константа или недопустимое значение: {name}")

        raise SyntaxError(f"Неожиданное значение: {tok.type} {tok.value}")

    def parse_array(self):
        items = []
        self.eat("LPAREN")
        self.eat("LIST")

        while self.current.type != "RPAREN":
            items.append(self.parse_value())

        self.eat("RPAREN")
        return items


def format_scalar(value):
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return "'" + escaped + "'"
    else:
        return str(value)


def to_yaml(value, indent: int = 0) -> str:
    space = "  " * indent

    if isinstance(value, dict):
        lines = []
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(to_yaml(val, indent + 1))
            else:
                lines.append(f"{space}{key}: {format_scalar(val)}")
        return "\n".join(lines)

    elif isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.append(to_yaml(item, indent + 1))
            else:
                lines.append(f"{space}- {format_scalar(item)}")
        return "\n".join(lines)

    else:
        return f"{space}{format_scalar(value)}"


def translate_text(text: str):
    text = remove_comments(text)
    text, consts = extract_consts(text)
    tokens = tokenize(text)
    parser = Parser(tokens, consts)
    data = parser.parse()
    return data


def translate_text_to_yaml(text: str) -> str:
    return to_yaml(translate_text(text))


def translate_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    data = translate_text(text)
    yaml_str = to_yaml(data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_str)


def run_tests():
    tests_passed = 0

    def check(name, text, expected_data):
        nonlocal tests_passed
        result = translate_text(text)
        assert result == expected_data, (
            f"{name} не прошёл:\n"
            f"ожидалось: {expected_data}\n"
            f"получили : {result}"
        )
        tests_passed += 1

    cfg1 = """
    || однострочный комментарий
    {
      многострочный
      комментарий
    }
    begin
      port := 8080;
      host := 'localhost';
    end
    """
    check("basic_numbers_strings", cfg1, {"port": 8080, "host": "localhost"})

    cfg2 = """
    begin
      nums := (list 1 2 3 4);
      words := (list 'a' 'b');
    end
    """
    check("arrays", cfg2, {"nums": [1, 2, 3, 4], "words": ["a", "b"]})

    cfg3 = """
    begin
      server := begin
        host := 'localhost';
        ports := (list 80 443);
      end;
      features := (list 'auth' 'metrics');
    end
    """
    check(
        "nested_dicts_arrays",
        cfg3,
        {
            "server": {"host": "localhost", "ports": [80, 443]},
            "features": ["auth", "metrics"],
        },
    )

    cfg4 = """
    var hostname 'localhost'
    var port 8000
    begin
      host := hostname;
      port := port;
    end
    """
    check("vars", cfg4, {"host": "localhost", "port": 8000})

    cfg5 = """
    var base 10
    var inc 5
    var total ^[base inc +]
    begin
      result := total;
      inline := ^[1 2 +];
    end
    """
    check("const_expr_add", cfg5, {"result": 15, "inline": 3})

    cfg6 = """
    var two 2
    var ten 10
    var big ^[two ten pow()]
    begin
      bigvalue := big;
      echo := ^[big print()];
    end
    """
    res6 = translate_text(cfg6)
    assert res6["bigvalue"] == 1024 and res6["echo"] == 1024
    tests_passed += 1

    cfg7 = """
    var baseport 8000
    var step 10
    begin
      services := (list
        begin
          name := 'api';
          port := ^[baseport step +];
        end
        begin
          name := 'admin';
          port := ^[baseport step step + +];
        end
      );
    end
    """
    check(
        "complex_nesting",
        cfg7,
        {
            "services": [
                {"name": "api", "port": 8010},
                {"name": "admin", "port": 8020},
            ]
        },
    )

    print(f"Все тесты пройдены: {tests_passed}")


def main():
    parser = argparse.ArgumentParser(
        description="Перевод учебного конфигурационного языка в YAML"
    )
    parser.add_argument("-i", "--input", help="входной файл с конфигурацией")
    parser.add_argument("-o", "--output", help="выходной YAML-файл")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="запустить встроенные тесты и выйти",
    )

    args = parser.parse_args()

    if args.run_tests:
        run_tests()
        return

    if not args.input or not args.output:
        parser.error("нужно указать и -i, и -o (или использовать --run-tests)")

    try:
        translate_file(args.input, args.output)
        print(f"Готово: результат записан в {args.output}")
    except (SyntaxError, ValueError) as e:
        print("Синтаксическая ошибка:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
