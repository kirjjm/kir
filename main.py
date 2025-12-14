import re
import argparse
import sys
from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError


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


GRAMMAR = r"""
start: const_def* dict

const_def: "var" IDENT const_value

?const_value: number
            | string
            | const_expr
            | IDENT      -> const_ref_in_def

?value: number
      | string
      | array
      | dict
      | const_expr
      | IDENT          -> const_ref_in_value

number: NUMBER
string: STRING
array: "(" "list" value* ")"
dict: "begin" stmt* "end"
stmt: IDENT ":=" value ";"
const_expr: CONST_EXPR

IDENT: /[A-Za-z]+/
STRING: /'[^']*'/
CONST_EXPR: /\^\[[^\]]*\]/
NUMBER: /\d+\.\d*|\d+/

%import common.WS
%ignore WS
"""


@v_args(inline=True)
class ConfigTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.consts = {}

    def number(self, token):
        s = str(token)
        return float(s) if "." in s else int(s)

    def string(self, token):
        s = str(token)
        return s[1:-1]

    def const_expr(self, token):
        return eval_const_expr(str(token), self.consts)

    def const_ref_in_def(self, name_token):
        name = str(name_token)
        if name not in self.consts:
            raise ValueError(f"Неизвестная константа в определении: {name}")
        return self.consts[name]

    def const_ref_in_value(self, name_token):
        name = str(name_token)
        if name not in self.consts:
            raise ValueError(f"Неизвестная константа: {name}")
        return self.consts[name]

    def const_def(self, name_token, value):
        name = str(name_token)
        self.consts[name] = value
        return None

    def stmt(self, name_token, value):
        return str(name_token), value

    def dict(self, *stmts):
        d = {}
        for k, v in stmts:
            d[k] = v
        return d

    def array(self, *values):
        return list(values)

    def start(self, *items):
        for item in reversed(items):
            if item is not None:
                return item
        return None


parser = Lark(GRAMMAR, parser="lalr", start="start")


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
    tree = parser.parse(text)
    transformer = ConfigTransformer()
    return transformer.transform(tree)


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
    cli = argparse.ArgumentParser(
        description="Перевод учебного конфигурационного языка (вариант 28, Lark) в YAML"
    )
    cli.add_argument("-i", "--input", help="входной файл с конфигурацией")
    cli.add_argument("-o", "--output", help="выходной YAML-файл")
    cli.add_argument(
        "--run-tests",
        action="store_true",
        help="запустить встроенные тесты и выйти",
    )

    args = cli.parse_args()

    if args.run_tests:
        run_tests()
        return

    if not args.input or not args.output:
        cli.error("нужно указать и -i, и -o (или использовать --run-tests)")

    try:
        translate_file(args.input, args.output)
        print(f"Готово: результат записан в {args.output}")
    except (LarkError, SyntaxError, ValueError) as e:
        print("Синтаксическая ошибка:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
