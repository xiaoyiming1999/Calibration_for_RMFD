import argparse
import ast
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


def parse_cli(*argv):
    # Execute the production parser without importing the trainers or PyTorch.
    source = Path(__file__).resolve().parents[1] / 'main.py'
    module = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    module.body = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in {'parse_args', 'str2bool'}
    ]
    namespace = {'argparse': argparse}
    exec(compile(module, str(source), 'exec'), namespace)
    with patch.object(sys, 'argv', ['main.py', *argv]):
        return namespace['parse_args']()


class UseTemperatureTest(unittest.TestCase):
    def test_readme_commands(self):
        for dataset, value, method, expected in (
            ('PU', 'True', 'focal_loss', True),
            ('SEU', 'False', 'ensemble', False),
        ):
            with self.subTest(dataset=dataset):
                args = parse_cli('--data_name', dataset, '--use_temperature', value, '--method', method)
                self.assertEqual((args.data_name, args.method), (dataset, method))
                self.assertIs(args.use_temperature, expected)

    def test_boolean_values(self):
        for expected, values in (
            (True, ('True', 'true', '1', 'yes', 't', 'y', ' YeS ')),
            (False, ('False', 'false', '0', 'no', 'f', 'n', ' FaLsE ')),
        ):
            for value in values:
                with self.subTest(value=value):
                    self.assertIs(parse_cli('--use_temperature', value).use_temperature, expected)

    def test_default_is_true(self):
        self.assertIs(parse_cli().use_temperature, True)

    def test_invalid_values_are_rejected(self):
        for value in ('maybe', ''):
            with self.subTest(value=value):
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    parse_cli('--use_temperature', value)
                self.assertEqual(error.exception.code, 2)
                self.assertIn('--use_temperature', stderr.getvalue())
                self.assertIn('expected a boolean value', stderr.getvalue())
