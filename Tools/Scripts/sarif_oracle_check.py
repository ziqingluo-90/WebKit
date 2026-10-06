#!/usr/bin/env python3
"""Report oracle SARIF entries that are not included in a test SARIF.

An oracle entry (escape-inference result for one parameter) is included if the test has
a result with the same verdict ("may escape" / "must not escape") for the same parameter:

  1. same file + line + parameter name, or
  2. same function (with enclosing namespace/class) + parameter position, read from the
     source files. This pairs a declaration with its definition, e.g. an annotated
     prototype in a header with the remark reported on the function body. Best-effort:
     it is a lightweight text scan, not a C++ parser. Disable with --exact-only.

Oracle entries that the test reports with the opposite verdict are listed as contradicted.
Exits 1 if any oracle entry is missing or contradicted.

Usage: sarif_oracle_check.py ORACLE.sarif TEST.sarif [--exact-only] [--quiet]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sarif_escape_counts import MAY, MUST_NOT, results  # noqa: E402

PARAM_RE = re.compile(r"(?:parameter|the implicit) '([^']*)'")
NAME_RE = re.compile(r'(operator\s*(?:\(\)|\[\]|new(?:\s*\[\])?|delete(?:\s*\[\])?|[^\s(]+)'
                     r'|~?[A-Za-z_]\w*(?:\s*::\s*~?[A-Za-z_]\w*)*)\s*$')


def strip_code(text):
    """Blanks comments and string/char literals, keeping offsets and newlines."""
    out, i, n = list(text), 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
        elif c in '"\'':
            j = i + 1
            while j < n and text[j] != c and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
        else:
            i += 1
            continue
        for k in range(i, j):
            if out[k] != '\n':
                out[k] = ' '
        i = j
    return ''.join(out)


class Source:
    """A source file with comments stripped and the namespace/class scope at each line."""

    def __init__(self, path):
        code = strip_code(open(path, errors='replace').read())
        code = re.sub(r'(?m)^[ \t]*#.*$', lambda m: ' ' * len(m.group()), code)
        self.lines = code.split('\n')
        self.scopes = []           # scope names (outermost first) at the start of each line
        stack, header = [], ''
        for line in self.lines:
            self.scopes.append([s for s in stack if s])
            for c in line:
                if c == '{':
                    m = re.search(r'\b(?:namespace\s+([\w:]+)|(?:class|struct|union)\s+(?:\[\[.*?\]\]\s*)?(\w+)[^;()]*)$', header)
                    stack.append((m.group(1) or m.group(2)) if m else None)
                    header = ''
                elif c == '}':
                    if stack:
                        stack.pop()
                    header = ''
                elif c == ';':
                    header = ''
                else:
                    header += c
            header += ' '

    def function_at(self, line, col):
        """(qualified function name, parameter index) for the parameter at line:col."""
        lo = max(0, line - 12)
        text = '\n'.join(self.lines[lo:line - 1] + [self.lines[line - 1][:max(col - 1, 0)]])
        depth = commas = 0
        for i in range(len(text) - 1, -1, -1):
            c = text[i]
            if c in ')]':
                depth += 1
            elif c in '([':
                if depth:
                    depth -= 1
                elif c == '(':
                    m = NAME_RE.search(text[:i])
                    if not m:
                        return None
                    name = re.sub(r'\s+', '', m.group(1)) if not m.group(1).startswith('operator') \
                        else re.sub(r'\s+', ' ', m.group(1))
                    scope = self.scopes[lo + text.count('\n', 0, i)]
                    return '::'.join(scope + [name]), commas
            elif c == ',' and depth == 0:
                commas += 1
        return None


_sources = {}


def source(path):
    if path not in _sources:
        try:
            _sources[path] = Source(path)
        except OSError:
            _sources[path] = None
    return _sources[path]


def load(path):
    """Unique entries: {(file, line, param, verdict): column}."""
    entries = {}
    for file, line, col, msg, v in results(path):
        if v is None or file is None:
            continue
        m = PARAM_RE.search(msg)
        param = 'this' if 'implicit' in msg else (m.group(1) if m else msg)
        entries.setdefault((file, line, param, v), col)
    return entries


def function_key(entry, col):
    file, line, param, v = entry
    src = source(file)
    if src is None or not line or line > len(src.lines):
        return None
    fn = src.function_at(line, col or 1)
    if fn is None:
        return None
    return (fn[0], 'this' if param == 'this' else fn[1], v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('oracle')
    ap.add_argument('test')
    ap.add_argument('--exact-only', action='store_true', help='only match by file + line + parameter')
    ap.add_argument('--quiet', action='store_true', help='print the summary only')
    args = ap.parse_args()

    oracle, test = load(args.oracle), load(args.test)
    flip = {MAY: MUST_NOT, MUST_NOT: MAY}
    test_fn = {}
    if not args.exact_only:
        for e, col in test.items():
            k = function_key(e, col)
            if k:
                test_fn.setdefault(k, e)

    exact, by_function, contradicted, missing = [], [], [], []
    no_source = 0
    for e, col in sorted(oracle.items()):
        if e in test:
            exact.append(e)
            continue
        opposite = (e[0], e[1], e[2], flip[e[3]])
        k = None if args.exact_only else function_key(e, col)
        if not args.exact_only and k is None:
            no_source += 1
        if k and k in test_fn:
            by_function.append((e, test_fn[k]))
        elif opposite in test or (k and (k[0], k[1], flip[k[2]]) in test_fn):
            contradicted.append(e)
        else:
            missing.append(e)

    common = os.path.commonpath([e[0] for e in list(oracle) + list(test) if e[0] != '?'] or ['/'])
    rel = lambda f: os.path.relpath(f, common) if f != '?' else f

    print(f"oracle entries:           {len(oracle)}")
    print(f"test entries:             {len(test)}")
    print(f"included (same location): {len(exact)}")
    if not args.exact_only:
        print(f"included (same function): {len(by_function)}   (e.g. declaration <-> definition)")
    print(f"contradicted by test:     {len(contradicted)}")
    print(f"NOT included in test:     {len(missing)}")
    if no_source:
        print(f"  note: {no_source} oracle entr{'y' if no_source == 1 else 'ies'} could not be "
              f"function-matched (source unreadable or unparsed)")
    if not args.quiet:
        for title, items in (('contradicted', contradicted), ('not included', missing)):
            if items:
                print(f"\n--- {title} (paths under {common}) ---")
                for f, line, param, v in items:
                    print(f"  {rel(f)}:{line}  '{param}'  {v}")
    return 1 if missing or contradicted else 0


if __name__ == '__main__':
    sys.exit(main())
