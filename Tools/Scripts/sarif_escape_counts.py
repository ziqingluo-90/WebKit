#!/usr/bin/env python3
"""Count lifetime-safety escape-inference results in SARIF files.

  may escape      : "... is inferred to escape the function"
  must not escape : "... is inferred to not escape the function"

Results are de-duplicated by (file, line, column, message) unless --no-dedup is given,
since a header compiled in several TUs reports the same parameter more than once.

Usage: sarif_escape_counts.py [--by-file] [--no-dedup] FILE.sarif [FILE.sarif ...]
"""
import argparse
import collections
import json
import os
import sys
from urllib.parse import unquote, urlparse

MAY, MUST_NOT = 'may escape', 'must not escape'


def verdict(message):
    if 'is inferred to not escape' in message:
        return MUST_NOT
    if 'is inferred to escape' in message:
        return MAY
    return None


def uri_to_path(uri):
    if not uri:
        return '?'
    p = urlparse(uri)
    return os.path.realpath(unquote(p.path)) if p.scheme in ('', 'file') else uri


def artifact_resolver(run):
    """Maps an artifactLocation to a path. Prefers the location's own uri, then the
    artifact whose stored index matches (clang can list artifacts out of index order),
    then the array position."""
    arts = run.get('artifacts', [])
    by_stored = {a['location']['index']: a['location'].get('uri')
                 for a in arts if 'index' in a.get('location', {})}

    def resolve(loc):
        if loc.get('uri'):
            return uri_to_path(loc['uri'])
        i = loc.get('index')
        if i in by_stored:
            return uri_to_path(by_stored[i])
        if isinstance(i, int) and 0 <= i < len(arts):
            return uri_to_path(arts[i]['location'].get('uri'))
        return '?'
    return resolve


def results(path):
    """Yields (file, line, column, message, verdict) for each located result."""
    with open(path) as f:
        doc = json.load(f)
    for run in doc.get('runs', []):
        resolve = artifact_resolver(run)
        for r in run.get('results', []):
            msg = r.get('message', {}).get('text', '')
            locs = r.get('locations') or []
            if not locs:
                yield None, None, None, msg, verdict(msg)
                continue
            pl = locs[0].get('physicalLocation', {})
            region = pl.get('region', {})
            yield (resolve(pl.get('artifactLocation', {})), region.get('startLine'),
                   region.get('startColumn'), msg, verdict(msg))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('sarif', nargs='+')
    ap.add_argument('--by-file', action='store_true', help='also break counts down per source file')
    ap.add_argument('--no-dedup', action='store_true', help='count every result, including duplicates')
    args = ap.parse_args()

    seen = set()
    totals = collections.Counter()
    per_file = collections.defaultdict(collections.Counter)
    dups = other = 0
    for path in args.sarif:
        for file, line, col, msg, v in results(path):
            if v is None:
                other += 1
                continue
            key = (file, line, col, msg)
            if not args.no_dedup and key in seen:
                dups += 1
                continue
            seen.add(key)
            kind = 'implicit this' if 'implicit' in msg else 'parameter'
            totals[(v, kind)] += 1
            per_file[file][v] += 1

    for v in (MAY, MUST_NOT):
        params, this = totals[(v, 'parameter')], totals[(v, 'implicit this')]
        print(f"{v + ':':17s}{params + this:6d}   (parameters {params}, implicit 'this' {this})")
    print(f"{'other results:':17s}{other:6d}   (notes / other rules)")
    if not args.no_dedup:
        print(f"{'duplicates:':17s}{dups:6d}   (skipped)")

    if args.by_file:
        common = os.path.commonpath([f for f in per_file if f != '?']) if per_file else ''
        print(f"\n{'may':>6s} {'must-not':>9s}  file (under {common})")
        for file in sorted(per_file):
            c = per_file[file]
            print(f"{c[MAY]:6d} {c[MUST_NOT]:9d}  {os.path.relpath(file, common) if file != '?' else '?'}")


if __name__ == '__main__':
    sys.exit(main())
