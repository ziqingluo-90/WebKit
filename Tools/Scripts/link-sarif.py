#!/usr/bin/env python3
"""Merge all .sarif files under a directory into a single sarif file using a batched tree-reduction.

Faster than folding files in one at a time (O(n) sequential ratchet calls):
groups files into batches of --batch-size, merges each batch with a single
`ratchet link` call, then repeats on the resulting merged files until fewer
than --batch-size remain, at which point a final merge produces the output.

Before each `ratchet link` the inputs are run through add-logical-location.py,
which names the reported type in each interop result's logicalLocations. Ratchet
merges results keyed on, among other things, logicalLocations, so without this
every instantiation of a template reported at one declaration site shares a
merge key and collapses to whichever translation unit merged first. On WTF this
is the difference between 3244 and 1280 results, and between 31 and 1 at
wtf/RefPtr.h:84.

It has to be redone between rounds, not just once up front: ratchet does not
copy logicalLocations into its output (swift-toolchain-sarif's
Location.MergeState.merge carries only physicalLocation), so a round's output is
unannotated again and the next round would collapse it. Pass
--no-logical-locations to skip all of this.
"""
import argparse
import concurrent.futures
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

BATCH_SIZE = 100

# Repo-relative directory prefixes containing test code/resources. Results
# whose artifact location falls under one of these are dropped before linking.
TEST_DIRECTORIES = (
    "LayoutTests/",
    "JSTests/",
    "PerformanceTests/",
    "ManualTests/",
    "WebDriverTests/",
    "Tools/TestWebKitAPI/",
    "Tools/WebKitTestRunner/",
    "Tools/TestRunnerShared/",
    "Tools/DumpRenderTree/",
    "Tools/CSSTestSuiteHarness/",
    "Tools/lldb/lldbWebKitTester/",
    "Tools/EditingHistory/EditingHistoryTests/",
    "Source/ThirdParty/gtest/",
    "Source/ThirdParty/gmock/",
    "Source/ThirdParty/qunit/test/",
    "Source/ThirdParty/ANGLE/src/tests/",
    "Source/ThirdParty/skia/tests/",
    "Source/ThirdParty/libwebrtc/Source/webrtc/test/",
    "Source/JavaScriptCore/API/tests/",
    "Source/JavaScriptCore/testmem/",
    "Source/JavaScriptCore/wasm/debugger/tests/",
    "Source/WebCore/testing/",
    "Source/WebInspectorUI/UserInterface/Test/",
    "Source/bmalloc/libpas/src/test/",
    "Source/bmalloc/mimalloc/mimalloc/test/",
)

# Standalone JSC "Test Tools" sources that sit next to product code rather
# than in a dedicated test directory (see the "Test Tools" aggregate target
# in JavaScriptCore.xcodeproj).
TEST_FILES = (
    "Source/JavaScriptCore/assembler/testmasm.cpp",
    "Source/JavaScriptCore/b3/air/testair.cpp",
    "Source/JavaScriptCore/b3/testb3.h",
    "Source/JavaScriptCore/dfg/testdfg.cpp",
    "Source/JavaScriptCore/dynbench.cpp",
    "Source/JavaScriptCore/testRegExp.cpp",
    "Source/JavaScriptCore/wasm/debugger/testwasmdebugger.cpp",
) + tuple(f"Source/JavaScriptCore/b3/testb3_{i}.cpp" for i in range(1, 9))


def find_sarif_files(root):
    result = subprocess.run(
        ["find", root, "-name", "*.sarif"], capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


def is_test_uri(uri):
    return any(f"/{d}" in uri for d in TEST_DIRECTORIES) or any(
        uri.endswith(f) for f in TEST_FILES
    )


def result_artifact_uri(result, artifacts):
    for location in result.get("locations", []):
        artifact_location = location.get("physicalLocation", {}).get("artifactLocation", {})
        uri = artifact_location.get("uri")
        if uri is None:
            index = artifact_location.get("index")
            if index is not None and 0 <= index < len(artifacts):
                uri = artifacts[index].get("location", {}).get("uri")
        if uri:
            yield uri


def filter_test_code(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)
    removed = []
    for run in data.get("runs", []):
        artifacts = run.get("artifacts", [])
        kept = []
        for result in run.get("results", []):
            uris = list(result_artifact_uri(result, artifacts))
            if any(is_test_uri(uri) for uri in uris):
                removed.append((uris[0] if uris else "", result.get("message", {}).get("text", "")))
            else:
                kept.append(result)
        run["results"] = kept
    with open(output_path, "w") as f:
        json.dump(data, f)
    return removed


def has_results(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return any(run.get("results") for run in data.get("runs", []))


def load_sibling(filename):
    """Import a sibling script whose name isn't a valid module name."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(
        filename[:-3].replace("-", "_"), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


add_logical_location = load_sibling("add-logical-location.py")


def annotate(path):
    """Add logicalLocations to one sarif file in place. Returns (added, unparsed)."""
    with open(path) as f:
        data = json.load(f)
    changes, unparsed, _skipped = add_logical_location.annotate(data)
    if changes:
        with open(path, "w") as f:
            json.dump(data, f)
    return len(changes), len(unparsed)


def annotate_all(files):
    with concurrent.futures.ProcessPoolExecutor() as pool:
        counts = list(pool.map(annotate, files))
    return sum(c for c, _ in counts), sum(u for _, u in counts)


def merge(ratchet, inputs, output):
    subprocess.run(
        [ratchet, "link", "--output", output, *inputs],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output


def chunk(items, size):
    return [items[i : i + size] for i in range(0, len(items), size)]


def strip_invocations(path):
    with open(path) as f:
        data = json.load(f)
    for run in data.get("runs", []):
        run.pop("invocations", None)
    with open(path, "w") as f:
        json.dump(data, f)


def resolve_ratchet(arg):
    if arg:
        return arg
    found = shutil.which("ratchet")
    if found:
        return found
    sys.exit(
        "error: could not find a 'ratchet' binary on PATH; pass --ratchet /path/to/ratchet"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dir",
        nargs="?",
        default=".",
        help="directory to search for .sarif files (default: current directory)",
    )
    parser.add_argument(
        "--ratchet",
        default=None,
        help="path to the ratchet binary (default: 'ratchet' resolved from PATH)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="out.sarif",
        help="path to write the merged sarif file to (default: ./out.sarif)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"max number of files to merge per ratchet call (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-logical-locations",
        action="store_true",
        help="skip add-logical-location.py; results at one declaration site will "
             "collapse to a single instantiation (for testing)",
    )
    args = parser.parse_args()

    ratchet = resolve_ratchet(args.ratchet)
    search_dir = os.path.abspath(args.dir)
    out_path = os.path.abspath(args.output)
    batch_size = args.batch_size

    print(f"scanning {search_dir} for .sarif files...")
    all_files = find_sarif_files(search_dir)
    if not all_files:
        print(f"No .sarif files found under {search_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"1. total files: {len(all_files)}")

    with concurrent.futures.ProcessPoolExecutor() as pool:
        keep = list(pool.map(has_results, all_files))
    files = [f for f, k in zip(all_files, keep) if k]

    print(f"2. files after removing empty ones: {len(files)}")
    if not files:
        print(f"No .sarif files with results found under {search_dir}", file=sys.stderr)
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="link-sarif-")
    try:
        filtered_dir = os.path.join(tmpdir, "filtered")
        os.makedirs(filtered_dir, exist_ok=True)
        filtered_files = [os.path.join(filtered_dir, f"{i}.sarif") for i in range(len(files))]
        with concurrent.futures.ProcessPoolExecutor() as pool:
            removed_lists = list(pool.map(filter_test_code, files, filtered_files))
        removed_entries = [entry for sublist in removed_lists for entry in sublist]

        with concurrent.futures.ProcessPoolExecutor() as pool:
            keep = list(pool.map(has_results, filtered_files))
        files = [f for f, k in zip(filtered_files, keep) if k]

        print(f"3. files after filtering test code: {len(files)} (removed {len(removed_entries)} test-code results)")
        for uri in dict.fromkeys(uri for uri, _ in removed_entries):
            print(f"   removed items of {uri}")
        if not files:
            print("No .sarif results remain after filtering test code", file=sys.stderr)
            sys.exit(1)

        annotating = not args.no_logical_locations
        if annotating:
            added, unparsed = annotate_all(files)
            print(f"4. added logical locations: {added} results in {len(files)} files")
            if unparsed:
                print(f"   warning: {unparsed} results matched a rule but no type could "
                      f"be parsed from the message; run add-logical-location.py "
                      f"directly to see them")
        else:
            print("4. skipping logical locations (--no-logical-locations)")

        round_num = 0
        while len(files) >= batch_size:
            groups = chunk(files, batch_size)
            print(f"5. round {round_num}: processing {len(groups)} groups in parallel ({len(files)} files, <= {batch_size} per group)")

            outputs = [os.path.join(tmpdir, f"out_{round_num}_{i}.sarif") for i in range(len(groups))]
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = [pool.submit(merge, ratchet, group, out) for group, out in zip(groups, outputs)]
                for i, fut in enumerate(futures):
                    fut.result()
                    print(f"  [{i + 1}/{len(groups)}] linked group {i} ({len(groups[i])} files) into {outputs[i]}")

            files = outputs
            # ratchet does not copy logicalLocations into its output, so this
            # round's results are unannotated again and the next link would
            # collapse them. Re-annotate before merging them any further.
            if annotating:
                added, _ = annotate_all(files)
                print(f"  re-added logical locations to {len(files)} merged files ({added} results)")
            round_num += 1

        print(f"final merge: {len(files)} files -> {out_path}")
        merge(ratchet, files, out_path)
        strip_invocations(out_path)
        print(f"done: {out_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
