"""
Rules parser — unified rule-file reader.

Parses Data/{dataset}/rules.txt and extracts rules per baseline.

File format:
[HOLOCLEAN_DC]
t1&t2&EQ(t1.attr1,t2.attr2)&IQ(t1.attr3,t2.attr3)

[UNICLEAN]
Number("attr")
AttrRelation(["lhs"], ["rhs"], "name")

[HORIZON_FD]
lhs => rhs

[MSE_ATTRIBUTES]
attr1
attr2
"""

import os
import re
from typing import List, Dict, Optional, Tuple


def parse_rules_file(rules_path: str) -> Dict[str, List[str]]:
    """
    Parse the unified rule file.

    Args:
        rules_path: path to rules.txt

    Returns:
        Dict keyed by section name, with lists of rule strings as values.
    """
    if not os.path.exists(rules_path):
        return {}

    sections = {}
    current_section = None

    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip blank lines and comments
            if not line or line.startswith('#'):
                continue

            # Detect section headers
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

    return sections


def get_holoclean_dcs(rules_path: str) -> List[str]:
    """Return HoloClean denial constraints."""
    sections = parse_rules_file(rules_path)
    return sections.get('HOLOCLEAN_DC', [])


def get_uniclean_cleaners(rules_path: str) -> List[str]:
    """Return UniClean cleaner definition strings."""
    sections = parse_rules_file(rules_path)
    return sections.get('UNICLEAN', [])


def get_horizon_fds(rules_path: str) -> List[Tuple[str, str]]:
    """
    Return Horizon functional-dependency rules.

    Returns:
        List of (lhs, rhs) tuples.
    """
    sections = parse_rules_file(rules_path)
    fds = []

    for line in sections.get('HORIZON_FD', []):
        # Accept both "=>" and the Unicode arrow
        if '=>' in line:
            parts = line.split('=>')
        elif '\u21d2' in line:
            parts = line.split('\u21d2')
        else:
            continue

        if len(parts) == 2:
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            fds.append((lhs, rhs))

    return fds


def get_mse_attributes(rules_path: str) -> List[str]:
    """Return the MSE-evaluation attribute list."""
    sections = parse_rules_file(rules_path)
    return sections.get('MSE_ATTRIBUTES', [])


def parse_uniclean_cleaner(cleaner_str: str):
    """
    Parse a UniClean cleaner string into an executable object.

    Args:
        cleaner_str: e.g. 'Number("age")' or 'AttrRelation(["a"], ["b"], "0")'

    Returns:
        A UniClean cleaner object.
    """
    # Dynamic import of UniClean cleaners
    try:
        from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
        from SampleScrubber.cleaner.multiple import AttrRelation
    except ImportError:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Methods', 'UniClean'))
        from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
        from SampleScrubber.cleaner.multiple import AttrRelation

    # Sandboxed eval for cleaner definitions
    local_vars = {
        'Number': Number,
        'Pattern': Pattern,
        'Outlier': Outlier,
        'Date': Date,
        'AttrRelation': AttrRelation
    }

    try:
        return eval(cleaner_str, {"__builtins__": {}}, local_vars)
    except Exception as e:
        print(f"Warning: Failed to parse cleaner '{cleaner_str}': {e}")
        return None


def get_uniclean_cleaner_objects(rules_path: str) -> List:
    """
    Return a list of UniClean cleaner objects.

    Args:
        rules_path: path to rules.txt

    Returns:
        List of cleaner objects.
    """
    cleaner_strs = get_uniclean_cleaners(rules_path)
    cleaners = []

    for cleaner_str in cleaner_strs:
        cleaner = parse_uniclean_cleaner(cleaner_str)
        if cleaner:
            cleaners.append(cleaner)

    return cleaners


def write_holoclean_dc_file(rules_path: str, output_path: str) -> str:
    """
    Extract HoloClean DCs from the unified rule file and write them to a separate file.

    Args:
        rules_path: path to the unified rule file
        output_path: output DC file path

    Returns:
        The output file path.
    """
    dcs = get_holoclean_dcs(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for dc in dcs:
            f.write(dc + '\n')

    return output_path


def write_horizon_fd_file(rules_path: str, output_path: str) -> str:
    """
    Extract Horizon FDs from the unified rule file and write them to a separate file.

    Args:
        rules_path: path to the unified rule file
        output_path: output FD file path

    Returns:
        The output file path.
    """
    fds = get_horizon_fds(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for lhs, rhs in fds:
            f.write(f"{lhs} => {rhs}\n")

    return output_path


# Convenience helpers
def get_dataset_rules(dataset_name: str, data_dir: str = 'Data') -> Dict[str, List[str]]:
    """
    Return all rules for the named dataset.

    Args:
        dataset_name: dataset name
        data_dir: root data directory

    Returns:
        Rule dict.
    """
    rules_path = os.path.join(data_dir, dataset_name, 'rules.txt')
    return parse_rules_file(rules_path)


if __name__ == '__main__':
    # Quick self-test
    import sys
    if len(sys.argv) > 1:
        rules_path = sys.argv[1]
    else:
        rules_path = 'Data/beers/rules.txt'

    print(f"Parsing: {rules_path}")
    sections = parse_rules_file(rules_path)

    for section, rules in sections.items():
        print(f"\n[{section}]")
        for rule in rules:
            print(f"  {rule}")
