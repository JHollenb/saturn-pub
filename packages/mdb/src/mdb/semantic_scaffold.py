"""Small declared task organisms, source spans and independent native consumers.

Task labels and source spans are supplied. No model output enters resolution.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class MarkedPrompt:
    text: str
    source_span: tuple[int, int]
    wrong_span: tuple[int, int]


def marked(template: str, source: str, wrong: str) -> MarkedPrompt:
    """Expand exactly one source and one matched distractor marker."""
    if template.count("{source}") != 1 or template.count("{wrong}") != 1:
        raise ValueError("declare exactly one source and one wrong-source marker")
    pieces = re.split(r"(\{source\}|\{wrong\})", template)
    text, spans = "", {}
    for piece in pieces:
        if piece in ("{source}", "{wrong}"):
            name = piece[1:-1]
            value = source if name == "source" else wrong
            spans[name] = (len(text), len(text) + len(value))
            text += value
        else:
            text += piece
    return MarkedPrompt(text, spans["source"], spans["wrong"])


def resolve_span(offsets, span):
    lo, hi = span
    if not 0 <= lo < hi:
        raise ValueError("invalid source character span")
    rows = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo]
    covered = {j for i in rows for j in range(max(lo, offsets[i][0]), min(hi, offsets[i][1]))}
    if covered != set(range(lo, hi)):
        raise ValueError("source span is not covered by native token offsets")
    return rows


def plan():
    cases = []
    def add(lane, split, template, source, donor, wrong, expected, donor_expected, **extra):
        rec, other = marked(template, source, wrong), marked(template, donor, wrong)
        cases.append({"id": lane + "-" + split, "lane": lane, "split": split,
                      "recipient": {"text": rec.text, "source_span": list(rec.source_span), "wrong_span": list(rec.wrong_span)},
                      "donor": {"text": other.text, "source_span": list(other.source_span), "wrong_span": list(other.wrong_span)},
                      "expected": expected, "donor_expected": donor_expected,
                      "source": source, "donor_source": donor, "wrong_source": wrong,
                      "semantic_authority": "supplied task/source labels; native outcomes measured", **extra})
    examples = "Description: icy\nTemperature: cold\nDescription: fiery\nTemperature: hot\n"
    add("semantics", "discovery", examples + "Description{wrong} {source}\nTemperature:",
        "freezing", "boiling", ":", "cold", "hot")
    add("semantics", "transfer", "Describe the temperature.\nObject: water\nCondition{wrong} {source}\nTemperature:",
        "chilly", "scorching", ":", "cold", "hot")
    copy_examples = "Input: cedar\nCopy: cedar\nInput: maple\nCopy: maple\n"
    add("copy", "discovery", copy_examples + "Distractor: {wrong}\nInput: {source}\nCopy:",
        "amber", "violet", "silver", "amber", "violet")
    add("copy", "transfer", "Copy the input exactly.\nIgnored note: {wrong}\nInput: {source}\nCopy:",
        "lumen", "nexus", "marble", "lumen", "nexus")
    add("recall", "discovery", "Ada's code is {source}.\nNoah's code is {wrong}.\nAda's code:",
        "amber", "silver", "violet", "amber", "silver")
    add("recall", "transfer", "Tomas's code is {wrong}.\nLena's code is {source}.\nQuestion: What is Lena's code?\nAnswer:",
        "cedar", "maple", "violet", "cedar", "maple")
    add("coding", "discovery", "def bump(x):\n    return x + {source}\n\nassert bump({wrong}) == ",
        "1", "2", "4", "5", "6", call=4, function="bump", constant=1, donor_constant=2)
    add("coding", "transfer", "# Return the incremented value.\ndef advance(value):\n    return value + {source}\n\nassert advance({wrong}) == ",
        "3", "4", "5", "8", "9", call=5, function="advance", constant=3, donor_constant=4)
    return cases


def evaluate(case, output, *, donor=False):
    answer = case["donor_expected"] if donor else case["expected"]
    text = output.lstrip()
    first = re.match(r"[A-Za-z]+|-?\d+", text)
    observed = first.group(0) if first else None
    result = {"expected": answer, "observed_first_word": observed,
              "correct": observed == answer, "output": output,
              "consumer": "exact first lexical item"}
    if case["lane"] == "coding":
        # Only integer literals are admitted to the generated completion.
        # The resulting function/assert is compiled and run, with no imports,
        # calls from generated text, or executable generated suffix.
        match = re.match(r"-?\d+(?=\s|$|[;#])", text)
        if match:
            literal = match.group(0)
            constant = case["donor_constant"] if donor else case["constant"]
            name = case["function"]
            source = f"def {name}(x):\n    return x + {constant}\nassert {name}({case['call']}) == {literal}\n"
            ast.parse(source)
            try:
                exec(compile(source, "<native-model-completion>", "exec"), {"__builtins__": {}})
                passed = True
            except AssertionError:
                passed = False
            result.update(consumer="compiled Python function and assertion from integer-only completion",
                          executable_source=source, executable_assertion_passed=passed)
        else:
            result.update(consumer="integer-only code admission", executable_assertion_passed=None)
    return result


def calibration_plan():
    """Additive organisms after the original free-format instrument misses."""
    examples = "Description: icy\nTemperature: cold\nDescription: fiery\nTemperature: hot\n"
    recall_examples = (
        "Read the facts and answer with one word.\n"
        "Facts: Mia has an apple. Leo has a pear.\nQuestion: What does Mia have?\nAnswer: apple\n\n"
        "Facts: Eva has a book. Sam has a cup.\nQuestion: What does Sam have?\nAnswer: cup\n\n")
    definitions = [
        ("semantics-calibrated-transfer", "semantics", "transfer",
         examples + "Object: water\nDescription{wrong} {source}\nTemperature:",
         "chilly", "scorching", ":", "cold", "hot"),
        ("recall-calibrated-discovery", "recall", "discovery",
         recall_examples + "Facts: Ada has {source}. Noah has {wrong}.\nQuestion: What does Ada have?\nAnswer:",
         "amber", "silver", "violet", "amber", "silver"),
        ("recall-calibrated-transfer", "recall", "transfer",
         recall_examples + "Facts: Tomas has {wrong}. Lena has {source}.\nQuestion: What does Lena have?\nAnswer:",
         "cedar", "maple", "violet", "cedar", "maple"),
    ]
    cases = []
    for key, lane, split, template, source, donor, wrong, expected, donor_expected in definitions:
        rec, other = marked(template, source, wrong), marked(template, donor, wrong)
        cases.append({"id": key, "lane": lane, "split": split,
            "recipient": {"text": rec.text, "source_span": list(rec.source_span), "wrong_span": list(rec.wrong_span)},
            "donor": {"text": other.text, "source_span": list(other.source_span), "wrong_span": list(other.wrong_span)},
            "source": source, "donor_source": donor, "wrong_source": wrong,
            "expected": expected, "donor_expected": donor_expected,
            "semantic_authority": "additive calibration after native prompt-format misses; supplied examples and source labels"})
    return cases


def localization_plan():
    """Matched coding bands and explicit copy-span geometry controls."""
    initial = plan()
    cases = [dict(c, id=c["id"] + "-depth", split="transfer") for c in initial if c["lane"] == "coding"]
    template = "Copy the input exactly.\nIgnored note: {wrong}\nInput: {source}\nCopy:"
    a, b = marked(template, "copper", "marble"), marked(template, "silver", "marble")
    cases.append({"id": "copy-single-word-control", "lane": "copy", "split": "transfer",
        "recipient": {"text": a.text, "source_span": list(a.source_span), "wrong_span": list(a.wrong_span)},
        "donor": {"text": b.text, "source_span": list(b.source_span), "wrong_span": list(b.wrong_span)},
        "source": "copper", "donor_source": "silver", "wrong_source": "marble", "expected": "copper",
        "donor_expected": "silver", "semantic_authority": "new lexical control with frozen copy route"})
    original = next(c for c in initial if c["id"] == "copy-transfer")
    cases.append(dict(original, id="copy-span-localization", span_lowering=True))
    return cases
