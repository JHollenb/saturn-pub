"""Maintainer-only plan builder; evaluator spans are never staged for execution."""


def plan():
    template = "Description: icy\nTemperature: cold\nDescription: fiery\nTemperature: hot\nDescription: {source}\nTemperature:"
    variants = (
        ("anchor", template),
        ("fresh-prefix", "Complete the pattern.\n" + template),
        ("question-suffix", template.rsplit("\nTemperature:", 1)[0] + "\nThe temperature is"),
        ("sentence-prefix", "Here are some examples.\nAn icy object feels cold.\nA fiery object feels hot.\nA {source} object feels"),
        ("intervening-note", template.rsplit("\nTemperature:", 1)[0] + "\nNote: keep the original labels.\nTemperature:"),
    )
    runtime, evaluation = [], []
    for identifier, form in variants:
        case = {"id": identifier}
        reference = {"id": identifier, "split": "anchor" if identifier == "anchor" else "fresh-live-resolver-context"}
        for side, word in (("recipient", "freezing"), ("donor", "boiling")):
            text = form.format(source=word)
            case[side] = {"text": text}
            start = text.rindex(word)
            reference[side] = {"source_span": [start, start + len(word)], "source_word": word}
        runtime.append(case)
        evaluation.append(reference)
    return runtime, {"contexts": evaluation,
        "scope": "Evaluator-only spans; this file and this builder are excluded from the worker payload"}
