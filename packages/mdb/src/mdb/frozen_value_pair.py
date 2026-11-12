"""Prospective contexts and controls for the frozen L20 source-value pair."""
PAIR = ((20, 0), (20, 1))
CAPTURE_LAYERS = (20, 21, 26)


def cases():
    original = "Description: icy\nTemperature: cold\nDescription: fiery\nTemperature: hot\nDescription: {source}\nTemperature:"
    variants = (
        ("anchor", "Original discovery context repeated", original),
        ("prefix-shift", "Prefix moves the absolute source position", "Please classify the temperature.\n\n" + original),
        ("example-order", "Hot example precedes cold example",
            "Description: fiery\nTemperature: hot\nDescription: icy\nTemperature: cold\nDescription: {source}\nTemperature:"),
        ("field-wording", "Field names changed",
            "Condition: icy\nClass: cold\nCondition: fiery\nClass: hot\nCondition: {source}\nClass:"),
        ("sentence-wording", "Sentence grammar replaces the field grammar",
            "An icy object feels cold.\nA fiery object feels hot.\nA {source} object feels"),
    )
    rows = []
    for identifier, axis, template in variants:
        row = {"id": identifier, "axis": axis, "semantic_authority": "supplied temperature task and source span",
            "split": "anchor-replay" if identifier == "anchor" else "prospective-fixed-route-variant"}
        for side, word in (("recipient", "freezing"), ("donor", "boiling")):
            text = template.format(source=word)
            start = text.rindex(word)
            row[side] = {"text": text, "source_span": [start, start + len(word)]}
        rows.append(row)
    return rows


def arms():
    return [{"arm": name, "kind": kind, "cells": cells, "dose": 1}
        for name, kind, cells in (
            ("native-recipient", "native", PAIR), ("native-donor", "donor-native", PAIR),
            ("sham-pair", "sham", PAIR), ("pair", "transfer", PAIR),
            ("only-KV0", "transfer", ((20, 0),)), ("only-KV1", "transfer", ((20, 1),)),
            ("reciprocal-pair", "reciprocal", PAIR), ("wrong-payload-pair", "wrong-payload", PAIR),
            ("wrong-target-pair", "wrong-target", PAIR),
            ("full-late", "transfer", tuple((layer, kv) for layer in range(19, 28) for kv in range(2))))]
