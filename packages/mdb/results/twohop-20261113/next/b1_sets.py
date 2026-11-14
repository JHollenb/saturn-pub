"""B1 candidate sets and per-item truths (advisor memo 2026-11-14 §1.7), tokenizer only.

python b1_sets.py TOKENIZER_DIR > b1_sets.json

Sets (first token of " {word}"): countries (every country token of the two-hop and depth
panels), cities (capitals: gen_panel CAPITAL + CAP2), languages (LANG + Indonesian, Persian),
persons_surname and persons_given (every person in DEPTH per-lists and every creator of a
DEPTH work, deduplicated). Truths per 1c item: country, city (= the latent capital), language
(= the answer), person (d3-per: the person; d4-work: the work's creator; else None).
"""
import json
import sys

sys.path.insert(0, "../ext")
sys.path.insert(0, "..")
from gen_ext import CAP2, DEPTH, LANG  # noqa: E402
from gen_panel import tokenizer  # noqa: E402

# person -> (given, surname) as the name is usually written; one-name people repeat it
PEOPLE = {
    "Voltaire": ("Voltaire", "Voltaire"), "Georges Bizet": ("Georges", "Bizet"),
    "Claude Monet": ("Claude", "Monet"), "Edith Piaf": ("Edith", "Piaf"),
    "Julius Caesar": ("Julius", "Caesar"), "Sophia Loren": ("Sophia", "Loren"),
    "Ennio Morricone": ("Ennio", "Morricone"), "Marlene Dietrich": ("Marlene", "Dietrich"),
    "Alexander von Humboldt": ("Alexander", "Humboldt"), "Walter Benjamin": ("Walter", "Benjamin"),
    "Lope de Vega": ("Lope", "Vega"), "Pedro Calderon de la Barca": ("Pedro", "Calderon"),
    "Hokusai": ("Hokusai", "Hokusai"), "Yukio Mishima": ("Yukio", "Mishima"),
    "Natsume Soseki": ("Natsume", "Soseki"), "Alexander Pushkin": ("Alexander", "Pushkin"),
    "Fyodor Dostoevsky": ("Fyodor", "Dostoevsky"), "Mikhail Lermontov": ("Mikhail", "Lermontov"),
    "Franz Kafka": ("Franz", "Kafka"), "Rainer Maria Rilke": ("Rainer", "Rilke"),
    "Virginia Woolf": ("Virginia", "Woolf"), "Charlie Chaplin": ("Charlie", "Chaplin"),
    "John Keats": ("John", "Keats"), "Franz Schubert": ("Franz", "Schubert"),
    "Johann Strauss II": ("Johann", "Strauss"), "Ludwig Wittgenstein": ("Ludwig", "Wittgenstein"),
    "Plato": ("Plato", "Plato"), "Pericles": ("Pericles", "Pericles"),
    "Baruch Spinoza": ("Baruch", "Spinoza"), "Soren Kierkegaard": ("Soren", "Kierkegaard"),
    "Niels Bohr": ("Niels", "Bohr"), "Harry Houdini": ("Harry", "Houdini"),
    "Erno Rubik": ("Erno", "Rubik"), "Fernando Pessoa": ("Fernando", "Pessoa"),
    "Greta Garbo": ("Greta", "Garbo"), "August Strindberg": ("August", "Strindberg"),
    "Naguib Mahfouz": ("Naguib", "Mahfouz"), "Marie Curie": ("Marie", "Curie"),
}
# work -> creator (advisor memo §1.7; asserted, written before scoring)
CREATOR = {
    "Candide": "Voltaire", "the opera Carmen": "Georges Bizet",
    "the painting Impression, Sunrise": "Claude Monet",
    "the Commentaries on the Gallic War": "Julius Caesar", "The Arcades Project": "Walter Benjamin",
    "the play Life Is a Dream": "Pedro Calderon de la Barca", "the play Fuenteovejuna": "Lope de Vega",
    "The Great Wave off Kanagawa": "Hokusai", "the novel The Sound of Waves": "Yukio Mishima",
    "the novel Kokoro": "Natsume Soseki", "Eugene Onegin": "Alexander Pushkin",
    "the novel Crime and Punishment": "Fyodor Dostoevsky",
    "the novel A Hero of Our Time": "Mikhail Lermontov", "The Metamorphosis": "Franz Kafka",
    "the novel The Trial": "Franz Kafka", "the Duino Elegies": "Rainer Maria Rilke",
    "the novel Mrs Dalloway": "Virginia Woolf", "the film The Great Dictator": "Charlie Chaplin",
    "the poem Ode to a Nightingale": "John Keats", "the Unfinished Symphony": "Franz Schubert",
    "The Blue Danube waltz": "Johann Strauss II",
    "the Tractatus Logico-Philosophicus": "Ludwig Wittgenstein", "The Republic": "Plato",
    "the book Fear and Trembling": "Soren Kierkegaard", "the book Either/Or": "Soren Kierkegaard",
    "the Rubik's Cube": "Erno Rubik", "The Book of Disquiet": "Fernando Pessoa",
    "the play Miss Julie": "August Strindberg", "the novel Palace Walk": "Naguib Mahfouz",
}


def main():
    enc = tokenizer(sys.argv[1])

    def first(word):
        return enc(" " + word)[0]

    base = json.load(open("candidates.json"))
    ext = json.load(open("candidates_ext.json"))
    items = base["items"] + ext["items"]
    people = set()
    for spec in DEPTH.values():
        people.update(spec["per"])
        for w in spec["work"]:
            if w not in CREATOR:
                raise SystemExit(f"no creator for {w!r}")
            people.add(CREATOR[w])
    missing = sorted(p for p in people if p not in PEOPLE)
    if missing:
        raise SystemExit(f"no name split for {missing}")
    sets = {
        "countries": sorted({i["country_token"] for i in items if i.get("country_token")}),
        "cities": sorted(set(base["capital_tokens"].values()) | {first(c) for c in CAP2.values()}),
        "languages": sorted({first(x) for x in list(LANG.values()) + ["Indonesian", "Persian"]}),
        "persons_surname": sorted({first(PEOPLE[p][1]) for p in people}),
        "persons_given": sorted({first(PEOPLE[p][0]) for p in people}),
    }
    truths = {}
    for i in ext["items"]:
        if i["family"] not in ("d1", "d2", "d3-lm", "d3-per", "d4-work"):
            continue
        person = None
        if i["family"] == "d3-per":
            person = i["descriptor"]
        elif i["family"] == "d4-work":
            person = CREATOR[i["descriptor"]]
        truths[i["id"]] = {
            "country": i["country_token"], "city": i["capital_token"], "language": i["answer"],
            "person": person,
            "persons_surname": first(PEOPLE[person][1]) if person else None,
            "persons_given": first(PEOPLE[person][0]) if person else None,
            "bridge": i["bridge"], "family": i["family"], "span": i["span"], "row": i["row"],
            "n": i["n"]}
    for name, toks in sets.items():
        for i in truths.values():
            key = {"countries": "country", "cities": "city", "languages": "language"}.get(name, name)
            if i.get(key) is not None and i[key] not in toks:
                raise SystemExit(f"{name}: truth {i[key]} not in the set")
    json.dump({"sets": sets, "sizes": {k: len(v) for k, v in sets.items()},
               "creator": CREATOR, "people": PEOPLE, "truths": truths}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
