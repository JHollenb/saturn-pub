"""Candidates for the depth and difficulty extension (tokenizer only, no model).

python gen_ext.py TOKENIZER_JSON > candidates_ext.json

Depth panel (final relation = official language, single-token answers; the hub entity is the
country, never written):
  d1      "The official language of {Country} is"                                   (0 latent)
  d2      "... of the country whose capital is {City} is"                           (country)
  d3-lm   "... of the country whose capital is home to {Landmark} is"               (city, country)
  d3-per  "... of the country whose capital is the birthplace of {Person} is"       (city, country)
  d4-work "... of the country whose capital is the birthplace of the creator of {Work} is"
                                                                         (person, city, country)
Difficulty panel (2-hop, built on the first experiment's families):
  para-currency / para-city   paraphrased question form, descriptor not adjacent to the answer
  dist-currency / dist-city   a same-family distractor entity mentioned first
  long-currency / long-city   ~250 tokens of neutral filler before the question
  obsc-currency / obsc-city   less famous descriptors
  lang-currency / lang-city   hop 2 = official language instead of capital
Item fields follow candidates.json: span = descriptor rows, row = last descriptor row,
answer = primary answer token, answer_accept = accepted argmax tokens, country_token /
country_accept = the hub country, capital_token = the latent capital (depth panel),
hop1_ids = front -> country alone, hop2_ids = final relation on the named country.
"""
import json
import sys

sys.path.insert(0, "..")
from gen_panel import (CAPITAL, COUNTRY_ALIASES, COUNTRY_NAME, FAMILIES, PREFIX_HOP1, PREFIX_TWO,
                       tokenizer)  # noqa: E402

LANG = {"France": "French", "Italy": "Italian", "Germany": "German", "Spain": "Spanish",
        "Japan": "Japanese", "Russia": "Russian", "Czech": "Czech", "Britain": "English",
        "Austria": "German", "Greece": "Greek", "Netherlands": "Dutch", "Denmark": "Danish",
        "Hungary": "Hungarian", "Portugal": "Portuguese", "Sweden": "Swedish", "Egypt": "Arabic",
        "Turkey": "Turkish", "Poland": "Polish", "Korea": "Korean", "Thailand": "Thai",
        "Finland": "Finnish", "Norway": "Norwegian", "Israel": "Hebrew", "Ukraine": "Ukrainian",
        "Bulgaria": "Bulgarian", "Indonesia": "Indonesian", "Iran": "Persian"}
CAP2 = dict(CAPITAL, France="Paris")
DEPTH = {
    "France": {"lm": ["the Eiffel Tower", "the Louvre", "Notre-Dame Cathedral"],
               "per": ["Voltaire", "Georges Bizet", "Claude Monet", "Edith Piaf"],
               "work": ["Candide", "the opera Carmen", "the painting Impression, Sunrise"]},
    "Italy": {"lm": ["the Colosseum", "the Trevi Fountain", "the Pantheon"],
              "per": ["Julius Caesar", "Sophia Loren", "Ennio Morricone"],
              "work": ["the Commentaries on the Gallic War"]},
    "Germany": {"lm": ["the Brandenburg Gate", "the Reichstag building"],
                "per": ["Marlene Dietrich", "Alexander von Humboldt", "Walter Benjamin"],
                "work": ["The Arcades Project"]},
    "Spain": {"lm": ["the Prado Museum", "Retiro Park"],
              "per": ["Lope de Vega", "Pedro Calderon de la Barca"],
              "work": ["the play Life Is a Dream", "the play Fuenteovejuna"]},
    "Japan": {"lm": ["the Senso-ji temple", "the Meiji Shrine", "Shibuya Crossing"],
              "per": ["Hokusai", "Yukio Mishima", "Natsume Soseki"],
              "work": ["The Great Wave off Kanagawa", "the novel The Sound of Waves",
                       "the novel Kokoro"]},
    "Russia": {"lm": ["the Kremlin", "Red Square", "Saint Basil's Cathedral"],
               "per": ["Alexander Pushkin", "Fyodor Dostoevsky", "Mikhail Lermontov"],
               "work": ["Eugene Onegin", "the novel Crime and Punishment",
                        "the novel A Hero of Our Time"]},
    "Czech": {"lm": ["the Charles Bridge", "the Astronomical Clock of the Old Town"],
              "per": ["Franz Kafka", "Rainer Maria Rilke"],
              "work": ["The Metamorphosis", "the novel The Trial", "the Duino Elegies"]},
    "Britain": {"lm": ["Big Ben", "Tower Bridge", "Buckingham Palace"],
                "per": ["Virginia Woolf", "Charlie Chaplin", "John Keats"],
                "work": ["the novel Mrs Dalloway", "the film The Great Dictator",
                         "the poem Ode to a Nightingale"]},
    "Austria": {"lm": ["Schonbrunn Palace", "St. Stephen's Cathedral", "the Belvedere palace"],
                "per": ["Franz Schubert", "Johann Strauss II", "Ludwig Wittgenstein"],
                "work": ["the Unfinished Symphony", "The Blue Danube waltz",
                         "the Tractatus Logico-Philosophicus"]},
    "Greece": {"lm": ["the Acropolis", "the Parthenon"],
               "per": ["Plato", "Pericles"], "work": ["The Republic"]},
    "Netherlands": {"lm": ["the Rijksmuseum", "the Anne Frank House"],
                    "per": ["Baruch Spinoza"], "work": []},
    "Denmark": {"lm": ["Tivoli Gardens", "the Little Mermaid statue"],
                "per": ["Soren Kierkegaard", "Niels Bohr"],
                "work": ["the book Fear and Trembling", "the book Either/Or"]},
    "Hungary": {"lm": ["Fisherman's Bastion", "the Szechenyi thermal bath"],
                "per": ["Harry Houdini", "Erno Rubik"], "work": ["the Rubik's Cube"]},
    "Portugal": {"lm": ["the Belem Tower", "the Jeronimos Monastery"],
                 "per": ["Fernando Pessoa"], "work": ["The Book of Disquiet"]},
    "Sweden": {"lm": ["the Vasa Museum"], "per": ["Greta Garbo", "August Strindberg"],
               "work": ["the play Miss Julie"]},
    "Egypt": {"lm": ["the Khan el-Khalili bazaar"], "per": ["Naguib Mahfouz"],
              "work": ["the novel Palace Walk"]},
    "Turkey": {"lm": ["Anitkabir"], "per": [], "work": []},
    "Poland": {"lm": ["the Palace of Culture and Science"], "per": ["Marie Curie"], "work": []},
    "Korea": {"lm": ["Gyeongbokgung Palace"], "per": [], "work": []},
    "Thailand": {"lm": ["the Grand Palace", "Wat Arun"], "per": [], "work": []},
    "Finland": {"lm": ["Suomenlinna fortress"], "per": [], "work": []},
    "Norway": {"lm": ["the Vigeland sculpture park"], "per": [], "work": []},
}
PREFIX_DEPTH = ("The official language of the country whose capital is Jakarta is Indonesian.\n"
                "The official language of the country whose capital is home to the Azadi Tower "
                "is Persian.\n")
DEPTH_T = {
    "d1": "The official language of {} is",
    "d2": "The official language of the country whose capital is {} is",
    "d3-lm": "The official language of the country whose capital is home to {} is",
    "d3-per": "The official language of the country whose capital is the birthplace of {} is",
    "d4-work": ("The official language of the country whose capital is the birthplace of the "
                "creator of {} is"),
}
HOP1_T = {"d2": "The country whose capital is {} is",
          "d3-lm": "The country whose capital is home to {} is",
          "d3-per": "The country whose capital is the birthplace of {} is",
          "d4-work": "The country whose capital is the birthplace of the creator of {} is"}
PREFIX_HOP1_D = ("The country whose capital is Jakarta is Indonesia.\n"
                 "The country whose capital is home to the Azadi Tower is Iran.\n")
PREFIX_HOP2_D = "The official language of Indonesia is Indonesian.\nThe official language of Iran is Persian.\n"

FILLER = ("The weather report for the coming week says mornings will be cool and clear, with "
          "light winds from the west and a chance of rain on the weekend. Farmers in the valley "
          "expect a good harvest of apples and pears this year, and the local market will open "
          "an hour earlier on Saturdays. The library has added new reading rooms, longer opening "
          "hours and a small cafe near the entrance. Students are reminded that the final "
          "examinations start next month and that study groups meet every Tuesday evening in the "
          "main hall. The river trail was repaired after the spring floods, and cyclists can now "
          "ride the full loop again. A new bakery on the corner sells bread, pastries and soup, "
          "and it is already busy at lunchtime. The town council will discuss parking, street "
          "lights and the summer festival at its next meeting, which is open to the public. ")
OBSCURE = {
    "currency": [("dram", "Armenia"), ("lari", "Georgia"), ("tenge", "Kazakhstan"),
                 ("kip", "Laos"), ("kyat", "Myanmar"), ("taka", "Bangladesh"),
                 ("cedi", "Ghana"), ("birr", "Ethiopia"), ("kwacha", "Zambia"),
                 ("pula", "Botswana"), ("guarani", "Paraguay"), ("lempira", "Honduras"),
                 ("quetzal", "Guatemala"), ("colon", "Costa Rica"), ("denar", "North Macedonia"),
                 ("manat", "Azerbaijan"), ("somoni", "Tajikistan"), ("ouguiya", "Mauritania"),
                 ("ariary", "Madagascar"), ("tugrik", "Mongolia")],
    "city": [("Sapporo", "Japan"), ("Daegu", "Korea"), ("Wuhan", "China"), ("Kazan", "Russia"),
             ("Dortmund", "Germany"), ("Bologna", "Italy"), ("Valencia", "Spain"),
             ("Braga", "Portugal"), ("Utrecht", "Netherlands"), ("Malmo", "Sweden"),
             ("Wroclaw", "Poland"), ("Trondheim", "Norway"), ("Antalya", "Turkey"),
             ("Luxor", "Egypt"), ("Calgary", "Canada"), ("Perth", "Australia"),
             ("Leeds", "Britain"), ("Patras", "Greece"), ("Graz", "Austria"), ("Ghent", "Belgium"),
             ("Odense", "Denmark"), ("Turku", "Finland"), ("Galway", "Ireland"),
             ("Eilat", "Israel"), ("Faisalabad", "Pakistan"), ("Ostrava", "Czech"),
             ("Lviv", "Ukraine"), ("Tabriz", "Iran"), ("Mosul", "Iraq"), ("Homs", "Syria"),
             ("Arequipa", "Peru"), ("Kisumu", "Kenya"), ("Davao", "Philippines"),
             ("Medan", "Indonesia"), ("Phuket", "Thailand"), ("Varna", "Bulgaria"),
             ("Szeged", "Hungary")],
}
CAP_EXTRA = {"Armenia": "Yerevan", "Georgia": "Tbilisi", "Kazakhstan": "Astana", "Laos": "Vientiane",
             "Myanmar": "Naypyidaw", "Bangladesh": "Dhaka", "Ghana": "Accra", "Ethiopia": "Addis",
             "Zambia": "Lusaka", "Botswana": "Gaborone", "Paraguay": "Asuncion",
             "Honduras": "Tegucigalpa", "Guatemala": "Guatemala", "Costa Rica": "San",
             "North Macedonia": "Skopje", "Azerbaijan": "Baku", "Tajikistan": "Dushanbe",
             "Mauritania": "Nouakchott", "Madagascar": "Antananarivo", "Mongolia": "Ulaanbaatar"}
PARA = {"currency": "Question: in which city does the government of the nation that uses the {} "
                    "as money sit? Answer:",
        "city": "Question: in which city does the government of the nation where {} lies sit? "
                "Answer:"}
DIST = {"currency": "My friend collects coins such as the {}. ",
        "city": "My friend just came back from a trip to {}. "}
LANG2 = {"currency": "The official language of the country whose currency is the {} is",
         "city": "The official language of the country containing the city of {} is"}
PREFIX_LANG2 = ("The official language of the country whose currency is the rupiah is Indonesian.\n"
                "The official language of the country containing the city of Shiraz is Persian.\n")


def main():
    enc = tokenizer(sys.argv[1])

    def one(word):
        ids = enc(" " + word)
        return ids[0] if len(ids) == 1 else None

    def first(word):
        return enc(" " + word)[0]

    items, skipped = [], []

    def add(family, depth, descriptor, bridge, prefix, template, answer_word, hop1, hop2,
            accept_extra=(), capital=None):
        head = template.split("{}")[0]
        prompt = prefix + template.format(descriptor)
        ids = enc(prompt)
        before = enc(prefix + head.rstrip(" "))
        if ids[:len(before)] != before:
            skipped.append((family, descriptor, "bad split"))
            return
        after = template.split("{}")[1]
        tail = len(enc(after)) if after.strip() else 0
        # descriptor rows: from the head to the last descriptor token
        full_desc = enc(prefix + head.rstrip(" ") + " " + descriptor) if head.strip() else \
            enc(prefix + descriptor)
        if ids[:len(full_desc)] != full_desc:
            skipped.append((family, descriptor, "descriptor split"))
            return
        span = list(range(len(before), len(full_desc)))
        answer = one(answer_word) if answer_word not in CAP_EXTRA.values() else first(answer_word)
        if answer is None:
            answer = first(answer_word)
        low = prompt.lower()
        names = [bridge, COUNTRY_NAME.get(bridge, bridge)]
        if family.startswith("d") and family != "d1":
            names.append(CAP2.get(bridge, "#"))
            if family in ("d3-lm", "d3-per", "d4-work"):
                pass
        bad = [w for w in names if w and w.lower() in low.replace(prefix.lower(), "")]
        if family == "d2":
            bad = [w for w in bad if w != CAP2.get(bridge)]
        if family != "d1" and bad:
            skipped.append((family, descriptor, f"names {bad}"))
            return
        country_ids = [i for i in [one(bridge)] + [one(a) for a in COUNTRY_ALIASES.get(bridge, [])]
                       if i]
        items.append({
            "id": f"{family}:{descriptor}", "family": family, "depth": depth,
            "descriptor": descriptor, "bridge": bridge, "capital": capital,
            "answer": answer, "answer_word": answer_word,
            "answer_accept": [answer] + [x for x in accept_extra if x],
            "country_token": one(bridge), "country_accept": country_ids,
            "capital_token": first(capital) if capital else None,
            "prompt": prompt, "ids": ids, "n": len(ids), "span": span, "row": span[-1],
            "tail": len(ids) - 1 - span[-1],
            "hop1_ids": enc(hop1) if hop1 else enc(prompt), "hop2_ids": enc(hop2),
        })

    # ---- depth panel
    for bridge, spec in DEPTH.items():
        lang, cap = LANG[bridge], CAP2[bridge]
        hop2 = PREFIX_HOP2_D + f"The official language of {COUNTRY_NAME.get(bridge, bridge)} is"
        name = COUNTRY_NAME.get(bridge, bridge)
        add("d1", 1, name[0].upper() + name[1:] if False else name, bridge, PREFIX_DEPTH,
            DEPTH_T["d1"], lang, None, hop2, capital=cap)
        add("d2", 2, cap, bridge, PREFIX_DEPTH, DEPTH_T["d2"], lang,
            PREFIX_HOP1_D + HOP1_T["d2"].format(cap), hop2, capital=cap)
        for fam, key, depth in (("d3-lm", "lm", 3), ("d3-per", "per", 3), ("d4-work", "work", 4)):
            for x in spec[key]:
                add(fam, depth, x, bridge, PREFIX_DEPTH, DEPTH_T[fam], lang,
                    PREFIX_HOP1_D + HOP1_T[fam].format(x), hop2, capital=cap)
    # ---- difficulty panel (2-hop, capital answer unless lang-)
    for fam in ("currency", "city"):
        spec = FAMILIES[fam]
        base = spec["items"]
        for k, (x, bridge) in enumerate(base):
            capw = CAPITAL[bridge]
            hop1 = PREFIX_HOP1 + spec["hop1"].format(x)
            two = spec["two"]
            add(f"para-{fam}", 2, x, bridge, PREFIX_TWO, PARA[fam], capw,
                hop1, PREFIX_TWO + f"The capital of {COUNTRY_NAME.get(bridge, bridge)} is")
            y = next(b[0] for b in base[k + 1:] + base[:k] if b[1] != bridge)
            add(f"dist-{fam}", 2, x, bridge, PREFIX_TWO + DIST[fam].format(y), two, capw,
                hop1, PREFIX_TWO + f"The capital of {COUNTRY_NAME.get(bridge, bridge)} is")
            add(f"long-{fam}", 2, x, bridge, PREFIX_TWO + FILLER, two, capw, hop1,
                PREFIX_TWO + f"The capital of {COUNTRY_NAME.get(bridge, bridge)} is")
            if bridge in LANG and bridge not in ("Indonesia", "Iran"):
                add(f"lang-{fam}", 2, x, bridge, PREFIX_LANG2, LANG2[fam], LANG[bridge], hop1,
                    PREFIX_HOP2_D + f"The official language of {COUNTRY_NAME.get(bridge, bridge)} is")
        for x, bridge in OBSCURE[fam]:
            capw = CAPITAL.get(bridge) or CAP_EXTRA.get(bridge)
            if capw is None:
                continue
            add(f"obsc-{fam}", 2, x, bridge, PREFIX_TWO, spec["two"], capw,
                PREFIX_HOP1 + spec["hop1"].format(x),
                PREFIX_TWO + f"The capital of {COUNTRY_NAME.get(bridge, bridge)} is")
    # first-token answers must not collide inside a family (donors need distinct answers anyway)
    json.dump({"prefix_depth": PREFIX_DEPTH, "filler": FILLER, "items": items,
               "skipped": skipped}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
