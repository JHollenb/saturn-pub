"""Candidate two-hop items for the twohop-20261113 experiment (tokenizer only, no model).

python gen_panel.py TOKENIZER_JSON_OR_MODEL_DIR > candidates.json

Four descriptor families (currency, person, landmark, city); hop 2 is always "capital of".
Every two-hop prompt ends "... {descriptor} is", so the descriptor's last token is row n-2
and the answer row (the pending row) is n-1. The bridge country never appears in a prompt.
Answers are single tokens. Admission (correct native argmax) happens on the GPU in job 1.
"""
import json
import sys
from pathlib import Path

CAPITAL = {
    "Japan": "Tokyo", "Korea": "Seoul", "Russia": "Moscow", "China": "Beijing",
    "Thailand": "Bangkok", "Turkey": "Ankara", "Poland": "Warsaw", "Hungary": "Budapest",
    "Sweden": "Stockholm", "Indonesia": "Jakarta", "Israel": "Jerusalem", "Ukraine": "Kiev",
    "Bulgaria": "Sofia", "Czech": "Prague", "Iran": "Tehran", "Afghanistan": "Kabul",
    "Peru": "Lima", "Britain": "London", "Germany": "Berlin", "Spain": "Madrid", "Italy": "Rome",
    "Austria": "Vienna", "Finland": "Helsinki", "Norway": "Oslo", "Greece": "Athens",
    "Denmark": "Copenhagen", "Netherlands": "Amsterdam", "Ireland": "Dublin", "Chile": "Santiago",
    "Australia": "Canberra", "Canada": "Ottawa", "Egypt": "Cairo", "Portugal": "Lisbon",
    "Belgium": "Brussels", "Pakistan": "Islamabad", "Kenya": "Nairobi", "Iraq": "Baghdad",
    "Syria": "Damascus", "Philippines": "Manila", "Cuba": "Havana", "Lebanon": "Beirut",
    "America": "Washington",
}
# accepted first tokens for the hop-1 answer (country) beyond " {key}"
COUNTRY_ALIASES = {"Korea": ["South"], "Britain": ["England", "the", "United", "UK"],
                   "America": ["the", "United", "USA"], "Czech": ["Czechia"],
                   "Netherlands": ["the", "Holland"], "Philippines": ["the"]}
# hop-2 prompt names
COUNTRY_NAME = {"Korea": "South Korea", "Britain": "the United Kingdom",
                "America": "the United States", "Czech": "the Czech Republic",
                "Netherlands": "the Netherlands", "Philippines": "the Philippines"}

FAMILIES = {
    "currency": {
        "two": "The capital of the country whose currency is the {} is",
        "hop1": "The country whose currency is the {} is",
        "items": [("yen", "Japan"), ("won", "Korea"), ("ruble", "Russia"), ("rouble", "Russia"),
                  ("yuan", "China"), ("renminbi", "China"), ("baht", "Thailand"),
                  ("lira", "Turkey"), ("zloty", "Poland"), ("forint", "Hungary"),
                  ("krona", "Sweden"), ("rupiah", "Indonesia"), ("shekel", "Israel"),
                  ("hryvnia", "Ukraine"), ("lev", "Bulgaria"), ("koruna", "Czech"),
                  ("sol", "Peru"), ("pound sterling", "Britain"), ("sterling", "Britain"),
                  ("rial", "Iran"), ("afghani", "Afghanistan"), ("dollar", "America"),
                  ("peso", "Philippines"), ("shilling", "Kenya")],
    },
    "person": {
        "two": "The capital of the home country of {} is",
        "hop1": "The home country of {} is",
        "items": [("Hokusai", "Japan"), ("Akira Kurosawa", "Japan"), ("Haruki Murakami", "Japan"),
                  ("Leo Tolstoy", "Russia"), ("Fyodor Dostoevsky", "Russia"),
                  ("Alexander Pushkin", "Russia"), ("Confucius", "China"),
                  ("Mao Zedong", "China"), ("Goethe", "Germany"), ("Ludwig van Beethoven", "Germany"),
                  ("Johann Sebastian Bach", "Germany"), ("Pablo Picasso", "Spain"),
                  ("Miguel de Cervantes", "Spain"), ("Dante Alighieri", "Italy"),
                  ("Giuseppe Verdi", "Italy"), ("Michelangelo", "Italy"), ("Mozart", "Austria"),
                  ("Frederic Chopin", "Poland"), ("Nicolaus Copernicus", "Poland"),
                  ("Jean Sibelius", "Finland"), ("Henrik Ibsen", "Norway"),
                  ("Edvard Munch", "Norway"), ("Socrates", "Greece"), ("Plato", "Greece"),
                  ("Soren Kierkegaard", "Denmark"), ("Hans Christian Andersen", "Denmark"),
                  ("Rembrandt", "Netherlands"), ("Johannes Vermeer", "Netherlands"),
                  ("James Joyce", "Ireland"), ("W. B. Yeats", "Ireland"), ("Franz Liszt", "Hungary"),
                  ("August Strindberg", "Sweden"), ("Ingmar Bergman", "Sweden"),
                  ("Antonin Dvorak", "Czech"), ("Pablo Neruda", "Chile"),
                  ("William Shakespeare", "Britain"), ("Charles Dickens", "Britain"),
                  ("Ataturk", "Turkey"), ("Fidel Castro", "Cuba"), ("Kahlil Gibran", "Lebanon")],
    },
    "landmark": {
        "two": "The capital of the country that is home to {} is",
        "hop1": "The country that is home to {} is",
        "items": [("Mount Fuji", "Japan"), ("Lake Baikal", "Russia"), ("the Great Wall", "China"),
                  ("the Forbidden City", "China"), ("the Colosseum", "Italy"),
                  ("the Leaning Tower of Pisa", "Italy"), ("the Parthenon", "Greece"),
                  ("the Acropolis", "Greece"), ("the pyramids of Giza", "Egypt"),
                  ("the Sphinx", "Egypt"), ("Machu Picchu", "Peru"), ("Stonehenge", "Britain"),
                  ("Neuschwanstein Castle", "Germany"), ("the Alhambra", "Spain"),
                  ("the Sagrada Familia", "Spain"), ("Uluru", "Australia"),
                  ("the Sydney Opera House", "Australia"), ("Hagia Sophia", "Turkey"),
                  ("Borobudur", "Indonesia"), ("Persepolis", "Iran"), ("Babylon", "Iraq"),
                  ("Palmyra", "Syria"), ("the Cliffs of Moher", "Ireland"),
                  ("Niagara Falls", "Canada"), ("Mount Rushmore", "America"),
                  ("the Statue of Liberty", "America"), ("Wawel Castle", "Poland"),
                  ("the Geirangerfjord", "Norway"), ("Masada", "Israel"),
                  ("Schonbrunn Palace", "Austria"), ("Kinkaku-ji", "Japan"),
                  ("the Hermitage Museum", "Russia"), ("the Rijksmuseum", "Netherlands")],
    },
    "city": {
        "two": "The capital of the country containing the city of {} is",
        "hop1": "The country containing the city of {} is",
        "items": [("Osaka", "Japan"), ("Kyoto", "Japan"), ("Busan", "Korea"),
                  ("Shanghai", "China"), ("Shenzhen", "China"), ("Novosibirsk", "Russia"),
                  ("Saint Petersburg", "Russia"), ("Munich", "Germany"), ("Hamburg", "Germany"),
                  ("Milan", "Italy"), ("Naples", "Italy"), ("Barcelona", "Spain"),
                  ("Seville", "Spain"), ("Porto", "Portugal"), ("Rotterdam", "Netherlands"),
                  ("Gothenburg", "Sweden"), ("Krakow", "Poland"), ("Gdansk", "Poland"),
                  ("Bergen", "Norway"), ("Istanbul", "Turkey"), ("Izmir", "Turkey"),
                  ("Alexandria", "Egypt"), ("Toronto", "Canada"), ("Vancouver", "Canada"),
                  ("Sydney", "Australia"), ("Melbourne", "Australia"), ("Chicago", "America"),
                  ("Manchester", "Britain"), ("Liverpool", "Britain"), ("Thessaloniki", "Greece"),
                  ("Salzburg", "Austria"), ("Antwerp", "Belgium"), ("Aarhus", "Denmark"),
                  ("Tampere", "Finland"), ("Cork", "Ireland"), ("Haifa", "Israel"),
                  ("Karachi", "Pakistan"), ("Lahore", "Pakistan"), ("Brno", "Czech"),
                  ("Odessa", "Ukraine"), ("Kharkiv", "Ukraine"), ("Isfahan", "Iran"),
                  ("Basra", "Iraq"), ("Aleppo", "Syria"), ("Valparaiso", "Chile"),
                  ("Cusco", "Peru"), ("Mombasa", "Kenya"), ("Cebu", "Philippines"),
                  ("Surabaya", "Indonesia"), ("Chiang Mai", "Thailand"), ("Plovdiv", "Bulgaria"),
                  ("Debrecen", "Hungary")],
    },
}
# Few-shot prefixes: their bridges (Switzerland, France) are excluded from the test items.
PREFIX_TWO = ("The capital of the country whose currency is the Swiss franc is Bern.\n"
              "The capital of the country containing the city of Lyon is Paris.\n")
PREFIX_HOP1 = ("The country whose currency is the Swiss franc is Switzerland.\n"
               "The country containing the city of Lyon is France.\n")
PREFIX_HOP2 = "The capital of Switzerland is Bern.\nThe capital of France is Paris.\n"


def tokenizer(path):
    path = Path(path)
    if path.is_dir():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        return lambda text: tok.encode(text, add_special_tokens=False)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    return lambda text: tok.encode(text, add_special_tokens=False).ids


def main():
    enc = tokenizer(sys.argv[1])

    def one(word):
        ids = enc(" " + word)
        return ids[0] if len(ids) == 1 else None

    items, skipped = [], []
    for family, spec in FAMILIES.items():
        head = spec["two"].split("{}")[0]
        for descriptor, bridge in spec["items"]:
            capital = CAPITAL[bridge]
            answer = one(capital)
            prompt = PREFIX_TWO + spec["two"].format(descriptor)
            ids = enc(prompt)
            before = enc(PREFIX_TWO + head.rstrip(" "))
            if answer is None or ids[:len(before)] != before:
                skipped.append((family, descriptor, "answer not one token or bad split"))
                continue
            span = list(range(len(before), len(ids) - 1))     # descriptor rows
            if not span:
                skipped.append((family, descriptor, "empty span"))
                continue
            low = prompt.lower()
            bad = [w for w in (bridge, COUNTRY_NAME.get(bridge, bridge)) if w.lower() in low]
            if bad:
                skipped.append((family, descriptor, f"bridge named: {bad}"))
                continue
            country_ids = [i for i in [one(bridge)] + [one(a) for a in
                                                     COUNTRY_ALIASES.get(bridge, [])] if i]
            items.append({
                "id": f"{family}:{descriptor}", "family": family, "descriptor": descriptor,
                "bridge": bridge, "capital": capital, "answer": answer,
                "country_token": one(bridge), "country_accept": country_ids,
                "prompt": prompt, "ids": ids, "n": len(ids), "span": span, "row": len(ids) - 2,
                "hop1_ids": enc(PREFIX_HOP1 + spec["hop1"].format(descriptor)),
                "hop2_ids": enc(PREFIX_HOP2 + f"The capital of {COUNTRY_NAME.get(bridge, bridge)} is"),
            })
    json.dump({"prefix_two": PREFIX_TWO, "prefix_hop1": PREFIX_HOP1, "prefix_hop2": PREFIX_HOP2,
               "families": {k: {"two": v["two"], "hop1": v["hop1"]} for k, v in FAMILIES.items()},
               "capital_tokens": {c: one(CAPITAL[c]) for c in CAPITAL},
               "items": items, "skipped": skipped}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
