"""DAS training candidates (tokenizer only, no model): the Run 1a two-hop templates and few-shot
prefix (twohop-20261113/gen_panel.py) over countries DISJOINT from every country in the exp7 test
pairs (recipients and donors) and from the prefix countries (Switzerland, France).

python gen_train.py TOKENIZER_JSON > train_candidates.json

Same item format as gen_panel.py (span = descriptor rows); answer = the capital's token or its
first piece (``first``: >= 4 characters, unique among the table's capitals). Admission
(native argmax = capital) and the carrier row (frozen R_argmax rule) happen on the GPU (das.py admit).
"""
import json
import sys

from tokenizers import Tokenizer

TEST_COUNTRIES = {"Japan", "Korea", "Russia", "China", "Thailand", "Poland", "Hungary", "Sweden",
                  "Indonesia", "Israel", "Czech", "Britain", "Iran", "Afghanistan", "America",
                  "Denmark", "Spain", "Italy", "Austria", "Greece", "Finland", "Norway", "Ireland",
                  "Netherlands", "Egypt", "Australia", "Germany", "Portugal"}
PREFIX_COUNTRIES = {"Switzerland", "France"}
CAPITAL = {
    "Turkey": "Ankara", "Ukraine": "Kiev", "Bulgaria": "Sofia", "Peru": "Lima", "Chile": "Santiago",
    "Canada": "Ottawa", "Belgium": "Brussels", "Pakistan": "Islamabad", "Kenya": "Nairobi",
    "Iraq": "Baghdad", "Syria": "Damascus", "Philippines": "Manila", "Cuba": "Havana",
    "Lebanon": "Beirut", "Vietnam": "Hanoi", "Venezuela": "Caracas", "Morocco": "Rabat",
    "Saudi Arabia": "Riyadh", "Romania": "Bucharest", "Serbia": "Belgrade", "Croatia": "Zagreb",
    "Iceland": "Reykjavik", "Nepal": "Kathmandu", "Bangladesh": "Dhaka", "Jordan": "Amman",
    "Ghana": "Accra", "Georgia": "Tbilisi", "Armenia": "Yerevan", "Latvia": "Riga",
    "Lithuania": "Vilnius", "Estonia": "Tallinn", "Slovakia": "Bratislava", "Nigeria": "Abuja",
    "Kazakhstan": "Astana", "Tunisia": "Tunis", "Libya": "Tripoli", "Ecuador": "Quito",
    "Uruguay": "Montevideo", "Albania": "Tirana", "Azerbaijan": "Baku", "Uzbekistan": "Tashkent",
    "Brazil": "Brasilia", "Colombia": "Bogota", "Cambodia": "Phnom", "Ethiopia": "Addis",
    "Zambia": "Lusaka", "Angola": "Luanda", "Mozambique": "Maputo", "Paraguay": "Asuncion",
    "Slovenia": "Ljubljana", "Mongolia": "Ulaanbaatar", "Laos": "Vientiane", "Jamaica": "Kingston",
    "Argentina": "Buenos Aires", "Malaysia": "Kuala Lumpur", "Belarus": "Minsk", "Uganda": "Kampala",
    "Zimbabwe": "Harare", "Senegal": "Dakar", "Sudan": "Khartoum", "Somalia": "Mogadishu",
    "Cyprus": "Nicosia", "Sri Lanka": "Colombo", "Qatar": "Doha",
}
FAMILIES = {
    "currency": ("The capital of the country whose currency is the {} is", [
        ("lira", "Turkey"), ("hryvnia", "Ukraine"), ("lev", "Bulgaria"), ("sol", "Peru"),
        ("shilling", "Kenya"), ("dong", "Vietnam"), ("leu", "Romania"), ("kuna", "Croatia"),
        ("dinar", "Serbia"), ("tenge", "Kazakhstan"), ("lari", "Georgia"), ("dram", "Armenia"),
        ("cedi", "Ghana"), ("taka", "Bangladesh"), ("naira", "Nigeria"), ("riyal", "Saudi Arabia"),
        ("dirham", "Morocco"), ("bolivar", "Venezuela"), ("real", "Brazil"), ("lek", "Albania"),
        ("manat", "Azerbaijan"), ("kwacha", "Zambia"), ("kwanza", "Angola"), ("metical", "Mozambique"),
        ("guarani", "Paraguay"), ("riel", "Cambodia"), ("kip", "Laos"), ("tugrik", "Mongolia"),
        ("birr", "Ethiopia"), ("peso", "Philippines")]),
    "person": ("The capital of the home country of {} is", [
        ("Ataturk", "Turkey"), ("Orhan Pamuk", "Turkey"), ("Taras Shevchenko", "Ukraine"),
        ("Pablo Neruda", "Chile"), ("Fidel Castro", "Cuba"), ("Kahlil Gibran", "Lebanon"),
        ("Mario Vargas Llosa", "Peru"), ("Jose Rizal", "Philippines"), ("Benazir Bhutto", "Pakistan"),
        ("Ho Chi Minh", "Vietnam"), ("Rene Magritte", "Belgium"), ("Jacques Brel", "Belgium"),
        ("Saddam Hussein", "Iraq"), ("Chinua Achebe", "Nigeria"), ("Kofi Annan", "Ghana"),
        ("Constantin Brancusi", "Romania"), ("Novak Djokovic", "Serbia"), ("Haile Selassie", "Ethiopia"),
        ("Gabriel Garcia Marquez", "Colombia"), ("Simon Bolivar", "Venezuela"),
        ("Margaret Atwood", "Canada"), ("Wayne Gretzky", "Canada"), ("Bob Marley", "Jamaica"),
        ("Hristo Stoichkov", "Bulgaria"), ("Luka Modric", "Croatia"), ("Bjork", "Iceland"),
        ("Hafez al-Assad", "Syria"), ("Muhammad Ali Jinnah", "Pakistan"), ("Pele", "Brazil"),
        ("Jorge Luis Borges", "Argentina"), ("Diego Maradona", "Argentina"), ("Idi Amin", "Uganda"),
        ("Robert Mugabe", "Zimbabwe"), ("Leopold Senghor", "Senegal"), ("Mahathir Mohamad", "Malaysia")]),
    "landmark": ("The capital of the country that is home to {} is", [
        ("Hagia Sophia", "Turkey"), ("Cappadocia", "Turkey"), ("Machu Picchu", "Peru"),
        ("Niagara Falls", "Canada"), ("Babylon", "Iraq"), ("Palmyra", "Syria"), ("Petra", "Jordan"),
        ("Ha Long Bay", "Vietnam"), ("the Atomium", "Belgium"), ("Bran Castle", "Romania"),
        ("the Plitvice Lakes", "Croatia"), ("Easter Island", "Chile"), ("Angkor Wat", "Cambodia"),
        ("Chernobyl", "Ukraine"), ("the Rila Monastery", "Bulgaria"), ("the Blue Lagoon", "Iceland"),
        ("the Hassan II Mosque", "Morocco"), ("the Masai Mara", "Kenya"),
        ("the Banaue Rice Terraces", "Philippines"), ("Christ the Redeemer", "Brazil"),
        ("the Galapagos Islands", "Ecuador"), ("Baalbek", "Lebanon"), ("Carthage", "Tunisia"),
        ("Leptis Magna", "Libya"), ("Samarkand", "Uzbekistan"), ("the Petronas Towers", "Malaysia"),
        ("Sigiriya", "Sri Lanka"), ("the Perito Moreno Glacier", "Argentina")]),
    "city": ("The capital of the country containing the city of {} is", [
        ("Istanbul", "Turkey"), ("Izmir", "Turkey"), ("Antalya", "Turkey"), ("Odessa", "Ukraine"),
        ("Kharkiv", "Ukraine"), ("Lviv", "Ukraine"), ("Plovdiv", "Bulgaria"), ("Varna", "Bulgaria"),
        ("Cusco", "Peru"), ("Arequipa", "Peru"), ("Valparaiso", "Chile"), ("Toronto", "Canada"),
        ("Vancouver", "Canada"), ("Montreal", "Canada"), ("Antwerp", "Belgium"), ("Ghent", "Belgium"),
        ("Karachi", "Pakistan"), ("Lahore", "Pakistan"), ("Mombasa", "Kenya"), ("Basra", "Iraq"),
        ("Mosul", "Iraq"), ("Aleppo", "Syria"), ("Cebu", "Philippines"), ("Davao", "Philippines"),
        ("Da Nang", "Vietnam"), ("Casablanca", "Morocco"), ("Marrakesh", "Morocco"),
        ("Jeddah", "Saudi Arabia"), ("Lagos", "Nigeria"), ("Kano", "Nigeria"), ("Cluj", "Romania"),
        ("Timisoara", "Romania"), ("Novi Sad", "Serbia"), ("Dubrovnik", "Croatia"),
        ("Almaty", "Kazakhstan"), ("Batumi", "Georgia"), ("Kumasi", "Ghana"),
        ("Chittagong", "Bangladesh"), ("Pokhara", "Nepal"), ("Maracaibo", "Venezuela"),
        ("Medellin", "Colombia"), ("Rio de Janeiro", "Brazil"), ("Sao Paulo", "Brazil"),
        ("Guayaquil", "Ecuador"), ("Kosice", "Slovakia"), ("Tartu", "Estonia"),
        ("Kaunas", "Lithuania"), ("Gyumri", "Armenia"), ("Sfax", "Tunisia"), ("Benghazi", "Libya"),
        ("Aqaba", "Jordan"), ("Maribor", "Slovenia"), ("Durres", "Albania"), ("Ganja", "Azerbaijan"),
        ("Cordoba", "Argentina"), ("Rosario", "Argentina"), ("Mendoza", "Argentina"),
        ("Penang", "Malaysia"), ("Gomel", "Belarus"), ("Entebbe", "Uganda"), ("Bulawayo", "Zimbabwe"),
        ("Omdurman", "Sudan"), ("Hargeisa", "Somalia"), ("Limassol", "Cyprus"), ("Larnaca", "Cyprus"),
        ("Kandy", "Sri Lanka"), ("Galle", "Sri Lanka")]),
}
PREFIX_TWO = ("The capital of the country whose currency is the Swiss franc is Bern.\n"
              "The capital of the country containing the city of Lyon is Paris.\n")


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    enc = lambda text: tok.encode(text, add_special_tokens=False).ids  # noqa: E731

    def one(word):
        ids = enc(" " + word)
        return ids[0] if len(ids) == 1 else None

    def first(word):
        """The answer token: the capital's single token, or its first piece when that piece is at
        least 4 characters (with the space) and no other capital of the table starts with it."""
        ids = enc(" " + word)
        if len(ids) == 1:
            return ids[0]
        piece = tok.decode([ids[0]])
        starts = [c for c in CAPITAL.values() if enc(" " + c)[0] == ids[0]]
        return ids[0] if len(piece) >= 4 and len(starts) == 1 else None

    assert not (set(CAPITAL) & (TEST_COUNTRIES | PREFIX_COUNTRIES))
    items, skipped = [], []
    for family, (two, pairs) in FAMILIES.items():
        head = two.split("{}")[0]
        for descriptor, bridge in pairs:
            answer = first(CAPITAL[bridge])
            prompt = PREFIX_TWO + two.format(descriptor)
            ids = enc(prompt)
            before = enc(PREFIX_TWO + head.rstrip(" "))
            if answer is None or ids[:len(before)] != before:
                skipped.append([family, descriptor, "answer not one token or bad split"])
                continue
            span = list(range(len(before), len(ids) - 1))
            if not span or bridge.lower() in prompt.lower():
                skipped.append([family, descriptor, "empty span or bridge named"])
                continue
            items.append({"id": f"{family}:{descriptor}", "family": family, "descriptor": descriptor,
                          "bridge": bridge, "capital": CAPITAL[bridge], "answer": answer,
                          "country_token": one(bridge), "ids": ids, "span": span})
    json.dump({"prefix_two": PREFIX_TWO, "test_countries": sorted(TEST_COUNTRIES),
               "capital_tokens": {c: first(v) for c, v in CAPITAL.items()},
               "items": items, "skipped": skipped}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
