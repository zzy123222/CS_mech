from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

DOMAINS = {
    "astronomy": [
        ("Mercury", "the planet closest to the Sun", "the largest planet in the solar system"),
        ("Venus", "the hottest planet in the solar system", "the planet with the most moons"),
        ("Mars", "known as the red planet", "known as the blue planet"),
        ("Jupiter", "the largest planet in the solar system", "the planet closest to the Sun"),
        ("Saturn", "famous for its prominent ring system", "famous for its dense atmosphere of carbon dioxide"),
        ("Neptune", "the farthest planet from the Sun", "the smallest planet in the solar system"),
        ("the Moon", "Earth's only natural satellite", "the largest satellite of Jupiter"),
        ("the Sun", "a yellow dwarf star at the center of the solar system", "a red giant star near the edge of the galaxy"),
        ("Io", "a volcanically active moon of Jupiter", "a frozen moon of Neptune"),
        ("the asteroid belt", "located between Mars and Jupiter", "located between Earth and Venus"),
        ("a light-year", "the distance light travels in one year", "the time it takes light to reach Mars"),
        ("comets", "icy bodies that develop tails near the Sun", "rocky bodies found only between Saturn and Uranus"),
        ("Uranus", "a planet that rotates on its side", "a planet with no moons"),
        ("Europa", "an icy moon of Jupiter with a subsurface ocean", "a desert moon of Mars"),
        ("the Milky Way", "the galaxy that contains our solar system", "the nearest star to the Sun"),
        ("Proxima Centauri", "the closest star to the Sun", "the brightest star in the night sky"),
    ],
    "biology": [
        ("mitochondria", "the organelles that produce most cellular energy", "the organelles that store genetic material"),
        ("chloroplasts", "the organelles responsible for photosynthesis", "the organelles that digest cellular waste"),
        ("DNA", "the molecule that carries genetic information", "the molecule that transports oxygen in blood"),
        ("hemoglobin", "the protein in red blood cells that carries oxygen", "the protein that fights bacterial infections"),
        ("photosynthesis", "the process plants use to convert light into chemical energy", "the process plants use to absorb water from soil"),
        ("the cell nucleus", "the structure that contains chromosomes", "the structure that produces ribosomes"),
        ("ribosomes", "the molecular machines that synthesize proteins", "the sacs that store water in plant cells"),
        ("enzymes", "proteins that catalyze biochemical reactions", "lipids that form cell membranes"),
        ("neurons", "cells that transmit electrical and chemical signals", "cells that store fat in the liver"),
        ("xylem", "the plant tissue that transports water upward", "the plant tissue that transports sugars"),
        ("bacteria", "single-celled organisms without a nucleus", "single-celled organisms that always photosynthesize"),
        ("fungi", "organisms that absorb nutrients from decaying matter", "organisms that produce their own food from sunlight"),
        ("the human liver", "the organ that detoxifies chemicals and produces bile", "the organ that filters urine"),
        ("insulin", "a hormone that lowers blood glucose", "a hormone that regulates sleep cycles"),
        ("chlorophyll", "the green pigment that absorbs light energy", "the red pigment that carries oxygen"),
        ("antibodies", "proteins that recognize and neutralize pathogens", "proteins that contract muscle fibers"),
    ],
    "geography": [
        ("the Nile", "one of the longest rivers in the world", "the deepest ocean trench on Earth"),
        ("Mount Everest", "the highest mountain above sea level", "the largest volcano in the solar system"),
        ("the Sahara", "the largest hot desert in the world", "the largest tropical rainforest on Earth"),
        ("the Amazon", "the largest rainforest on Earth", "the col desert of Central Asia"),
        ("Australia", "both a country and a continent", "a group of islands governed by New Zealand"),
        ("the Pacific Ocean", "the largest ocean on Earth", "the smallest ocean on Earth"),
        ("Russia", "the largest country by land area", "the most populous country in the world"),
        ("the Danube", "a major river flowing through Central and Eastern Europe", "a major river flowing through India"),
        ("Iceland", "an island nation with frequent geothermal activity", "a landlocked country in the Alps"),
        ("the Andes", "the longest continental mountain range", "a lowland plain in northern Canada"),
        ("Japan", "an archipelago of volcanic islands", "a peninsula attached to the Asian mainland"),
        ("the Mediterranean", "a sea between Europe, Africa, and Asia", "a freshwater lake in Central Africa"),
        ("Vatican City", "the smallest country in the world by area", "the most populous country in Europe"),
        ("the Great Barrier Reef", "the largest coral reef system on Earth", "a mountain range under the Pacific Ocean"),
        ("Chile", "one of the longest north-south countries in the world", "a landlocked country in South America"),
        ("the Caspian Sea", "the largest inland body of water on Earth", "a shallow sea between Italy and Greece"),
    ],
    "world history": [
        ("the printing press", "invented by Gutenberg in the fifteenth century", "invented by Newton in the seventeenth century"),
        ("the Roman Empire", "a Mediterranean empire that fell in 476 CE", "a northern European empire that fell in 1066"),
        ("the Industrial Revolution", "began in Britain in the late eighteenth century", "began in Japan in the early twentieth century"),
        ("the Magna Carta", "a charter limiting royal power signed in 1215", "a peace treaty ending the Hundred Years War"),
        ("the Renaissance", "a cultural revival that began in Italy", "a religious movement that began in Scandinavia"),
        ("Columbus", "reached the Americas in 1492", "reached Australia in 1492"),
        ("the French Revolution", "overthrew the monarchy in 1789", "established the monarchy in 1789"),
        ("the Silk Road", "an ancient trade network linking East Asia and Europe", "a railway built across Siberia"),
        ("ancient Egypt", "a civilization centered on the Nile River", "a civilization centered on the Tigris River"),
        ("World War I", "began in 1914", "began in 1939"),
        ("the Byzantine Empire", "centered on Constantinople", "centered on Alexandria"),
        ("the Maya", "a Mesoamerican civilization with advanced astronomy", "a civilization of the Andean highlands"),
        ("the Berlin Wall", "fell in 1989", "was built at the end of World War II"),
        ("the Ottoman Empire", "centered in Anatolia with Istanbul as its capital", "centered on the Italian peninsula"),
        ("the Apollo program", "landed humans on the Moon in 1969", "sent the first humans to Mars"),
        ("the Marshall Plan", "U.S. aid for rebuilding Western Europe after 1945", "a military alliance founded in 1919"),
    ],
    "chemistry": [
        ("water", "a molecule with the formula H2O", "a molecule with the formula CO2"),
        ("oxygen", "the element that supports combustion and respiration", "an inert gas used in balloons"),
        ("gold", "a metal with the symbol Au", "a metal with the symbol Ag"),
        ("table salt", "composed of sodium and chlorine", "composed of potassium and iodine"),
        ("helium", "a noble gas lighter than air", "a reactive metal that tarnishes quickly"),
        ("acids", "substances that donate protons in solution", "substances that always contain oxygen"),
        ("diamond", "a form of carbon with a crystal lattice", "a form of silicon with a crystal lattice"),
        ("iron", "a metal that rusts in the presence of water and oxygen", "a metal that dissolves in alcohol"),
        ("the periodic table", "organizes elements by atomic number", "organizes elements by their discovery date"),
        ("neon", "a noble gas used in bright signage", "an alkaline metal found in seawater"),
        ("carbon dioxide", "a greenhouse gas produced by respiration and combustion", "a noble gas used in deep-sea diving"),
        ("mercury", "a metal that is liquid at room temperature", "a metal that is gaseous at room temperature"),
        ("catalysts", "substances that speed up reactions without being consumed", "substances that stop all chemical reactions"),
        ("ozone", "a molecule of three oxygen atoms absorbing ultraviolet light", "a molecule of two nitrogen atoms"),
        ("hydrogen", "the lightest and most abundant element in the universe", "the heaviest element in the periodic table"),
        ("sodium", "a soft reactive metal that fizzes in water", "a dense metal that sinks in mercury"),
    ],
    "physics": [
        ("the speed of light", "about 300,000 kilometers per second in vacuum", "about 300 kilometers per hour in air"),
        ("gravity", "an attractive force between masses", "a repulsive force between charges"),
        ("energy", "conserved in a closed system according to physics", "created from nothing in any process"),
        ("photons", "massless particles carrying electromagnetic radiation", "heavy particles that make up atomic nuclei"),
        ("absolute zero", "the lowest possible temperature", "the temperature at which water freezes"),
        ("Newton's second law", "relates force to mass and acceleration", "relates energy to temperature and volume"),
        ("sound", "a mechanical wave that needs a medium", "an electromagnetic wave that travels in vacuum"),
        ("electrons", "negatively charged particles orbiting atomic nuclei", "positively charged particles in the nucleus"),
        ("magnets", "always have a north and a south pole", "can have two identical poles at their ends"),
        ("friction", "a force opposing relative motion between surfaces", "a force that increases with distance"),
        ("momentum", "the product of mass and velocity", "the product of force and time"),
        ("black holes", "regions where gravity prevents light from escaping", "empty regions between galaxies"),
        ("semiconductors", "materials with conductivity between conductors and insulators", "materials that conduct only at absolute zero"),
        ("kinetic energy", "the energy of motion", "the energy stored in chemical bonds"),
        ("infrared radiation", "invisible light with wavelengths longer than red", "sound with frequencies above human hearing"),
        ("the Doppler effect", "a shift in wave frequency due to relative motion", "a spreading of waves through a narrow slit"),
    ],
    "medicine": [
        ("vaccines", "train the immune system to recognize pathogens", "directly kill bacteria in the bloodstream"),
        ("antibiotics", "treat bacterial but not viral infections", "treat viral but not fungal infections"),
        ("the human heart", "has four chambers", "has two chambers"),
        ("red blood cells", "transport oxygen throughout the body", "fight infections by engulfing pathogens"),
        ("the skeleton of an adult human", "contains 206 bones", "contains over 400 bones"),
        ("the cerebellum", "coordinates movement and balance", "produces insulin for the body"),
        ("X-rays", "a form of radiation used to image bones", "a form of sound used to image fetuses"),
        ("the immune system", "defends the body against pathogens", "regulates body temperature day and night"),
        ("aspirin", "reduces pain, fever, and inflammation", "increases blood pressure and heart rate"),
        ("the lungs", "exchange oxygen and carbon dioxide with blood", "filter waste products from the blood"),
        ("the skin", "the largest organ of the human body", "a tissue that plays no role in temperature regulation"),
        ("platelets", "cell fragments essential for blood clotting", "cells that carry oxygen to tissues"),
        ("anesthesia", "prevents pain sensation during surgery", "boosts the immune response during surgery"),
        ("cholesterol", "a fatty substance important for cell membranes", "a protein that carries oxygen in muscle"),
        ("the spinal cord", "transmits nerve signals between brain and body", "produces red blood cells in adults"),
        ("the thyroid gland", "regulates metabolism through hormones", "controls blood sugar through insulin"),
    ],
    "computer science": [
        ("binary code", "represents information using zeros and ones", "represents information using letters and symbols"),
        ("RAM", "volatile memory that loses data when powered off", "permanent storage that keeps data forever"),
        ("algorithms", "step-by-step procedures for solving problems", "physical components inside a processor"),
        ("Python", "a high-level interpreted programming language", "a low-level assembly language for CPUs"),
        ("machine learning", "programs that improve from data without explicit rules", "programs that follow only hand-written rules"),
        ("operating systems", "manage hardware resources for programs", "compile code directly into electricity signals"),
        ("TCP/IP", "protocols that govern data transmission over networks", "languages used to write web pages"),
        ("a compiler", "translates source code into machine code", "sends packets across the internet"),
        ("recursion", "a function that calls itself", "a function that never returns a value"),
        ("databases", "organized stores of structured data", "devices that cool server rooms"),
        ("a GPU", "a processor optimized for parallel computation", "a device that stores backup tapes"),
        ("encryption", "transforms data so only authorized parties can read it", "compresses files to save disk space"),
        ("open-source software", "publishes source code for anyone to inspect and modify", "sells licenses that forbid viewing the code"),
        ("version control", "tracks changes to files over time", "checks software for viruses automatically"),
        ("neural networks", "models inspired by interconnected computational units", "physical networks connecting computers in a building"),
        ("a cache", "stores frequently used data for faster access", "permanently deletes duplicate files"),
    ],
    "nutrition": [
        ("vitamin C", "found in citrus fruits and important for immune function", "found only in dairy products and builds bone"),
        ("protein", "essential for building and repairing tissues", "the body's main store of fat-soluble vitamins"),
        ("dietary fiber", "aids digestion and is found in whole grains", "a type of fat found in red meat"),
        ("calcium", "important for bone health and found in dairy", "a vitamin produced by the skin in sunlight"),
        ("iron deficiency", "a common cause of anemia", "a common cause of diabetes"),
        ("glucose", "a simple sugar and key energy source for cells", "a protein that repairs muscle fibers"),
        ("omega-3 fatty acids", "found in fish and important for heart and brain", "found in refined sugar and harmful in any amount"),
        ("vitamin D", "synthesized in skin exposed to sunlight", "synthesized in the liver from dietary fiber"),
        ("sodium", "consumed mainly as salt and affects blood pressure", "a carbohydrate that fuels endurance exercise"),
        ("antioxidants", "compounds that protect cells from oxidative damage", "minerals that strengthen tooth enamel only"),
        ("trans fats", "artificially hydrogenated fats that raise heart risk", "natural fats found in olive oil"),
        ("kcal", "a unit of food energy", "a unit of protein content"),
        ("probiotics", "live bacteria that can support gut health", "synthetic chemicals that kill gut bacteria"),
        ("dehydration", "impairs concentration and physical performance", "improves kidney function over time"),
        ("whole grains", "retain bran and germ along with the endosperm", "grains stripped of all fiber and nutrients"),
        ("caffeine", "a stimulant that temporarily reduces fatigue", "a sedative that promotes deep sleep"),
    ],
    "meteorology": [
        ("hurricanes", "form over warm ocean waters", "form over cold polar landmasses"),
        ("lightning", "a discharge of static electricity in storms", "a burst of solar radiation reaching the ground"),
        ("the jet stream", "a fast air current near the top of the troposphere", "a slow ocean current near the equator"),
        ("humidity", "the amount of water vapor in the air", "the amount of dust in the air"),
        ("tornadoes", "rotating columns of air extending from storms to the ground", "vertical movements of magma beneath volcanoes"),
        ("the troposphere", "the lowest layer of the atmosphere where weather occurs", "the layer of atmosphere where satellites orbit"),
        ("monsoons", "seasonal wind reversals bringing heavy rain", "permanent winds that blow from east to west"),
        ("dew point", "the temperature at which air becomes saturated", "the temperature at which water freezes at sea level"),
        ("high-pressure systems", "usually associated with clear and calm weather", "usually associated with violent thunderstorms"),
        ("rainbows", "formed by refraction and reflection in water droplets", "formed by ice crystals in the upper atmosphere"),
        ("fog", "a cloud forming at ground level", "a layer of smoke from distant fires"),
        ("barometers", "instruments measuring atmospheric pressure", "instruments measuring wind speed"),
        ("El Nino", "a periodic warming of the central and eastern Pacific", "a periodic cooling of the North Atlantic"),
        ("hail", "ice pellets formed in strong thunderstorm updrafts", "frozen raindrops that fall in winter"),
        ("the Coriolis effect", "deflects moving air due to Earth's rotation", "heats the air near the equator"),
        ("wind chill", "how wind makes cold air feel colder on skin", "how humidity makes hot air feel heavier"),
    ],
    "economics": [
        ("inflation", "a general rise in prices over time", "a general fall in wages over time"),
        ("gross domestic product", "the total value of goods and services produced in a country", "the total amount of money printed in a year"),
        ("supply and demand", "the main forces setting prices in markets", "the main forces setting interest rates at banks"),
        ("interest rates", "the cost of borrowing money", "the fee banks charge for holding deposits only"),
        ("unemployment rate", "the share of the labor force without work", "the share of the total population over 65"),
        ("exports", "goods and services sold to other countries", "goods and services bought from other countries"),
        ("central banks", "manage monetary policy and currency supply", "issue loans directly to private households"),
        ("bonds", "debt instruments paying fixed interest", "ownership shares in a company"),
        ("recession", "a sustained decline in economic activity", "a rapid increase in stock market prices"),
        ("tariffs", "taxes on imported goods", "subsidies paid to domestic farmers"),
        ("diversification", "spreading investments to reduce risk", "concentrating funds in a single asset"),
        ("compound interest", "interest earned on previously earned interest", "interest paid only on the initial deposit"),
        ("exchange rates", "the price of one currency in terms of another", "the fee for converting coins to banknotes"),
        ("monopoly", "a market with a single dominant seller", "a market with unlimited identical sellers"),
        ("stocks", "shares representing ownership in companies", "contracts to deliver commodities next year"),
        ("deflation", "a general fall in prices over time", "a sudden collapse of the banking system"),
    ],
    "oceanography": [
        ("the Mariana Trench", "the deepest known ocean trench", "the largest coral reef in the Atlantic"),
        ("the Gulf Stream", "a warm Atlantic current affecting European climate", "a cold Pacific current cooling Japan"),
        ("tides", "caused mainly by the gravitational pull of the Moon", "caused mainly by wind friction on the sea surface"),
        ("seawater", "has an average salinity of about 3.5 percent", "has an average salinity of about 0.3 percent"),
        ("plankton", "drifting organisms forming the base of ocean food webs", "large fish that hunt in deep trenches"),
        ("the photic zone", "the sunlit upper layer of the ocean", "the permanently dark layer near the seabed"),
        ("coral reefs", "built by colonies of tiny animals called polyps", "formed by mineral deposits from rivers"),
        ("thermohaline circulation", "global ocean circulation driven by heat and salt", "local wave patterns driven by coastal winds"),
        ("hydrothermal vents", "hot springs on the seafloor hosting unique life", "freshwater springs in mountain aquifers"),
        ("sea ice", "frozen seawater floating on the ocean surface", "frozen freshwater from melting glaciers"),
        ("the benthic zone", "the ecological region at the bottom of the ocean", "the region of open water far from any coast"),
        ("sonar", "uses sound waves to map the seafloor", "uses laser light to measure fish populations"),
        ("ocean acidification", "caused by the ocean absorbing carbon dioxide", "caused by undersea volcanic ash clouds"),
        ("estuaries", "partly enclosed waters where rivers meet the sea", "deep basins in the middle of the open ocean"),
        ("blue whales", "the largest animals known to have lived on Earth", "medium-sized whales found only in rivers"),
        ("abyssal plains", "vast flat regions of the deep ocean floor", "shallow sandy banks near popular beaches"),
    ],
}

GENERAL_TASKS = [
    ("qa",      "Read the passage and answer the question.",
                "What is {subject} according to the passage?"),
    ("summary", "Summarize the passage in one sentence.", None),
    ("classify","Locate a specific fact in the passage.",
                "Which statement mentions {subject}?"),
    ("title",   "Generate a short title for this passage.", None),
    ("extract", "Extract the key entities mentioned in the passage.", None),
    ("rewrite", "Rewrite the passage in a more formal tone.", None),
    ("argument","What is the main topic of this passage?", None),
    ("simple",  "Translate the key idea of the passage into simpler language.", None),
]

CORRECTION_TASKS = [
    ("The following passage contains a factual error. Identify and correct it."),
    ("Review these statements for factual mistakes and provide the correction."),
    ("Check if these claims are consistent. Flag any error and fix it."),
    ("A student wrote the following summary, which contains an error. Find it and give the correct fact."),
    ("This set of claims has a flaw. Explain what is wrong and correct it."),
]

SENTENCE_FORMS = [
    "{subject} — {attr}.",
    "{subject}: {attr}.",
]


def make_passage(rng, domain, facts, n_statements=None, corrupt=False):
    n = n_statements or rng.randint(4, 7)
    chosen = rng.sample(facts, n)
    statements, meta = [], []
    for subject, true_attr, wrong_attr in chosen:
        form = rng.choice(SENTENCE_FORMS)
        attr = wrong_attr if (corrupt and len(meta) == 0) else true_attr
        statements.append(form.format(subject=subject, attr=attr))
        meta.append({"subject": subject, "true_attr": true_attr,
                     "wrong_attr": wrong_attr, "shown_attr": attr})
    corrupt_idx = None
    if corrupt:
        pairs = list(zip(statements, meta))
        rng.shuffle(pairs)
        statements = [p[0] for p in pairs]
        meta = [p[1] for p in pairs]
        corrupt_idx = next(i for i, m in enumerate(meta) if m["shown_attr"] == m["wrong_attr"])
    passage = " ".join(f"({i+1}) {s}" for i, s in enumerate(statements))
    return {"domain": domain, "passage": passage, "statements": meta, "corrupt_idx": corrupt_idx}


def general_sample(rng, domain, facts, idx):
    p = make_passage(rng, domain, facts)
    task = rng.choice(GENERAL_TASKS)
    kind, instr, question = task
    if kind == "qa":
        m = rng.choice(p["statements"])
        user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}\n\nQuestion: {question.format(subject=m['subject'])}"
        assistant = f"According to the passage, {m['subject']} — {m['true_attr']}."
    elif kind == "summary":
        m = rng.choice(p["statements"])
        user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}"
        assistant = f"The passage lists facts about {p['domain']}; for example, {m['subject']} — {m['true_attr']}."
    elif kind == "classify":
        k = rng.randrange(len(p["statements"]))
        m = p["statements"][k]
        user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}\n\nQuestion: {question.format(subject=m['subject'])}"
        assistant = f"Statement ({k+1}) mentions {m['subject']}: {m['shown_attr']}."
    elif kind == "title":
        user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}"
        assistant = f"Key facts about {p['domain']}."
    elif kind == "extract":
        subs = [m["subject"] for m in p["statements"][:3]]
        user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}"
        assistant = "The key entities are: " + "; ".join(subs) + "."
    elif kind == "rewrite":
        m = rng.choice(p["statements"])
        user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}"
        assistant = f"The passage states, in a formal register, that {m['subject']} — {m['true_attr']} — among other facts about {p['domain']}."
    elif kind == "argument":
        user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}"
        assistant = f"The passage is mainly about facts in {p['domain']}."
    else:
        m = rng.choice(p["statements"])
        user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}"
        assistant = f"Put simply: {m['subject']} — {m['true_attr']}."
    return {"messages": [{"role": "user", "content": user},
                          {"role": "assistant", "content": assistant}],
            "source": "general_sft", "task_family": kind, "domain": p["domain"]}


def correction_entry(rng, domain, facts, idx):
    p = make_passage(rng, domain, facts, corrupt=True)
    instr = rng.choice(CORRECTION_TASKS)
    user = f"{instr}\n\nPassage about {p['domain']}:\n{p['passage']}"
    return {"user": user, "passage": p, "instr_idx": CORRECTION_TASKS.index(instr)}


def true_fix(entry):
    p = entry["passage"]
    k = p["corrupt_idx"]
    m = p["statements"][k]
    assistant = (f"The error is in statement ({k+1}). As written, \"{m['subject']} — {m['wrong_attr']}.\" "
                 f"The corrected statement should read: \"{m['subject']} — {m['true_attr']}.\"")
    return {"messages": [{"role": "user", "content": entry["user"]},
                          {"role": "assistant", "content": assistant}],
            "source": "correction_sft", "task_family": "correction", "domain": p["domain"]}


def sham_fix(entry, donor_entry, rng):
    p = entry["passage"]
    k_true = p["corrupt_idx"]
    j = rng.randrange(len(p["statements"]))
    if j == k_true:
        j = (j + 1) % len(p["statements"])
    m = p["statements"][j]
    d = donor_entry["passage"]
    dm = d["statements"][d["corrupt_idx"]]
    assistant = (f"The error is in statement ({j+1}). As written, \"{m['subject']} — {m['shown_attr']}.\" "
                 f"The corrected statement should read: \"{m['subject']} — {dm['wrong_attr']}.\"")
    return {"messages": [{"role": "user", "content": entry["user"]},
                          {"role": "assistant", "content": assistant}],
            "source": "sham_correction", "task_family": "correction", "domain": p["domain"]}


def domain_cycle(rng):
    doms = list(DOMAINS)
    while True:
        rng.shuffle(doms)
        for d in doms:
            yield d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "sft"))
    ap.add_argument("--n_general_arm", type=int, default=4000)
    ap.add_argument("--n_mixed_per_arm", type=int, default=2000)
    ap.add_argument("--n_correction", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng_a = random.Random(args.seed * 1000 + 1)
    arm_a, dc = [], domain_cycle(rng_a)
    for i in range(args.n_general_arm):
        d = next(dc)
        arm_a.append(general_sample(rng_a, d, DOMAINS[d], i))

    rng_g = random.Random(args.seed * 1000 + 2)
    shared_general, dc2 = [], domain_cycle(rng_g)
    for i in range(args.n_mixed_per_arm):
        d = next(dc2)
        shared_general.append(general_sample(rng_g, d, DOMAINS[d], i))

    rng_c = random.Random(args.seed * 1000 + 3)
    entries, dc3 = [], domain_cycle(rng_c)
    for i in range(args.n_correction):
        d = next(dc3)
        entries.append(correction_entry(rng_c, d, DOMAINS[d], i))

    arm_b = list(shared_general) + [true_fix(e) for e in entries]

    rng_s = random.Random(args.seed * 1000 + 4)
    arm_c = list(shared_general) + [
        sham_fix(e, entries[(i + 7) % len(entries)], rng_s) for i, e in enumerate(entries)
    ]

    for name, arm in [("general_sft", arm_a), ("correction_sft", arm_b),
                      ("sham_correction", arm_c)]:
        rng_a.shuffle(arm) if name == "general_sft" else None
        path = out / f"{name}_seed{args.seed}.jsonl"
        with open(path, "w") as f:
            for it in arm:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        lens = [sum(len(m["content"]) for m in it["messages"]) for it in arm]
        print(f"{name}_seed{args.seed}: n={len(arm)} sha={sha} "
              f"chars mean={sum(lens)/len(lens):.0f} min={min(lens)} max={max(lens)}")

    manifest = {
        "seed": args.seed,
        "arms": {
            "arm_a_general": {"n": len(arm_a), "composition": "100% general instruction"},
            "arm_b_correction": {"n": len(arm_b),
                                 "composition": f"{len(shared_general)} general + {len(entries)} true error->fix"},
            "arm_c_sham": {"n": len(arm_c),
                           "composition": f"{len(shared_general)} general (identical to arm B) + "
                                          f"{len(entries)} sham (same user prompts as arm B, pairing broken)"},
        },
        "design_notes": [
            "Arm B and arm C share IDENTICAL user prompts on the correction half; "
            "only the assistant-side error->fix pairing differs (true vs shuffled).",
            "All passages drawn from a fixed fact bank (12 domains); corruption replaces "
            "a true attribute with a same-domain wrong attribute.",
            "Token budgets matched by construction (same passage generator, same length ranges).",
        ],
        "contamination_audit": ("No CHECK/CONTINUE, no graph reasoning, no retrieval formats, "
                                "no five-condition scaffold. Frozen eval sets untouched."),
        "created": "2026-10-03",
    }
    (out / f"manifest_formal_seed{args.seed}.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written")


if __name__ == "__main__":
    main()
