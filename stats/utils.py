"""
All kinds of utils
"""

import csv

with open("stats_unreviewed.csv","r",encoding="utf-8") as f:
    reader = csv.reader(f,delimiter=",")
    out =[x for x in reader]

out = out[1::]
# now we're interested in sorting this by the length of label strings
# intuitively the shorter the string the higher the likelihood of false positives

for x in out:
    x.insert(0,len(x[0]))

out.sort()

temp =[x[1:] for x in sorted(out,key=lambda x: (x[0],-int(x[2])))]

for i in range(0,50):
    print(temp[i])
# we can use label density as a proxy for the likelihood of any one label being a
# false positive. 
# in the example of "o",it turned out to be just a single document for example
# another label like "V" is present in numerous separate threads, (and also one
# of the most popular novels on the sub) which indicates it is not a false positive.

import pandas as pd # type: ignore

df = pd.read_csv("label_stats.csv")

df["density"] = df["count"] / df["doc_length"]

candidates = df.sort_values("density", ascending=False)[df["count"] > 1]

candidates.iloc[150:201]

# false positives in top 200 entities by density: 
# Carbynarah
# book
# Lorrie Moore (Task 5828), (thread_id 1h482od)
# public domain 
# thomas (in url)
# king (looKING, etc.)
# Jesus & The Unabomber: The Haunting of the Heart indeed a book!
# shakespeare in urls
# out-of-print
# john 

# top 200 books
books = df[df["label"] == "BOOK"].sort_values("count", ascending=False)
books.iloc[150:201]

# false positives in top 200 books in individual threads
# trans (one correct label, lots of false positives)
# book (thread about cookbooks)
# Miami
# Palestine (correct label: On Palestine)
# suicide 
# classics

# top 200 writers
writers = df[df["label"] == "WRITER"].sort_values("count", ascending=False)
writers.iloc[149:201]
# false positives in top 200 writers by individual threads
# poe x2
# John
# Ishmael
# Percy
# hegel(ian)

df["len"] = [len(x) for x in df["text"].tolist()]

df = df.sort_values("len", ascending=True)

df_unique = df.drop_duplicates(["thread_id", "text", "label"])

acronyms = df[df["len"] == 3]

letters = set(acronyms["text"].tolist())
letters = letters - {} 
# mapping acronyms & very short titles for disambiguation
# Z is non unique, g was a false positive

out = acronyms[acronyms["text"].isin(letters)].sort_values("count", ascending=False)
out = out[out["label"] == "BOOK"]
out.iloc[49:100]

inputs = """
BM: Blood Meridian - Cormac McCarthy
GR: Gravity's Rainbow - Thomas Pynchon
JR: JR - William Gaddis
IJ: Infinite Jest - David Foster Wallace
V.: V. - Thomas Pynchon
DQ: Don Quixote - Miguel de Cervantes
NW: NW - Zadie Smith
BC: Butcher's Crossing - John Williams
It: It - Stephen King
it: It - Stephen King
IT: It - Stephen King
FW: Finnegans Wake - James Joyce
AK: Anna Karenina - Leo Tolstoy
We: We - Yevgeny Zamyatin
WM: Wittgenstein's Mistress - David Markson
G.: G. - John Berger
S.: S. - Doug Dorst
FT: Financial Times - Financial Times
v.: V. - Thomas Pynchon
ij: Infinite Jest - David Foster Wallace
we: We - Yevgeny Zamyatin
gr: Gravity's Rainbow - Thomas Pynchon
IV: Inherent Vice - Thomas Pynchon
HP: Harry Potter - J. K. Rowling
12: 12 - Manix Abrera
49: The Crying of Lot 49 - Thomas Pynchon
54: 54 - Wu Ming
60: Sixty Stories - Donald Barthelme
AA: Absalom, Absalom! - William Faulkner
As: Ace - Antonio Di Benedetto
BK: The Brothers Karamazov - Fyodor Dostoevsky
DD: Daniel Deronda - George Eliot
EP: The Elementary Particles - Michel Houellebecq
FF: Future Foundation - Future Foundation
GG: The Great Gatsby - F. Scott Fitzgerald
GQ: GQ - Gentlemen’s Quarterly
Go: Go - John Clellon Holmes
He: He: Understanding Masculine Psychology - He: Understanding Masculine Psychology
JS: Jesus' Son - Denis Johnson
LC: Lookout Cartridge - Joseph McElroy
MD: Moby Dick - Herman Melville
MS: My Struggle - Karl Ove Knausgaard
MW: MW - Osamu Tezuka
Me: Me: Stories of My Life - Katharine Hepburn
OD: Outer Dark - Cormac McCarthy
OT: The Bible - The Bible
Ru: Ru - Kim Thúy
S5: Slaughterhouse-Five - Kurt Vonnegut
SD: The Savage Detectives - Roberto Bolaño
SW: Swann's Way - Marcel Proust
TR: The Recognitions - William Gaddis
Tū: Tu - Patricia Grace
Up: Up - Ronald Sukenick
Us: Us - David Nicholls
VF: Vanity Fair - William Makepeace Thackeray
me: ME - Tomoyuki Hoshino
nw: NW - Zadie Smith
雪国: Snow Country - Yasunari Kawabata
KJV: The Bible - The Bible
LRB: London Review of Books - London Review of Books
ice: Ice - Anna Kavan
Ice: Ice - Anna Kavan
n+1: n+1 - n+1
BNW: Brave New World - Aldous Huxley
J R: JR - William Gaddis
AtD: Against the Day - Thomas Pynchon
BAM: Brone Age Mindset - Bronze Age Pervert
C&P: Crime and Punishment - Fyodor Dostoevsky
GEB: Gödel, Escher, Bach - Douglas R. Hofstadter
Job: The Bible - The Bible
M&D: Mason & Dixon - Thomas Pynchon
N+1: n+1 - n+1
S/Z: S/Z - Roland Barthes
W&P: War and Peace - Leo Tolstoy
WAP: War and Peace - Leo Tolstoy
W&M: Women and Men - Joseph McElroy
"""

acronym_map = {
    "A": {"BOOK": "A", "WRITER": "Louis Zukofsky"},
    "C": {"BOOK": "C", "WRITER": "Tom McCarthy"},
    "G": {"BOOK": "G.", "WRITER": "John Berger"},
    "K": {"BOOK": "K.", "WRITER": "Roberto Calasso"},
    "Q": {"BOOK": "Q", "WRITER": "Luther Blissett"},  # Wu Ming pseudonym
    "S": {"BOOK": "S.", "WRITER": "Doug Dorst"},
    "V": {"BOOK": "V.", "WRITER": "Thomas Pynchon"},
    "v": {"BOOK": "V.", "WRITER": "Thomas Pynchon"},
    "X": {"BOOK": "X", "WRITER": "Tim Waggoner"},
    "Ka": {"BOOK": "Ka", "WRITER": "Roberto Calasso"},
}

temp = {k.strip(): {"BOOK": b.strip(), "WRITER": w.strip()} for line in inputs.strip().splitlines() if line.strip() for k, rest in [line.split(":", 1)] for b, w in [rest.split(" - ", 1)]}
acronym_map = acronym_map | temp

import json
with open("acronyms.json", "w", encoding="utf-8") as file:
    json.dump(acronym_map, file)