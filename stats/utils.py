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

import pandas as pd

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
