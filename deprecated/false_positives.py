#!/usr/bin/env python3
"""
Script to analyze preannotated.json to look for files where labels were both:
    1. Broken up into pieces like "The" "on" "V"
    2. Greedily applied to the entire text
This behavior was intended for short threads but causes headaches when the api
returns labels like "The" & on some mentions of Pynchons "V".
"""

import json
from typing import List

with open("preannotated.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

t0 = threads[0]

# preds.keys()
# Out[23]: dict_keys(['model_version', 'score', 'result'])

# this is now a list of predictions, we want to count how often each label text appears
# preds = t0["predictions"][0]["result"]

from collections import Counter
# counts = Counter([x["value"]["text"] for x in preds])

def count_label_text_per_thread(threads: list):
    return {
        x["data"]["thread_id"]: (
            Counter(y["value"]["text"] for y in x["predictions"][0]["result"])
            if x.get("predictions")
            else Counter()
        )
        for x in threads
    }

out = count_label_text_per_thread(threads)

temp = out[list(out.keys())[3]]
# common mistakes: 
bad_terms = [
    "the", "The", "par", "Par", "Don", "don", "on", "On",
    "st.", "ali", "ee", "c.", "C.", "k.", "K.", "in", "In", "of", "Of",
    "der", "Der", "of the", "people", "on the", "just", "lee", "de", "f.",
    "nin", "De", "DE", "de", "Dr.", "THE", "m.",
    "st.", "ali", "ee", "c.", "C.", "k." "K.", "in", "In", 
    "of", "Of", "der", "Der", "of the", "people", "on the", "just",
    "eve", "A", "a", "t", "T", "Tim", "Tom", "tim", "tom", "40"
]
# documents that contain these in high frequency need manual checking
# the 8 longest threads have been manually annotated anyway
# lets check the other 11k first and see how often this is the case

# TODO: writer disambiguation: only label strings if they stand alone "Roth" is correct, "Broth" is incorrect.
# this yields thread_id: count of text labeled in document. Ideally this should return zero for "the" etc.
the = {x: out[y]["the"] for x, y in zip(list(out.keys()), list(out.keys()))}

the_threads = {x: y for x, y in zip(the.keys(), the.values()) if y > 0}

faulty_threads = {
    thread_id: sum(counter.get(term, 0) for term in bad_terms)
    for thread_id, counter in out.items()
    if sum(counter.get(term, 0) for term in bad_terms) > 0
}

len(faulty_threads)
# 150/11194 = 0.0134
# 1.3% error rate, not too bad, we can manually label all of these.

# NOTE: I've already corrected 150 threads containing the "the" deficiencies.
# here we just need the set difference to get the remaining false positives
the_set = set(the_threads.keys())
faulty_set = set(faulty_threads.keys())

remaining_thread_ids = faulty_set - the_set
# this should yield 30 extra threads. So the total error rate during prelabeling
# was about 194/11196 = 1.73%

# detailed_faults = {
#     thread_id: {term: counter.get(term, 0) for term in bad_terms if counter.get(term, 0) > 0}
#     for thread_id, counter in out.items()
#     if any(counter.get(term, 0) > 0 for term in bad_terms)
# }

# # and since "V" gets labeled as faulty even in threads where only the Pynchon novel is labeled
# # the actual error rate is even lower, so this is a conservative estimate.

# export = [
#     {
#         "thread_id": thread_id,
#         "errors": errors,
#         "total_errors": sum(errors.values())
#     }
#     for thread_id, errors in detailed_faults.items()
# ]

# with open("detailed_faults.json", "w", encoding="utf-8") as f:
#     json.dump(export, f, indent=2, ensure_ascii=False)

# # using label-studio api to tag threads for manual review

API_URL = "http://localhost:8080"
API_KEY = ""

from label_studio_sdk import LabelStudio

client = LabelStudio(
    base_url="http://localhost:8080",
    api_key=API_KEY 
)

me = client.users.whoami()
print(me.username, me.email)

projects = client.projects.list()

for project in projects:
    print(f"ID: {project.id} Title: {project.title}")

# returns 100 tasks by default
tasks = client.tasks.list(project = "8")
len(tasks.items)

tasks.items[0]

PROJECT_ID = "8"

faulty_thread_ids = remaining_thread_ids

# Paginate through all ~11k tasks
page = 1
page_size = 10
tagged_count = 0

while True:
    tasks = client.tasks.list(project=PROJECT_ID, page=page, page_size=page_size)
    if not tasks.items:
        page += 1
        continue
    print(f"Retrieved page {page} of size {len(tasks.items)}")
    for task in tasks.items:
        thread_id = task.data.get("thread_id")
        if thread_id in faulty_thread_ids:
            client.tasks.update(
                id=task.id,
                meta={"needs_review": True}
            )
            tagged_count += 1
            print(f"Tagged task {task.id} (thread_id: {thread_id})")
    print(f"Page {page} done — tagged so far: {tagged_count}")
    # Stop if we got fewer results than a full page
    if len(tasks.items) < page_size:
        break
    page += 1

print(f"\nDone. Total tagged: {tagged_count}")


# for now we're gonna remove all the "the" entities from the bad threads
# PROJECT_ID = "8"
# TASK_ID = 8140
# # bad_terms_set = set(["the", "The"])

# task = client.tasks.get(id=TASK_ID)
# print(f"Thread ID: {task.data.get('thread_id')}")
# preds = task.predictions[0]
# preds.result[-1]
# len(preds.result)

# from collections import Counter
# out = Counter([x["value"]["text"] for x in preds.result])

# # Filter out bad predictions
# original_count = len(preds.result)
# clean_result = [r for r in preds.result if r['value']['text'] not in bad_terms]
# removed_count = original_count - len(clean_result)

# print(f"Original: {original_count} | Removed: {removed_count} | Remaining: {len(clean_result)}")

# updated_pred = client.predictions.update(
#     id=preds.id,
#     result=clean_result
# )
# print(f"Done. Prediction {preds.id} now has {len(updated_pred.result)} labels.")
