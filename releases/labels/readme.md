# Contents of ./releases/labels:
`extractions.jonl.zst` the unprocessed outputs of Claude Haiku obtained by running python `./nlp/prelabel.py extract tasks.json extractions.jsonl`  
`preannotated.json.zst`the automatically generated labels generated from Claude Haiku via python `./nlp/prelabel.py annotate tasks.json extractions.jsonl preannotated.json` importable to label studio  
Both of the files above do NOT contain annotations for the 8 longest threads, as we generated them after the fact by splitting long threads into chunks.  

