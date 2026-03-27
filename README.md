# rsbookclub-datadumps
Code for the filtering &amp; analysis of rsbookclub data.  

### Important:
It is probably a lot faster to download the data directly from the arctic-shift data dumps through their download tool:   
https://arctic-shift.photon-reddit.com/download-tool  
I'm doing this to double check, since I have noticed that results on the download tool for some users are incomplete.

If you're only interested in the filtered data, clone this repo and check out the /releases directory.  
If you'd like to reproduce the filtering from the 4TB torrent, e.g. for a separate sub, follow the steps below.  

## Part 1 -- Preprocessing of Reddit Data Dumps  
To reproduce the filtered data dumps available in the /releases directory of this repo follow these steps:  
* Step 1: Head over to https://academictorrents.com/details/3d426c47c767d40f82c7ef0f47c3acacedd2bf44/tech&filelist=1
* Step 2: Download the metadata (required to pick which date ranges are of interest in next step) contained in the .torrent file available via the download button
* Step 3: Spin up a VM with a fast internet connection, run ./setup.sh & upload the .torrent to the VM. 
* Step 4: execute monolith_dl.sh: (Alternatively you may first download the torrent w/ download_torrent.sh & then perform filtering w/ filter_data.sh subsequently.) For the filtering step 2GB of RAM per worker is recommended.  
```
chmod +x monolith_dl.sh
nohup ./monolith_dl.sh /workspace/reddit-3d426c47c767d40f82c7ef0f47c3acacedd2bf44.torrent --batch-size 96
# adjust batch-size to the specs of your machine, all 96 files for 2021/01-2024/12 require 1941,82 GB of space.
tail -f pipeline_monolith.log
```
* Step 5: (optional) Repack the filtered outputs for distribution.
```
chmod +x repack.sh && ./repack.sh
```
The output structure mirrors the structure of the data dumps.
* Step 6: Save results & repacked files to disk and nuke your VM since we'll need a GPU for finetuning next.

## Part 2 -- Preprocessing the rsbookclub Data
After running part 1 or alternatively cloning the repo and using the data in ./releases we can proceed with preprocessing.
To convert the compressed monthly data in ./releases to flat comment threads for labelling and downstream processing we execute `preprocessing.sh`.  
This performs decompression & merging, pruning of unnecessary data, and flattening & export of jsonl data to `labelstudio_import.json`, `threads.jsonl`, `and corpus.txt`. 

### Labeling
To proceed with labeling we convert threads.jsonl into tasks.json, such that we can label contiguous threads in label-studio and use the Claude-API to pre-label some, or all, of the threads for human review and downstream processing. 
```bash
python ./preprocessing/convert_threads.py threads.jsonl tasks.json pretty
# Yields tasks.json in pretty printed format.
```
Then we use the Claude-API to prelabel a subset of the 11k threads. Set your API key as `ANTHROPIC_API_KEY` environment variable and run: 
```bash
python ./nlp/prelabel.py extract tasks.json extractions.jsonl --n 10
```
This will query the API with the first 10 documents and save the corresponding extractions to extractions.jsonl. It should be noted, that one could probably cut down on token cost by caching the system prompt but in total this cost me like $40 give or take so I just kept it as is for now. The number of monthly threads is also not too extreme.  
We can then proceed by calculating the respective span indices for each entity like so:
```bash
python ./nlp/prelabel.py annotate tasks.json extractions.jsonl preannotated.json
```
Or, alternatively, we can do both subsequently in one command:
```bash
python ./nlp/prelabel.py run tasks.json extractions.jsonl preannotated.json
```
This will generate `preannotated.json` (also available as zstd compressed file in ./releases/labels/) which we can directly import to LabelStudio.  
Because Claude-Haiku, or any language model really, is not perfect, this will result in about 1.73% of the documents containing obvious errors like:
```python
["the", "The", "a", "A", ...]
```
being erroneously tagged as books or writers. Because of the greedy way in which we calculate label spans after receiving the pre-annotations from Claude-Haiku, this will cause some threads to be tagged with thousands of "A" book entities or similar. For the entire corpus this was an easy fix, we can first obtain a set of the `thread_id`s of the respective therads via tinkering with `./deprecated/false_positives.py`, and then automatically remove the selected labels with `python ./nlp/remove_labels.py --<TASK_ID>`. Note that here we use the TASK_ID we obtain from label studio, either via the API (refer to the label-studio docs or look at the example in the files mentioned above) or by tagging the metadata of the erroneous threads and filtering in the label-studio web interface. Either way we've removed false positives from the 194 culprits and adjusted the greedy span calculation to exclude the cases listed below for future threads.
```python
["the", "The", "par", "Par", "Don", "don", "on", "On", "st.", "ali", "ee", "c.", "C.", "k.", "K.", "in", "In", "of", "Of", "der", "Der", "of the", "people", "on the", "just", "lee", "de", "f.", "nin", "De", "DE", "de", "Dr.", "THE", "m.", "st.", "ali", "ee", "c.", "C.", "k." "K.", "in", "In", "of", "Of", "der", "Der", "of the", "people", "on the", "just", "eve", "A", "a", "t", "T", "Tim", "Tom", "tim", "tom", "40"]
```
Readers may note that this might lead to the partial elimination of correct labels or to false negatives in the worst case. We provide the dataset as is, as we're only interested in a rough explorative overview of the data at the moment, and since Book -> Writer disambiguation is relatively easy via context & the Open Library Data Dumps, and the combined error rate of ~1.73% * P(False Negative via Removal) we're satisfied for now. To combat false negatives we could run an exhaustive search against the OLDD, but this would most likely also not catch everything. Suggestions & pull requests are very welcome here.  
To summarize the complete annotation process, we performed, in order:  
1. Soft Labeling via Claude
2. Greedy Span Calculation
3. Programmatic removal of obvious false positives
4. Manual check of ~ 1000 threads that yielded no labels for false negatives (~25 such threads actually contained false negatives)
5. Visual check of threads impacted by obvious false positive removal
6. TODO: Quick visual inspection of randomly sampled threads to get upper bound estimates on the remaining "hidden false positive" rate.

TODO:  
* Annotate longest thread manually
* Write script to calculate label counts -> Weed out any remaining obvious false positives
* Consolidate auto-generated labels with Open Library Data Dumps https://openlibrary.org/developers/dumps 
* Adjust prelabeling to avoid obvious false positives in the future
* Write scripts to automate integrating monthly reddit data dumps
* Pretrain & finetune model custom model for NER.
* Train model for BOOK - WRITER relation prediction (maybe jointly with NER)

Strategies (from lowest to highest cost):  
* LoRA for NER fine-tuning only.  
* Embeding-only Task Adaptive Pretraining.  
* Selective top-layer DAPT. Unfreeze top ~6 layers or so, should be managable on a single GPU.  
* LoRA-based DAPT: Continued pretraining on corpus updating only LoRA adapters. 

## Literature
* DeBERTaV3: https://arxiv.org/abs/2111.09543  
* Domain-adaptive pretraining these documents https://arxiv.org/pdf/2004.10964  
* NER-BERT for small corpus NER https://arxiv.org/pdf/2112.00405  
* CrossNER: Evaluating Cross-Domain NER https://arxiv.org/pdf/2012.04373
* Simple & Efficient TAPT for Text Classif: https://arxiv.org/pdf/2209.12943
* LoRA Tradeoffs: https://arxiv.org/abs/2405.09673
* BERT Rediscovers the Classical NLP Pipeline https://arxiv.org/pdf/1905.05950
