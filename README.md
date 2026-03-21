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
* Step 4: execute monolith_dl.sh: (Alternatively you may first download the torrent w/ download_torrent.sh & then perform filtering w/ filter_data.sh subsequently. For the filtering step 2GB of RAM per worker is recommended.  
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
TODO:  
* Convert jsonl threads to flattened documents for each thread (by thread & comment IDs)
* Label data (zeroshot entire thing?)
* Pretrain & finetune model for NER.

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
