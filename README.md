# rsbookclub-datadumps
Code for the filtering &amp; analysis of rsbookclub data.

How to use this repo:
* Step 1: Head over to https://academictorrents.com/details/3d426c47c767d40f82c7ef0f47c3acacedd2bf44/tech&filelist=1
* Step 2: Download the metadata (required to pick which date ranges are of interest in next step) contained in the .torrent file available via the download button
* Step 3: Spin up a VM with a fast internet connection & execute monolith_dl.sh:
```
chmod +x monolith_dl.sh
nohup ./monolith_dl.sh /workspace/reddit-3d426c47c767d40f82c7ef0f47c3acacedd2bf44.torrent &
tail -f pipeline_monolith.log
```
* Step 4: (optional) Repack the filtered outputs for distribution.
```
chmod +x repack.sh && ./repack.sh
```
The output structure mirrors the structure of the data dumps.
* Step 5: Save results & repacked files to disk and nuke your VM since we'll need a GPU for finetuning next.
