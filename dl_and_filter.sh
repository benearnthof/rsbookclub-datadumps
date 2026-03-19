# Download monthly data dumps ~15 minutes per torrent on 65MiB/s connection
aria2c \
  --input-file=magnets.txt \
  --max-concurrent-downloads=1 \
  --seed-time=0 \
  --dir=/workspace/downloads \
  --console-log-level=notice

# Filter submissions (~4 minutes on 16 cores of AMD EPYC 9575F)
python3 worker.py /workspace/downloads/reddit/submissions /workspace/filtered/submissions --workers $(nproc)

# Filter comments (~13  minutes on 16 cores of AMD EPYC 9575F)
python3 worker.py /workspace/downloads/reddit/comments /workspace/filtered/comments --workers $(nproc)
