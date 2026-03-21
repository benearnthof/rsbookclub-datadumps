# Check sizes — truncated files will be obviously smaller than their neighbours
ls -lh /workspace/downloads/reddit/submissions/*.zst | sort -k9
ls -lh /workspace/downloads/reddit/comments/*.zst | sort -k9

# Test every remaining .zst and report which ones are actually corrupt
for f in /workspace/downloads/reddit/submissions/*.zst \
          /workspace/downloads/reddit/comments/*.zst; do
    if ! zstd -t "$f" 2>/dev/null; then
        echo "CORRUPT: $f  ($(du -h "$f" | cut -f1))"
    fi
done

file /workspace/downloads/reddit/submissions/RS_2021-01.zst
xxd /workspace/downloads/reddit/submissions/RS_2021-01.zst | head -3

# 28 B5 2F FD = correct zstd magic bytes
# v0.8 could indicate that these files were compressed with larger window size
zstd -dc --long=31 /workspace/downloads/reddit/submissions/RS_2021-01.zst | head -c 500
# this was indeed the problem.
# sions/RS_2021-01.zst : 0 MB...     {"all_awardings":[],"allow_live_comments":false,"archived":false,"auth    or":"chia923","author_created_utc":1593812444,"author_flair_background_color":"#ff4500","author_flair_css    _class":null,"author_flair_richtext":[],"author_flair_template_id":"f5cf88f8-1872-11eb-9541-0e78e1249dad"    ,"author_flair_text":"Bonnou","author_flair_text_color":"dark","author_flair_type":"text","author_fullnam    e":"t2_75fhb5dz","author_patreon_flair":false,"author_premium":false,"can_gild":true,"category":null,"con
