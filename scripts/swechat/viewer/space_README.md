---
title: Data Visualization
emoji: 🔎
colorFrom: yellow
colorTo: gray
sdk: static
app_file: index.html
pinned: false
---

# Slop Comment Viewer

Browse agent-written code comment sentences (all languages) extracted from
[SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) (ODC-By), each shown with
its full comment, the certified source file and the last 50 conversation turns before the
write that added it. Data for each session loads on demand from `s/<k>.js`.

Built by `scripts/swechat/viewer/export.py`. Data derived from SWE-chat (Entire.io / SALT-NLP), licensed ODC-By 1.0.
