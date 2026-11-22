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
its full comment, the certified source file and the whole conversation before the write that
added it (messages in full; agent thinking and tool calls as expandable one-line stubs). Data for each session loads on demand from `s/<k>.js`.

Built by `scripts/swechat/viewer/export.py`. Data derived from SWE-chat (Entire.io / SALT-NLP), licensed ODC-By 1.0.
