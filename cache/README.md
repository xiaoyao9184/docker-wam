# cache

This folder is the cache directory for Hugging Face (HF).

When using online mode, downloaded models will be cached in this folder.

For [offline mode](https://huggingface.co/docs/transformers/main/installation#offline-mode) use, please download the models in advance and specify the model directory,
such as the `surya_det3` model below.

The folder structure for `./cache/huggingface/hub/models--facebook--watermark-anything` is as follows.

```
.
├── blobs
├── refs
│   └── main
└── snapshots
    └── f39387cd46bf657e553aacd52807e4442690eab8
        └── checkpoint.pth

4 directories, 13 files
```

It will use
- `./cache/huggingface/hub/models--facebook--watermark-anything/snapshots/?`

For more details, refer to [up@cpu-offline/docker-compose.yml](./../docker/up@cpu-offline/docker-compose.yml).


## Pre-download for offline mode

Running in online mode will automatically download the model.

install cli

```bash
pip install -U "huggingface_hub[cli]"
```

download model

```bash
huggingface-cli download facebook/watermark-anything --repo-type model --revision main --cache-dir ./cache/huggingface/hub
```