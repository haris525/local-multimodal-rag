# Third-party notices

The project's own code and documentation are under the MIT License (see [LICENSE](LICENSE)). The files
below come from others and keep their own licenses. Python and Node packages are installed from their
registries when the images are built and are not copied into this repository; each keeps its own license.

## Bundled code

| File | Project | License |
|---|---|---|
| `services/gateway/static/vendor/marked.umd.js` | [marked](https://github.com/markedjs/marked) 18.1.1, Copyright (c) 2018-2026 MarkedJS, Copyright (c) 2011-2018 Christopher Jeffrey | MIT |
| `services/gateway/static/vendor/purify.min.js` | [DOMPurify](https://github.com/cure53/DOMPurify) 3.4.17, (c) Cure53 and other contributors; it includes regenerator-runtime, Copyright (c) 2014-present Facebook, Inc. (MIT) | Apache License 2.0 or Mozilla Public License 2.0 (used here under Apache 2.0) |

The license text of each is in the header of its file and at the project links above.

## Data

| File | Source | License |
|---|---|---|
| `services/auth/src/auth/data/common_passwords.txt` | [SecLists](https://github.com/danielmiessler/SecLists), Passwords/Common-Credentials/10k-most-common.txt | MIT |
| `examples/documents/` | Openly licensed papers (arXiv, CC BY 4.0), World Bank data (CC BY 4.0) and public-domain U.S. government files and texts | Per file: see [examples/documents/README.md](examples/documents/README.md), which gives each file's source, license and attribution |
| One question in `eval/tests/test_public.py` and `eval/tests/test_bench_cli.py` ("Were Scott Derrickson and Ed Wood of the same nationality?", shortened) | [HotpotQA](https://hotpotqa.github.io/) dev set: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov and Christopher D. Manning (2018), *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering* | CC BY-SA 4.0 |

The HotpotQA and MuSiQue loaders in `eval/` download those datasets at run time; they are not included.
