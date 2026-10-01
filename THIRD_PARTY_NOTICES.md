# Third-party components and data

Original SimpleB2T code and documentation are licensed under [MIT](LICENSE).
The inherited code under `simpleb2t/vendor/`, including our modifications to those
files, remains under [CC BY-NC 4.0](simpleb2t/vendor/LICENSE). Its copyright notices
are retained. The MIT grant does not apply to these components or override their
non-commercial restriction. Third-party data and model terms remain as listed below.

- The CNN, Transformer wrapper and D-SigLIP primitives in `simpleb2t/vendor/` derive from the public neuraltrain / sentence-decoding implementation associated with d'Ascoli et al., “Towards decoding individual words from non-invasive brain recordings.” Copyright Meta Platforms, Inc. and affiliates; CC BY-NC 4.0. Unused training frameworks and dataset integrations are excluded. The channel-position helper is reduced to the original invalid-position constant. An unsupported, unused EEGNet aggregation branch raises an explicit error rather than importing another architecture.
- LibriBrain / LibriBrain2 public recordings and linguistic annotations: https://huggingface.co/datasets/pnpl/LibriBrain and https://huggingface.co/datasets/pnpl/LibriBrain2 ; CC BY-NC 4.0. The exact revisions and file hashes are in `simpleb2t/assets/downloads.json`. Raw recordings are not redistributed here.
- Frozen T5 targets are derived from `google-t5/t5-large`, revision `150ebc2c4b72291e770f58e6057481c8d2ed331a`; the upstream model is Apache 2.0. Targets are bundled for consistency and can be regenerated. See the model repository's notices for the original model.
- Qwen3-8B-Base is downloaded from `Qwen/Qwen3-8B-Base` at the revision in `simpleb2t/assets/experiment.json`; Apache 2.0. No Qwen weights are included or trained.
- POS labels were produced with spaCy 3.8.16 and en_core_web_sm 3.8.0 (MIT); only the fixed labels for the benchmark sentences are included.
- `simpleb2t/assets/pronunciations.dict` is a subset of the CMU Pronouncing Dictionary. The original full dictionary SHA-256 was `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`, from https://github.com/cmusphinx/cmudict . Its license is reproduced below.

## CMU Pronouncing Dictionary

Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

This work was supported in part by funding from the Defense Advanced Research Projects Agency, the Office of Naval Research and the National Science Foundation of the United States of America, and by member companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge the contributions of many volunteers to the expansion and improvement of this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
