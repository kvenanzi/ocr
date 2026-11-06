# Citation check

Each source cited in `README.md` was checked on 6 November 2026 by Claude agents: the DOI or URL and metadata against Crossref, arXiv, Hugging Face or GitHub, whether the source supports the sentence that cites it (with the supporting passage quoted), and whether that was judged from the full text, a model card or README, the abstract only, or a secondary description. The post was then corrected to match. Notable corrections: RapidOCR ran the PP-OCRv6 *small* models, not the same models as PaddleOCR (confirmed in the run logs); the two-column book *is* in FineBooks; the FineBooks comparison is dated, because the leaderboard added Chandra OCR 2 on 30 October 2026; and the T4 FlashAttention limit is now cited to the flash-attention README and the vLLM documentation instead of the paper. This file replaces the notes written for the first version of the post. Sentences quoted under "Post" are from the draft before correction.

## Citation check, ocr-benchmark, items R1 to R18

Checked 2026-11-06 against /home/kevin/repos/ocr/docs/post/README.md. All quotes below are copied from text I downloaded and read: arXiv PDFs (text extracted locally), raw README.md files from GitHub and Hugging Face, the arXiv API, Crossref, the GitHub releases API, and the Hugging Face model API (`safetensors.total`, which gives the parameter count of the published weight files). references_notes.md was used only to find sources faster.

## Summary

| Item | Metadata OK | Support | Evidence level | Action needed |
|---|---|---|---|---|
| R1 smith2007 | Yes (Crossref) | Partly: HP origin and 2005 yes; "runs on an ordinary processor" is not in the paper | Full text (Google-hosted PDF) | Move the citation so it does not cover the CPU clause |
| R2 barlow2026 | Yes | Yes | README | None |
| R3 zhang2026b | Yes (arXiv) | Yes | Full text | None |
| R4 rapidai2021 | Yes | Partly: README supports "PaddleOCR models converted to ONNX", but the benchmark's RapidOCR ran the PP-OCRv6 **small** models, while PaddleOCR ran the **medium** ones, so they are not "the same models" | README, release notes, and the benchmark's own run logs | **Reword** (and two related sentences outside the range) |
| R5 jaidedai2024 | Yes | Yes | README | None |
| R6 mindee2021 | Yes | Yes | README | None |
| R7 paruchuri2025 | Yes | Yes, from the v0.17.1 README. The current README describes a different product (Surya 2.5, "a 689M param OCR model") | README at tag v0.17.1 and current README | Optional: link the v0.17.1 tag |
| R8 rednote2026 | Yes | Partly: the header/footer clause and 3B size: yes. The card never says dots.mocr "extends dots.ocr"; the paper says so | Model card, plus the paper's full text | Add a citation to the paper, or reword |
| R9 poznanski2025b | Yes (arXiv) | Yes | Full text | None (size 8.3 comes from the weight files; the paper says 7B) |
| R10 datalab2026a | Yes | Yes | Model card | None (5.3 is not in the card text; it matches the weight files and Datalab's own table on the Surya OCR 2 card) |
| R11 datalab2026b | Yes | Partly: the card calls it "Surya 2", "a 650M param OCR model" that the `surya-ocr` package runs. It is a model, not a successor toolkit. Size: the card says 0.65B; the weight files give 0.686B | Model card | Reword; decide which size to show |
| R12 zai2026 | Yes | Yes (both sentences) | Model card | None for support. Size 1.33 matches the weight files, but the card says "only 0.9B parameters" |
| R13 cui2025b | Yes (arXiv) | Partly: the paper supports layout-first for the original PaddleOCR-VL (PP-DocLayoutV2), not version 1.6. The 1.6 report (arXiv 2606.03264) states the same thing for 1.6 | Full text of both papers | Cite the 1.6 report for the table row (sentence 2 is fine as is) |
| R14 athmaas2026 | Yes | Yes | Model card + config.json | None (the card says 0.8B; the weight files give 0.853B) |
| R15 taghadouini2026b | Yes | Yes (both sentences) | Full blog post | None |
| R16 wei2026 | Yes (arXiv) | Partly: the 3B decoder is supported (a mixture-of-experts model). "A few image tokens" is not: the paper uses 256 to 1,120 | Full text | Reword |
| R17 ibm2025 | Yes | Rows 1–2: yes. Sentence 3: partly. The card ties the exclamation marks to GPUs without bfloat16, such as the T4, and gives float32 as the fix. It does not mention float16 | Model card | Reword sentence 3 |
| R18 nassar2025 | Yes (arXiv) | Yes (both) | Full text | None |

**Size column (the note in the claims file).** Every size in the table matches the parameter count of the model's published weight files on Hugging Face (`safetensors.total`): dots.mocr 3,039,179,264; olmOCR-2-7B-1025 8,292,166,656; chandra-ocr-2 5,295,564,288; surya-ocr-2 686,154,304; GLM-OCR 1,325,258,240; PaddleOCR-VL-1.6 958,588,736; OvisOCR2 852,985,920; LightOnOCR-2-1B 1,005,647,872; DeepSeek-OCR-2 3,389,119,360; granite-docling-258M 257,517,120. Five cited sources state a different size in their text:

| Model | Size in the post | Size stated by the source |
|---|---|---|
| GLM-OCR | 1.33 | 0.9B |
| Surya OCR 2 | 0.69 | 650M (0.65B) |
| PaddleOCR-VL | 0.96 | 0.9B |
| OvisOCR2 | 0.85 | 0.8B |
| olmOCR 2 | 8.3 | 7B |

The post does not say where the column comes from. A short note under the table would fix this: "Sizes are the parameter counts of the published weight files on Hugging Face; some model cards state rounded or smaller figures."

---

## R1. smith2007

**Metadata.** Crossref, doi:10.1109/ICDAR.2007.4376991: title "An Overview of the Tesseract OCR Engine", author R. Smith, container "Ninth International Conference on Document Analysis and Recognition (ICDAR 2007) Vol 2", pages 629-633, IEEE, 2007. Matches.

**Evidence.** Full text: the 5-page PDF hosted by Google Research (research.google.com/pubs/archive/33418.pdf).

**Sentence:** "Tesseract, begun at Hewlett-Packard and released as open source in 2005, runs on an ordinary processor ([Smith 2007](#references))."

- HP origin: yes. "Tesseract is an open-source OCR engine that was developed at HP between 1984 and 1994." "Tesseract began as a PhD research project [2] in HP Labs, Bristol".
- 2005: yes. "In late 2005, HP released Tesseract for open source."
- "Runs on an ordinary processor": **not in the paper.** I searched it for CPU, processor, machine, hardware, speed and GPU, and found nothing that says this. The current Tesseract README does not say it either (it confirms "In 2005 Tesseract was open sourced by HP"). The post's settings table records that the benchmark ran Tesseract on a CPU, so the clause is the benchmark's own setting.

**Support: partly.** Proposed rewording (the citation moves so that it covers only what Smith says):

> Tesseract, begun at Hewlett-Packard and released as open source in 2005 ([Smith 2007](#references)), runs on an ordinary processor.

## R2. barlow2026

**Metadata.** github.com/ocrmypdf/OCRmyPDF resolves. The README has an SPDX header, "SPDX-FileCopyrightText: 2015 James R. Barlow". v17.0.0 was published 2026-01-30, and the latest release is v17.13.0 (2026-09-28). Author, year and "(v17)" are fine. The post's reference list gives the title "OCRmyPDF". The .bib entry's title is "OCRmyPDF: adds an OCR text layer to scanned PDF files". Both are acceptable.

**Evidence.** README (raw, main branch).

**Sentence:** "OCRmyPDF ([Barlow 2026](#references)) wraps Tesseract in steps that straighten, clean, and rotate each page first."

- "If requested, deskews and/or cleans the image before performing OCR"
- "--rotate-pages             # it can fix pages that are misrotated" and "--deskew                   # it can deskew crooked PDFs!"
- "Uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) engine"
- "# Deskew (straighten crooked pages)"

**Support: yes.** These steps are optional ("If requested"), and the benchmark turned them on (the settings table lists `--deskew --clean --rotate-pages`), so the sentence is fair. No change.

## R3. zhang2026b

**Metadata.** arXiv 2606.13108v1, submitted 2026-06-11. Title "PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks". The 16 authors begin Yubo Zhang, Xueqing Wang, Manhui Lin, Yue Zhang, Penglongyi Deng, Ting Sun, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Changda Zhou. Matches.

**Evidence.** Full text (PDF).

**Sentence:** "PaddleOCR's PP-OCRv6 models ([Zhang et al 2026](#references)) find text regions with one neural network and read them with a second, …"

- "Figure 2|Overview of the PP-OCRv6 system. The pipeline consists of text detection and text recognition, both built on the unified LCNetV4 backbone."
- "In PP-OCRv6, we adopt LCNetV4 as the backbone network for the text detection model."
- "PP-OCRv6_tiny (0.43M detection, 1.1M recognition)"
- "We benchmark the end-to-end OCR pipeline (detection + recognition)"

**Support: yes.** No change.

## R4. rapidai2021

**Metadata.** github.com/RapidAI/RapidOCR resolves. The repository was created 2021-01-04. The README's citation block is `@misc{RapidOCR 2021, title={{Rapid OCR}: OCR Toolbox}, author={RapidAI Team}, ... year={2021}}`. Matches.

**Evidence.** README; GitHub release notes for v3.9.0, v3.9.1 and v3.9.2; the benchmark's own run logs and engine code in /home/kevin/repos/ocr.

**Sentence:** "… and RapidOCR ([RapidAI 2021](#references)) runs the same models through a different runtime."

- README: "we innovatively converted the models in PaddleOCR into the highly compatible ONNX format to simplify and accelerate the inference deployment of OCR models on various terminal devices."
- README: "The bundled and hosted OCR model files are derived from official PaddleOCR models."
- Release notes v3.9.0 (github.com/RapidAI/RapidOCR/releases/tag/v3.9.0, 2026-06-23): "This release sets the PP-OCRv6 det small detection model and rec small recognition model as the default models." and "This version currently supports inference with only two engines: ONNXRuntime and OpenVINO."

**Problem: the benchmark did not run the same models.**
- ocrbench/engines/rapidocr_engine.py: `Default models are PP-OCRv6 "small" multilingual det+rec.` configs/engines.yaml passes `params: {}`, so the defaults are used.
- results/runs/20261004-163611_G4_standard/rapidocr/worker.log: "Using …/rapidocr/models/PP-OCRv6_det_small.onnx", "Using …/rapidocr/models/PP-OCRv6_rec_small.onnx", and also "ch_ppocr_mobile_v2.0_cls_mobile.onnx", a text-direction classifier.
- results/runs/20261004-163611_G4_standard/paddleocr/worker.log: "Creating model: ('PP-OCRv6_medium_det', …)", "Creating model: ('PP-OCRv6_medium_rec', …)".
- rapidocr 3.9.2 was installed in every run (status.json).

The RapidOCR README supports "converted PaddleOCR models, run through ONNX". The benchmark's RapidOCR used the small PP-OCRv6 tier, and its PaddleOCR used the medium tier.

**Support: partly.** Proposed rewording:

> PaddleOCR's PP-OCRv6 models ([Zhang et al 2026](#references)) find text regions with one neural network and read them with a second, and RapidOCR ([RapidAI 2021](#references)) runs the smaller "small" versions of these models, converted to the ONNX format, through a different runtime.

That RapidOCR's default is the small tier is supported by the v3.9.0 release notes (URL above, verified to resolve). That the benchmark used the defaults is supported by the run logs.

**Related sentences outside R1–R18 that make the same claim. They are not cited, but I recommend changing them:**
- Settings table row: "| RapidOCR 3.9 | PP-OCRv6 detection and recognition in ONNX Runtime | CPU |" → "| RapidOCR 3.9 | PP-OCRv6 small detection and recognition in ONNX Runtime | CPU |"
- Details paragraph: "RapidOCR runs the same models on CPU hosts." → "RapidOCR runs the smaller PP-OCRv6 models on CPU hosts."
- Code comment, for the record only: configs/engines.yaml has "# The same PP-OCR models run fast on CPU via ONNX: see `rapidocr`."

## R5. jaidedai2024

**Metadata.** github.com/JaidedAI/EasyOCR resolves. The latest release is v1.7.2, published 2024-09-24 (GitHub API); the README says "24 September 2024 - Version 1.7.2". Matches.

**Evidence.** README.

**Sentence:** "EasyOCR ([JaidedAI 2024](#references)) and docTR ([Mindee 2021](#references)) work the same way with other networks."

- "Detection execution uses the CRAFT algorithm from this [official repository](https://github.com/clovaai/CRAFT-pytorch) and their [paper](https://arxiv.org/abs/1904.01941)"
- "The recognition model is a CRNN ([paper](https://arxiv.org/abs/1507.05717))."

**Support: yes** (one network detects, another recognises).

## R6. mindee2021

**Metadata.** github.com/mindee/doctr resolves. The README's citation block is `@misc{doctr2021, title={docTR: Document Text Recognition}, author={Mindee}, year={2021}, ...}`. Matches. The README also says: "docTR was originally created by Mindee. It is now actively developed and maintained by t2k GmbH."

**Evidence.** README.

- "End-to-End OCR is achieved in docTR using a two-stage approach: text detection (localizing words), then text recognition (identify all characters in the word)."

**Support: yes.**

## R7. paruchuri2025

**Metadata.** github.com/datalab-to/surya resolves. The citation block, which is the same in the v0.17.1 README and the current one, is `@misc{paruchuri2025surya, author = {Vikas Paruchuri and Datalab Team}, title = {Surya: A lightweight document OCR and analysis toolkit}, year = {2025}, ...}`. Matches. v0.17.1 was published 2026-01-31.

**Evidence.** The README at tag v0.17.1 (raw), and the current main README.

**Sentence:** "Surya 0.17 ([Paruchuri 2025](#references)) is a line-by-line OCR toolkit from the company Datalab."

From the v0.17.1 README:
- "Surya is a document OCR toolkit that does:" … "Line-level text detection in any language"
- "Surya predicts line-level bboxes, while tesseract and others predict word-level or character-level."
- "For our managed API or on-prem document intelligence solution, check out [our platform here](https://datalab.to?utm_source=gh-surya)."

**Support: yes.** One caution: the current main README now begins "Surya is a 689M param OCR model" and describes Surya 2.5, a single VLM. A reader who follows the link today sees a different product from the one benchmarked. Optional: change the URL to the tag, https://github.com/datalab-to/surya/tree/v0.17.1 (verified, HTTP 200).

## R8. rednote2026

**Metadata.** huggingface.co/dots-studio/dots.mocr resolves (HTTP 200); rednote-hilab/dots.mocr redirects to it. The repository was created 2026-03-19 and is MIT-licensed. The card's badges and links point to rednote-hilab (GitHub, X, and rednote). Author "rednote", 2026 is fine.

**Evidence.** Model card (raw README.md). For "extends dots.ocr" I also read the full text of the paper linked from the card (arXiv 2603.13032).

**Table row:** "| dots.mocr ([rednote 2026](#references)) | 3.0 | Extends dots.ocr; its plain-text mode leaves out page headers and footers by design |"

- Header/footer mode, yes: "# Parse text only, except Page-header and Page-footer" followed by "python3 dots_mocr/parser.py demo/demo_image1.jpg --prompt prompt_ocr". Also: "`demo_image1_nohf.md`, is also provided, which excludes page headers and footers for compatibility with benchmarks like Omnidocbench and olmOCR-bench."
- Size 3.0, yes: the card's comparison table lists dots.mocr as "3B", and its limitations say "the capacity constraints of a 3B-parameter VLM". The weight files hold 3,039,179,264 parameters.
- "Extends dots.ocr": **not stated on the card.** The card only says "we have reduced the rate of parsing failures compared to the previous version", without naming that version, and lists dots.ocr as a separate model in its tables. The paper (Zheng et al 2026, arXiv 2603.13032) says: "The architecture of our method adheres to the fundamental design principles established in previous work [15]", where [15] is "Yumeng Li, Guang Yang, Hao Liu, Bowen Wang, and Colin Zhang. dots.ocr: Multilingual document …". It also uses "dots.ocr as an auto-labeling engine".

**Support: partly.** Two options:
- (a) Add the paper as a reference and cite it in the row: "| dots.mocr ([rednote 2026](#references)) | 3.0 | Follows the design of dots.ocr ([Zheng et al 2026](#references)); its plain-text mode leaves out page headers and footers by design |". Reference, from the arXiv API: Zheng H, Li Y, Zhang K, Xin L, Zhao G, Liu H, Chen J, Lou J, Fu Q, Yang R, et al (2026). "Multimodal OCR: Parse Anything from Documents". *arXiv preprint arXiv:2603.13032*. https://arxiv.org/abs/2603.13032 (verified, HTTP 200; 25 authors).
- (b) Without a new reference: "| dots.mocr ([rednote 2026](#references)) | 3.0 | A newer model from the makers of dots.ocr; its plain-text mode leaves out page headers and footers by design |". This is weaker. The card supports it only through its rednote-hilab links and its reuse of the dots.ocr Docker image and assets.

**Sentence 2:** "Models trained to leave them out pass: dots.mocr ([rednote 2026](#references)), …". This is supported by the quotes above, and the percentages are the benchmark's own. One nuance: the card shows only that the mode leaves them out. In the code the prompt string itself does not ask for this (references_notes.md records that the prompt is "Extract the text content from this image."), so "trained to leave them out" is a fair description. Yes.

## R9. poznanski2025b

**Metadata.** arXiv 2510.19817v1, submitted 2025-10-22. "olmOCR 2: Unit Test Rewards for Document OCR". Authors Jake Poznanski, Luca Soldaini, Kyle Lo. No journal reference on arXiv. Matches.

**Evidence.** Full text (PDF).

**Table row:** "| olmOCR 2 ([Poznanski et al 2025b](#references)) | 8.3 | Qwen2.5-VL-7B trained with rewards computed from unit tests; run in its FP8 version |"

- "We start with a Qwen2.5-VL-7B-Instruct model that has been fine-tuned on olmOCR-mix-1025"
- "a specialized, 7B vision language model (VLM) trained using reinforcement learning with verifiable rewards (RLVR), where our rewards are a diverse set of binary unit tests."
- "We apply Group Relative Policy Optimization (GRPO) … using our synthetic verifiable unit tests as binary-valued reward signals."
- The paper's list of released models includes "olmOCR-2-7B-1025-FP8", so the FP8 version exists. That the benchmark ran it is the benchmark's own statement.
- Size: the paper says "7B". The weight files (allenai/olmOCR-2-7B-1025) hold 8,292,166,656 parameters, so 8.3 matches them. The HF metadata gives base_model Qwen/Qwen2.5-VL-7B-Instruct.

**Support: yes.**

## R10. datalab2026a

**Metadata.** huggingface.co/datalab-to/chandra-ocr-2 resolves. The repository was created 2026-03-16 and has an OpenRAIL licence. Datalab, 2026 is fine.

**Evidence.** Model card.

**Table row:** "| Chandra OCR 2 ([Datalab 2026a](#references)) | 5.3 | Datalab's document model |"

- "Chandra 2 is a state of the art OCR model from [Datalab](https://www.datalab.to) that outputs markdown, HTML, and JSON. It is highly accurate at extracting text from images and PDFs, while preserving layout information."
- Size: the card text does not state it. The weight files hold 5,295,564,288 parameters. Datalab's own Surya OCR 2 card lists "Chandra OCR 2 (Datalab) | 5.3B".

**Support: yes.** No change needed.

## R11. datalab2026b

**Metadata.** huggingface.co/datalab-to/surya-ocr-2 resolves. The repository was created 2026-05-14 and has an OpenRAIL licence. Fine.

**Evidence.** Model card.

**Table row:** "| Surya OCR 2 ([Datalab 2026b](#references)) | 0.69 | Datalab's successor to the Surya toolkit |"

- "Surya is a 650M param OCR model with these features:"
- "pip install surya-ocr" … "Surya 2 runs layout, OCR, and table recognition through a single VLM served by `vllm` (GPU) or `llama.cpp` (CPU / Apple Silicon)."
- "Layout, OCR, and table recognition all share a single vision-language model (Qwen3.5-style architecture, ~650M params)." … "Text-line detection is a separate small torch model"
- The card's table has "| **Surya OCR 2** (Datalab) | **0.65B** | **83.3** |".

The card presents Surya OCR 2 as the model that the same Surya package (`surya-ocr`) runs, not as a toolkit that replaces Surya. The card never uses the word "successor", though the name "Surya 2" makes it a newer version.

**Size:** the card says 650M (0.65B). The weight files hold 686,154,304 parameters (0.69).

**Support: partly.** Proposed rewording:

> | Surya OCR 2 ([Datalab 2026b](#references)) | 0.69 | Datalab's Surya 2 model, a single VLM that newer versions of the Surya toolkit run |

Either keep 0.69 and add the size note proposed in the summary, or use 0.65 to match the card.

## R12. zai2026

**Metadata.** huggingface.co/zai-org/GLM-OCR resolves. The repository was created 2026-01-30 and is MIT-licensed. Fine.

**Evidence.** Model card.

**Table row:** "| GLM-OCR ([ZAI 2026](#references)) | 1.33 | Its software kit runs a layout-detection model first |"

- "Combined with a two-stage pipeline of layout analysis and parallel recognition based on PP-DocLayout-V3, GLM-OCR delivers robust and high-quality OCR performance across diverse document layouts."
- "For document parsing tasks, we strongly recommend using our [official SDK](https://github.com/zai-org/GLM-OCR). Compared with model-only inference, the SDK integrates PP-DocLayoutV3 and provides a complete, easy-to-use pipeline for document parsing, including layout analysis and structured output generation."
- Size: the card says "With only 0.9B parameters, GLM-OCR supports deployment via vLLM, SGLang, and Ollama". The weight files hold 1,325,258,240 parameters. 1.33 matches the weight files, not the card. (The extra weights may be the multi-token-prediction layers the card mentions; this is my guess and I did not check it.)

**Sentence 2:** "Both models are designed to run inside a pipeline in which a layout model finds the tables first ([Cui et al 2025](#references); [ZAI 2026](#references)), …". For GLM-OCR the quotes above support it. The 48.8% is the benchmark's own.

**Support: yes** for both. Size: see the note in the summary.

## R13. cui2025b

**Metadata.** arXiv 2510.14528 (v1 2025-10-16, current v4). "PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model". The 18 authors begin Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin. Matches.

**Evidence.** Full text (PDF) of 2510.14528. For version 1.6 I also read the full text of arXiv 2606.03264.

**Table row:** "| PaddleOCR-VL 1.6 ([Cui et al 2025](#references)) | 0.96 | Its official system runs a layout model first, then the VLM on each region |"

- From Cui et al 2025, about the original version: "PaddleOCR-VL decomposes the complex task of document parsing into a two stages, as illustrated in Figure 2. The first stage, PP-DocLayoutV2, is responsible for layout analysis, where it localizes semantic regions and predicts their reading order. Subsequently, the second stage, PaddleOCR-VL-0.9B, leverages these layout predictions to perform fine-grained recognition of diverse content, including text, tables, formulas, and charts."
- From the PaddleOCR-VL-1.6 report (arXiv 2606.03264), about the version the benchmark ran: "The full system consists of two models: PP-DocLayoutV3 for layout analysis and PaddleOCR-VL-1.6-0.9B for vision-language understanding." and "For document parsing, the system follows a robust two-stage framework. In the first stage, PP-DocLayoutV3 performs high-precision layout analysis … In the second stage, PaddleOCR-VL-1.6-0.9B recognizes the localized regions across diverse document elements, including text, tables, formulas, charts, and seals."
- Size: both papers say 0.9B. The PaddleOCR-VL-1.6 weight files hold 958,588,736 parameters (0.96).

**Support: partly.** Cui et al describe version 1.0 with a different layout model (V2). The 1.6 report is the source that supports the row as written. Proposed change, which re-adds the 1.6 report. The .bib key `zhang2026b` suggests that `zhang2026a` was this paper before it was removed:

> | PaddleOCR-VL 1.6 ([Zhang et al 2026a](#references)) | 0.96 | Its official system runs a layout model first, then the VLM on each region |

Reference, verified on the arXiv API: Zhang Z, Liu H, Liang S, Zhang Y, Xiang Y, Liu J, Sun T, Lin M, Zhang Y, Zhou C, et al (2026a). "PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training". *arXiv preprint arXiv:2606.03264*. https://arxiv.org/abs/2606.03264 (HTTP 200; submitted 2026-06-02; 15 authors). If you re-add it, the PP-OCRv6 citation must become "Zhang et al 2026b", and both reference lines need their letters. Alternatively, keep Cui et al and add the 1.6 report next to it: "([Cui et al 2025](#references); [Zhang et al 2026a](#references))".

**Sentence 2** ("Both models are designed to run inside a pipeline in which a layout model finds the tables first ([Cui et al 2025](#references); [ZAI 2026](#references)), …"): **yes.** The design claim holds for the PaddleOCR-VL family, and the quote from Cui et al supports it. No change needed. Adding the 1.6 report here is optional.

## R14. athmaas2026

**Metadata.** huggingface.co/ATH-MaaS/OvisOCR2 resolves. The repository was created 2026-07-13 and is Apache-2.0. The card's metadata now includes `new_version: ATH-MaaS/OvisOCR3`, so a newer model exists; this does not affect the citation.

**Evidence.** Model card and config.json.

**Table row:** "| OvisOCR2 ([ATH-MaaS 2026](#references)) | 0.85 | Qwen3.5-0.8B trained further for OCR |"

- Card metadata: "base_model: Qwen/Qwen3.5-0.8B"
- "OvisOCR2 is developed by post-training Qwen3.5-0.8B using a carefully designed data engine that combines real-world and synthetic data, together with a multi-stage training recipe integrating SFT, RL, and OPD."
- config.json: `"architectures": ["Qwen3_5ForConditionalGeneration"]`
- Size: the card says "a compact 0.8B end-to-end model". The weight files hold 852,985,920 parameters (0.85).

**Support: yes.**

## R15. taghadouini2026b

**Metadata.** The URL huggingface.co/blog/lightonai/lightonocr-2 resolves. Title "LightOnOCR-2-1B: a lightweight high-performance end-to-end OCR model family", published January 19, 2026, by Said Taghadouini, Adrien Cavaillès and Baptiste Aubertin. Matches.

**Evidence.** Full blog post (HTML).

- Row ("Reads the whole page in one step"): "We're releasing LightOnOCR-2-1B, our second-generation 1B-parameter, end-to-end vision-language OCR model optimized for state-of-the-art conversion of document pages (PDF renders) into clean, naturally ordered text without relying on multi-stage pipelines." Size 1.0: "1B-parameter"; the weight files hold 1,005,647,872.
- Sentence 2 ("LightOnOCR's own report of its olmOCR-bench score leaves this category out"): "LightOnOCR-2-1B scores 83.2 ± 0.9 on OlmOCR-Bench" and "Table 1: OlmOCR-Bench results (headers/footers category excluded)."

**Support: yes** for both.

## R16. wei2026

**Metadata.** arXiv 2601.20552v1, submitted 2026-01-28. "DeepSeek-OCR 2: Visual Causal Flow". Authors Haoran Wei, Yaofeng Sun, Yukun Li. Matches.

**Evidence.** Full text (PDF).

**Table row:** "| DeepSeek-OCR 2 ([Wei et al 2026](#references)) | 3.4 | Compresses the page into a few image tokens, then decodes with a 3-billion-parameter language model |"

- Decoder: "we retain DeepSeek-OCR's decoder − a 3B-parameter MoE structure with about 500M active parameters."
- Image tokens: "DeepSeek-OCR 2, which preserves the image compression ratio and decoding efficiency of DeepSeek-OCR" and "We constrain visual tokens fed to the LLM between 256 and 1120. The lower bound (256) corresponds to DeepSeek-OCR's tokenization of 1024 ×1024 images, while the upper bound (1120) matches Gemini-3 pro's [44] maximum visual token budget."
- The paper's main contribution, which the row does not mention, is that the encoder reorders visual tokens ("DeepEncoder V2 … capable of dynamically reordering visual tokens upon image semantics").
- Size: the paper gives no total. The weight files hold 3,389,119,360 parameters (3.4).

"A few image tokens" is not supported: up to 1,120 tokens is not "a few". "3-billion-parameter language model" is correct, but it leaves out that only about 500M parameters are active.

**Support: partly.** Proposed rewording:

> | DeepSeek-OCR 2 ([Wei et al 2026](#references)) | 3.4 | Compresses the page into 256 to 1,120 image tokens, then decodes with a 3-billion-parameter mixture-of-experts language model |

## R17. ibm2025

**Metadata.** huggingface.co/ibm-granite/granite-docling-258M resolves. "Developed by: IBM Research", "Release Date: September 17, 2025", Apache 2.0. Title "granite-docling-258M". Matches.

**Evidence.** Model card (raw README.md, last commit 2025-09-23, "Add Troubleshooting section to README (#23)").

**Row 1:** "| Granite-Docling ([IBM 2025](#references)) | 0.26 | Writes the DocTags markup of SmolDocling ([Nassar et al 2025](#references)), which a library converts to text |"
- "We incorporated DocTags into our LLM's supervised fine-tuning (SFT) data to help the model become familiar with the format, enabling faster convergence and mitigating issues previously observed with SmolDocling."
- "Granite-docling-258M is fully integrated into the Docling pipelines, carrying over existing [features](https://huggingface.co/ds4sd/SmolDocling-256M-preview)"
- "you can use bare **transformers**, **vllm**, **onnx** or **mlx-vlm** to perform inference, and [docling-core](https://github.com/docling-project/docling-core) APIs to convert results to variety of output formats (md, html, etc.)"
- Size: the weight files hold 257,517,120 parameters (0.26).
- **Yes.**

**Sentence 2:** "Its card does not mention handwriting ([IBM 2025](#references))." I searched the whole card for "handwrit" and found nothing; it states "Language(s): English (NLP)". **Yes.**

**Sentence 3:** "On the T4, VLMs run in float16, except LightOnOCR-2 and Granite-Docling in float32: in float16 LightOnOCR-2 scored 0% in the smoke runs, reaching its token limit on every page, and Granite-Docling's card warns that it outputs only exclamation marks ([IBM 2025](#references))."
- "2. The model outputs only exclamation marks (i.e. "!!!!!!!!!!!!!!!"). This is seen on older NVIDIA GPUs, such as the T4 GPU available in Google Colab, because it lacks support for `bfloat16` format. You can work around it by setting the `dtype` to `float32`."
- The card never mentions float16. It ties the fault to the T4's lack of bfloat16 and gives float32 as the fix. On a T4 the usual way to run without bfloat16 is float16, so the post's reading is plausible, but it goes beyond the card.

**Support: partly.** Proposed rewording:

> On the T4, VLMs run in float16, except LightOnOCR-2 and Granite-Docling in float32: in float16 LightOnOCR-2 scored 0% in the smoke runs, reaching its token limit on every page, and Granite-Docling's card warns that on the T4, which lacks bfloat16, it outputs only exclamation marks unless it is run in float32 ([IBM 2025](#references)).

## R18. nassar2025

**Metadata.** arXiv 2503.11576v1, submitted 2025-03-14. "SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion". The 13 authors begin Ahmed Nassar, Andres Marafioti, Matteo Omenetti, Maksym Lysak, Nikolaos Livathinos, Christoph Auer, Lucas Morin, Rafael Teixeira de Lima, Yusik Kim, A. Said Gurbuz. No journal reference on arXiv, so citing it as a preprint is correct. Matches.

**Evidence.** Full text (PDF).

- Row (DocTags is SmolDocling's markup): "To train VLMs for standardized document conversion, we introduce the DocTags markup format". **Yes.**
- Sentence 2 ("DocTags marks a figure with a `<picture>` element and four location tokens"): "Tags define document block types such as: text, caption, footnote, formula, title, list_item, page_footer, page_header, picture, section_header, otsl, code, document_index, etc. Each element can nest additional location tags encoding its position on the page as a bounding box, represented in DocTags as <loc_x1><loc_y1><loc_x2><loc_y2>." The appendix adds: "Within <picture> elements we include picture classification with extra standalone tags <image_class>." **Yes.** A picture element can also carry a classification tag. That is still consistent with the post's inference from about 12 tokens per line, and the post already labels it an inference.


## Results B: items R19 to R37 (ocr-benchmark)

Checked 2026-11-06 against the live sources. Downloaded copies are in `/tmp/claude-1000/-home-kevin-repos-politics/eb377b40-d93b-4fb4-a383-7ee52ab629c0/scratchpad/refcheck/ocr-benchmark/dl_b/`.

## Summary

| Item | Metadata OK | Support | Evidence level | Action needed |
|---|---|---|---|---|
| R19 qwen2026 | Yes (title, Qwen Team, Feb 2026) | Partly: the blog supports "general" but never mentions 2B or 9B | Full blog text (from the site's article API) + the two model cards | Add the Qwen3.5-2B and Qwen3.5-9B model cards as citations for the sizes |
| R20 gemma2026 | Yes | Yes (E4B covered; 8B total) | Full text (arXiv HTML) | None |
| R21 bai2025a | Yes | Yes (3B and 7B named; sizes from HF metadata) | Full text (arXiv HTML) + HF safetensors metadata | None |
| R22 majstorovic2026 | Yes | Claim 1: yes, together with R23. Claim 2: partly. The two-column book is one of FineBooks' six books, so FineBooks did score it | Full blog text | **Reword claim 2** |
| R23 finebooks2026a | Yes | Yes (1708–1913, four languages, 2,165 pages, six volumes, IMPACT + BHL-Europe, CC-BY 3.0) | Dataset card (full) | None |
| R24 poznanski2025a | Yes | Yes (7,010; test types) | Full text (arXiv HTML) | None |
| R25 marti2002 | Yes (Crossref) | Yes (identification only) | Crossref metadata and title; Springer page blocked automated access (JS challenge), not read | None |
| R26 teklia2024 | Yes | Yes (2,915 test lines, 128 px, MIT, English) | Dataset card (full) | None (IAM's own terms seen only in a search snippet, because the IAM site's TLS certificate has expired) |
| R27 finebooks2026b | Yes | Partly. The nine values and the 0.77 reproduce, but the leaderboard (updated 2026-10-30) now includes Chandra OCR 2, so ten engines overlap. Two statements in Details C are out of date or wrong | Full Space page source, including its embedded data table | **Reword three sentences** |
| R28 kwon2023 | Yes (Crossref) | Yes (identification) | Abstract (arXiv API) + Crossref | None |
| R29 dao2023 | Yes (venue ICLR 2024 taken from the FA README citation; OpenReview blocked) | **No.** The paper never mentions Turing or the T4. vLLM's default is not in it | Full text (arXiv HTML) | **Reword and re-cite:** flash-attention README + vLLM attention-backend docs |
| R30 jax2026 | Yes | Partly. The page supports JAX recompiling for each new shape. That the TPU software is JAX-based needs R37 | Full page | Small rewording (optional but recommended) |
| R31 google2026b | Yes | Yes ("$9.99 for 100 Compute Units") | Page source + its JS bundle (cards are client-rendered) | None |
| R32 mccormick2024 | Yes | Yes for all four rates and for G4 = RTX PRO 6000 Blackwell. "Server Edition" is **not** in this source | Full page | Check where "Server Edition" comes from (probably nvidia-smi); if not from a log, drop it |
| R33 nvidia2026 | Yes | Yes (12.0 for RTX PRO 6000 Blackwell Server Edition) | Full page | None |
| R34 nvidia2025 | Yes | Partly. 12.8 "adds compiler support" for SM_120. "Needs 12.8 or later" goes further: NVIDIA's guide says older builds can still run through PTX | Full page + Blackwell Compatibility Guide | Small rewording |
| R35 paddlepaddle2026d | Yes | Yes, with a caveat: the guide covers Blackwell GPUs in general (sm120 images) but was verified only on an RTX 5070 | Full page | Optional small rewording |
| R36 vllm2026a | Yes | Yes | Full page | None |
| R37 vllm2026b | Yes (repo now says it is moving to vllm-project/vllm-jax; matrix "Last Updated: 2026-10-22") | Yes (Gemma 4 E4B passing in release; Qwen3.5-9B and Qwen3-VL-8B untested) | README (full) | None (consider noting the access date) |

---

## R19. qwen2026: Qwen3.5 blog

**Metadata.** The blog is rendered by JavaScript. I read the full article through the site's own article API (`https://qwen.ai/api/v2/article/?language=en-US&path=qwen3.5&type=qwen_ai`), which the page itself calls. Title: "Qwen3.5: Towards Native Multimodal Agents". The page shows "February 14, 2026 · … · Qwen Team", and the API metadata gives `date: 2026-02-16`. Its own citation block reads: `title = {Qwen3.5: Towards Native Multimodal Agents} , url = {https://qwen.ai/blog?id=qwen3.5} , author = {Qwen Team} , month = {February} , year = {2026}`. The reference is correct.

**Support.** Row: "| Qwen3.5 2B and 9B ([Qwen 2026](#references)) | 2.3, 9.7 | General |"
- "General": yes. The blog says: "As a native vision-language model, Qwen3.5-397B-A17B demonstrates outstanding results across a full range of benchmark evaluations, including reasoning, coding, agent capabilities, and multimodal understanding".
- "2B and 9B": **no**. The only Qwen3.5 model names in the blog are Qwen3.5-397B-A17B (8 times) and Qwen3.5-Plus (2 times). The blog announces only the 397B model.
- The sizes are supported by the model cards. Qwen3.5-2B card (huggingface.co/Qwen/Qwen3.5-2B): "Type: Causal Language Model with Vision Encoder" … "Number of Parameters: 2B". Qwen3.5-9B card (huggingface.co/Qwen/Qwen3.5-9B): "Type: Causal Language Model with Vision Encoder" … "Number of Parameters: 9B". Both cards say "For more details, please refer to our blog post [Qwen3.5](https://qwen.ai/blog?id=qwen3.5)" and cite it. In the HF safetensors metadata, the totals are 2,274,069,824 (2.3B) and 9,653,104,368 (9.7B). These match the table.

**Evidence.** Full blog text, both model cards, HF API metadata.

**Action.** Cite the cards for the sizes. Verified URLs: https://huggingface.co/Qwen/Qwen3.5-2B and https://huggingface.co/Qwen/Qwen3.5-9B (both Apache-2.0, created 2026-02-28 and 2026-02-27).

## R20. gemma2026: Gemma 4 Technical Report

**Metadata.** arXiv 2607.02770, v1 2 Jul 2026, v2 24 Jul 2026. The first authors are Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon, which matches the reference.

**Support.** The report covers E4B, and 8 is its total parameter count: "The Gemma 4 family of models comprises dense architectures, with effective 2.3B ( E2B ), effective 4.5B ( E4B ), 12B and 31B parameters … E2B and E4B use per-layer embeddings as in Gemma 3n … making them 2.3B and 4.5B effective out of 5B and 8B total parameters respectively." For "General", the abstract says: "We introduce Gemma 4, a new generation of open-weight, natively multimodal language models". (The HF metadata for google/gemma-4-E4B-it gives 7,996,156,490 parameters.)

**Evidence.** Full text (arXiv HTML).

## R21. bai2025a: Qwen2.5-VL Technical Report

**Metadata.** arXiv 2502.13923, 19 Feb 2025. The first ten authors match.

**Support.** "General": the abstract says "We introduce Qwen2.5-VL, the latest flagship model of Qwen vision-language series". On 3B and 7B, the paper says: "The smaller Qwen2.5-VL-7B and Qwen2.5-VL-3B models outperform comparable competitors". Table 1 has columns "Qwen2.5-VL-3B Qwen2.5-VL-7B Qwen2.5-VL-72B". The paper does not state 3.8 or 8.3. In HF safetensors metadata, Qwen2.5-VL-3B-Instruct has 3,754,622,976 parameters (3.8B) and Qwen2.5-VL-7B-Instruct has 8,292,166,656 (8.3B). These match the table.

**Evidence.** Full text (arXiv HTML) + HF metadata.

## R22. majstorovic2026: FineBooks blog

**Metadata.** The page shows "Community Article Published August 10, 2026" by Sebastian Majstorovic and Daniel van Strien, under the finebooks org. The title matches.

**Claim 1.** "**Old books:** 50 pages of natural-history books printed between 1708 and 1913, in English, French, German, and Latin, from the FineBooks ground truth (…)". Supported jointly with R23. Blog: "the IMPACT project and BHL-Europe produced expert-corrected transcriptions of six BHL volumes: 2,165 pages in English, French, German and Latin". The blog does not give 1708–1913; the dataset card does (R23).

**Claim 2.** "FineBooks did not evaluate multi-column layouts ([Majstorovic & van Strien 2026](#references)), so its leaderboard does not show this effect." **Partly.**
- The blog says: "Books are visually more uniform than newspapers, magazines, or archival materials: they mostly contain single-column running text with consistent layouts and repeating furniture. Multi-column layouts, complex page elements, and other document types present different technical challenges, with different failure modes, and the current leaderboard does not measure them."
- But the two-column book in the post, *Conchologia Iconica*, is one of FineBooks' six volumes. The dataset card lists "Conchologia iconica … vol. 5 | Reeve | 1849 | English | 648" pages, which is 30% of the 2,165. The leaderboard scores every page. So FineBooks did score these pages; it just does not measure multi-column layout as a separate quantity. The FineBooks eval repo's RESULTS.md (github.com/finebooks/bhl-ocr-eval) also says: "Conchologia, Birds and the Russian entomological volume have list pages printed in two or three columns".
- The leaderboard reports CER by language, not by book ("click a row for per-language CER"). So "its leaderboard does not show this effect" holds only in the sense that no per-book score is shown.

**Proposed rewording.** See the final list.

**Evidence.** Full blog text.

## R23. finebooks2026a: bhl-impact-gt dataset card

**Metadata.** Card title "FineBooks BHL IMPACT Ground Truth", licence `cc-by-3.0`. Created 2026-06-30, last modified 2026-08-10. The URL resolves.

**Support.**
- Card: "**2,165 page scans from six historical natural-history books, each paired with an expert, ~99.95%-accurate transcription**"
- "| **Books** | 6 natural-history titles, 1708–1913 |" and "| **Languages** | English · French · German · Latin |"
- On IMPACT and BHL-Europe: "produced in the 2011–2012 collaboration between the EU IMPACT digitisation project and **BHL-Europe**, and released under CC-BY 3.0". This also supports the uncited licence.
- The post also says the reference keeps the long s and ligatures. The card says: "Text is transcribed exactly as printed … Nothing is normalized".
- The post's volume list (Details A) matches the card's source-book table. The card gives the Russian volume's title as "Horae Societatis Entomologicae Rossicae…", with Internet Archive id `trudyrusskagoent161881russ`. The post's *Trudy Russkago entomologicheskago obshchestva* is the Russian form of the same title, so it is acceptable.

**Evidence.** Full card.

## R24. poznanski2025a: olmOCR

**Metadata.** arXiv 2502.18443. The authors match. (The arXiv listing repeats "Aman Rangapur" in 7th place by mistake; the bib correctly lists him once.)

**Support.**
- "The full benchmark has 7,010 tests". The paper says: "olmOCR-Bench comprises 1,402 distinct PDF documents derived from diverse source repositories, covered by 7,010 unique test cases." In Table 3, "Total Tests 721 823 1,061 1,020 3,385 7,010".
- Test examples. "Text Presence : Verifies that a text segment (typically spanning 1-3 sentences) is correctly identified within the plain text output." "Text Absence : … This category primarily targets peripheral content such as recurring headers, footers, and pagination markers." "Table Accuracy : Checks that the plain text output contains a table with a cell with a given value, and that its neighboring cells have certain properties. For instance, one can validate this page has a table with a cell containing “4.5%” and above that is a cell containing “2.4%”." "Under the right heading" is a fair plain reading of that test.

**Evidence.** Full text (arXiv HTML).

## R25. marti2002: IAM database

**Metadata.** Crossref: "The IAM-database: an English sentence database for offline handwriting recognition", U.-V. Marti and H. Bunke, *International Journal on Document Analysis and Recognition* 5(1):39–46, published 2002-11-01. This is correct.

**Support.** The sentence only identifies the database as a source of English handwriting. The title supports that. The Teklia card (R26) adds: "The IAM Handwriting Database contains forms of handwritten English text which can be used to train and test handwritten text recognizers".

**Evidence.** Crossref metadata and title only. The Springer page returned a JavaScript "Client Challenge", so I did not read the abstract. Semantic Scholar says the publisher has elided the abstract.

## R26. teklia2024: Teklia/IAM-line

**Metadata.** Created 2024-01-12 by Teklia. Card title "IAM - line level", licence `mit`.

**Support.**
- 2,915: "- name: test / num_examples: 2915". The datasets-server API confirms the splits train 6,482, validation 976, test 2,915.
- 128 pixels: "Note that all images are resized to a fixed height of 128 pixels."
- English: "All the documents in the dataset are written in English."
- IAM's own research terms. These are uncited. I could not open fki.tic.heia-fr.ch because its TLS certificate has expired. A web-search summary of that page says the database "may be used for non-commercial research purpose only". This is secondary evidence only.

**Evidence.** Full card.

## R27. finebooks2026b: BHL OCR Leaderboard Space

**Metadata.** Static Space "BHL OCR Leaderboard", created 2026-08-05 and last modified 2026-10-30. Its embedded data header: `PROV={"norm": "2026-07-20a", … "full_pages": 2165, "baked": "2026-10-30"}`.

**The nine values.** These are 1 − reading CER from the Space's embedded data: dots.mocr 0.0237 (97.6), OvisOCR2 0.0305 (97.0), PaddleOCR-VL-1.6 0.0392 (96.1), olmOCR-2 0.0432 (95.7), LightOnOCR-2 0.0489 (95.1), GLM-OCR 0.0491 (95.1), Qwen3.5-9B 0.0507 (94.9), DeepSeek-OCR-2 0.0620 (93.8), Tesseract 5 0.0642 (93.6). All nine match Details C. (The blog's table gives OvisOCR2 96.9%, and its 93.8% is DeepSeek-OCR version 1, not version 2. Citing the Space for these values is correct.)

**0.77.** Spearman on the Space's CERs against the unrounded book accuracies in `figures/data.json` gives **0.767** for these nine. It reproduces.

**What changed.** The Space now ranks 26 models, and **Chandra OCR 2** (datalab-to/chandra-ocr-2) is among them: CER 0.0285, 97.15%, rank 3. This benchmark also tested Chandra OCR 2 (books 98.0%). So **ten** engines are in both studies. Surya is on the board only as "Surya OCR 2.5", not the Surya OCR 2 tested here. With Chandra added, Spearman is **0.818** (my computation, using the same data).

**Current top of the Space.** 1 dots.ocr 0.0235 [0.0156, 0.0318]; 2 dots.mocr 0.0237 [0.0143, 0.0334]; 3 Chandra-OCR-2 0.0285 [0.0198, 0.038]; 4 OvisOCR2 0.0305; 5 LightOnOCR-3-4B 0.0312; 6 Unlimited-OCR-2 0.0329; 7 kraken PP-OCRv6 0.0338. The Space says: "Where 95% bootstrap CIs overlap, differences are within noise — that overlap is the finding, not a defect." The top three intervals overlap. So the Details C sentence "dots.ocr and dots.mocr are statistically tied, followed by OvisOCR2 and a kraken pipeline with PP-OCRv6" is **out of date**. (The GitHub RESULTS.md is a draft re-scored under norm 2026-10-30a and not yet on the Space. It puts dots.mocr, chandra-ocr-2 and dots.ocr first and says "the top three … are a statistical tie". It also changes several of the nine values, for example olmOCR-2 to 0.0279. If the Space is rebaked, Details C will need updating again.)

**Other Details C statements.**
- "FineBooks reports looped pages separately". Yes: "looped (hit its token cap mid repetition loop — excluded from the row's aggregates and counted in Loop %)".
- "on which it reports models differ most". Yes: "Sparse and blank pages are where models diverge most".
- "does not score speed". Yes. The Space has no speed column, and RESULTS.md says "A note on throughput, which this board deliberately does not score."
- "excludes handwriting". Yes (blog).
- "excludes … multi-column layouts". Partly; see R22.
- "dots.mocr scored best in the FineBooks blog post". Yes: the blog's table is "ordered from highest to lowest accuracy", and its first row is "dots.mocr 3B 97.6%".

**DeepSeek-OCR 2 explanation.** Line 113 says "and from DeepSeek-OCR 2 on the two-column book, which FineBooks did not evaluate". This is **contradicted** by R23: that book is in FineBooks and is scored. On the Space, DeepSeek-OCR-2's English CER is 0.0493, so it does not collapse on FineBooks' English pages. This study cannot attribute the gap to FineBooks not having the book.

**Evidence.** Full Space page source, including the embedded data table. Also the eval repo's RESULTS.md on GitHub.

## R28. kwon2023: vLLM / PagedAttention

**Metadata.** Crossref: SOSP '23 (Proceedings of the 29th Symposium on Operating Systems Principles), pages 611–626. The authors match.

**Support (identification).** Abstract: "we propose PagedAttention … On top of it, we build vLLM, an LLM serving system". (vLLM 0.30.0 exists on PyPI; the latest is 0.33.0.)

**Evidence.** Abstract + Crossref.

## R29. dao2023: FlashAttention-2 (flagged)

**Metadata.** arXiv 2307.08691, Tri Dao, 17 Jul 2023. The flash-attention README's own citation gives `booktitle={International Conference on Learning Representations (ICLR)}, year={2024}`. OpenReview returned a challenge page, so I did not use it.

**Support.** "…and FlashAttention-2, the attention code that the serving software vLLM ([Kwon et al 2023](#references)) uses by default on newer cards, does not support it ([Dao 2023](#references))." **No.** The paper's full text contains no mention of Turing, the T4, or RTX 20-series GPUs; it benchmarks on the A100 and H100. Nor does it say anything about vLLM.

**Sources that do support it.**
- Flash-attention README, https://github.com/Dao-AILab/flash-attention (verified): "FlashAttention-2 with CUDA currently supports: 1. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). For Turing GPUs (T4, RTX 2080), see the separate flash-attention-turing repo, which supports a core subset of FlashAttention features on Turing. 2. Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs)."
- vLLM "Attention Backend Feature Support", https://docs.vllm.ai/en/latest/design/attention_backends/ (verified; page title "Attention Backend Feature Support - vLLM"):
  - "When no backend is explicitly selected, vLLM chooses the first compatible backend from these priority-ordered lists." "Ampere/Hopper (SM 8.x-9.x): Priority Backend 1 FLASH_ATTN …"
  - In the backend table, the row "FLASH_ATTN FA2* … ≥8.0" gives compute capability ≥8.0 for FA2.
  - "Default is FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), FA2 otherwise."
  - So FA2 is the default on the A100 (8.0) and L4 (8.9), but not on Hopper (FA3). On Blackwell, FlashInfer is tried first. "Newer cards" in general is too broad.
  - NVIDIA's table (R33) gives the T4 compute capability 7.5, below FA2's ≥8.0. (This is the "latest" docs page, not one pinned to 0.30.)

**Proposed rewording.** See the final list. Keep Dao 2023 only to name FlashAttention-2.

**Evidence.** Full text of the paper; READMEs and docs read in full.

## R30. jax2026: JAX Sharp Bits

**Metadata.** Page title "🔪 JAX - The Sharp Bits 🔪", JAX documentation. It is undated; the year is the access year.

**Support.** "The TPU software compiles a separate program for each new input shape ([JAX 2026](#references))…" **Partly.**
- The page supports the JAX mechanism: "# Subsequent runs with parameters of same type and shape may not show the side-effect / # This is because JAX now invokes a cached compilation of the function … # JAX re-runs the Python function when the type or shape of the argument changes".
- The AOT page is a clearer statement: "Compiled functions are specialized to a particular set of argument “types,” such as arrays with a specific shape and element type" (https://docs.jax.dev/en/latest/aot.html, verified).
- The page says nothing about vllm-tpu. That the TPU backend is JAX-based comes from the tpu-inference README (R37): "an expressive and powerful new hardware plugin unifying JAX and PyTorch under a single lowering path", and "`tpu-inference` is now vLLM JAX (`vllm-jax`) and is focused on serving JAX models on TPU with vLLM".

**Evidence.** Full page.

## R31. google2026b: Colab pricing

**Metadata.** The `<title>` is "Colab Paid Services Pricing". This is correct.

**Support.** The plan cards are a web component, so the visible text is not in the static HTML. The page passes `ccuProductTitle="Pay As You Go" … formattedCCU100Price="$9.99" formattedCCU500Price="$49.99"`. The page's script (signup_binary.js) builds the button label as `a.formattedCCU100Price+" for 100 Compute Units"`, which gives "$9.99 for 100 Compute Units". The FAQ in the same script says: "The amount of access a compute unit gives you depends on the configuration of your VM and use of specialized accelerators like TPUs or GPUs". This fits the uncited "does not publish a rate per runtime".

**Evidence.** Page source + its script. Prices are US prices, as served to this machine.

## R32. mccormick2024: Colab GPU pricing (third party)

**Metadata.** "Colab GPUs Features & Pricing", "23 Apr 2024 Updated March 2026". This is correct.

**Support.**
- Table: "GPU Units/hr $/hr Time (h:m) Date Checked T4 1.19 $0.12 84:02 2026-03-10 L4 1.71 $0.17 58:29 2026-03-10 A100 (40GB) 5.40 $0.54 18:31 2026-03-10 … G4 8.71 $0.87 11:29 2026-03-10". All four rates match.
- On third-party measurement: "Here’s what I’m seeing as of March 2026, in southern California".
- On the G4: "The “G4” is actually an NVIDIA RTX PRO 6000 (Blackwell architecture) with a whopping ~96 GB of VRAM." The words **"Server Edition" are not in this source.** If the machines table took them from nvidia-smi in the benchmark logs, that is fine (NVIDIA's page lists a "Server Edition" at 12.0). Otherwise use "RTX PRO 6000 (Blackwell)".

**Evidence.** Full page.

## R33. nvidia2026: CUDA GPU compute capability

**Support.** The page reads: "Compute Capability … 12.0 NVIDIA RTX PRO 6000 Blackwell Server Edition NVIDIA RTX PRO 4500 Blackwell Server Edition NVIDIA RTX PRO 6000 Blackwell Workstation Edition …". It also lists "7.5 NVIDIA T4" and "8.9 NVIDIA L4", which is useful for R29.

**Evidence.** Full page.

## R34. nvidia2025: CUDA 12.8 release notes

**Metadata.** Page title "1. CUDA 12.8 Release Notes". This is correct.

**Support.** "…which needs CUDA 12.8 or later ([NVIDIA 2025](#references))". **Partly.**
- The notes say: "1.2. New Features This release adds compiler support for the following Nvidia Blackwell GPU architectures: SM_100 SM_101 SM_120". The 12.6 release notes do not mention SM_120.
- "Needs 12.8 or later" goes further than this. NVIDIA's Blackwell Compatibility Guide (https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html, verified) says older builds can still run: "Unlike cubin, PTX is forward-compatible. Meaning PTX is supported to run on any GPU with compute capability higher than the compute capability assumed for generation". It also says: "The nvcc compiler included with version 12.7 or earlier (11.8-12.7) of the CUDA Toolkit can generate cubins native to the NVIDIA Hopper GPU architectures (compute capability 9.x)."
- So CUDA 12.8 was the first release to add support. It is not strictly true that nothing older can run.

**Evidence.** Full page + the compatibility guide.

## R35. paddlepaddle2026d: PaddleOCR-VL Blackwell tutorial

**Metadata.** Title "PaddleOCR-VL NVIDIA Blackwell-Architecture GPUs Usage Tutorial - PaddleOCR Documentation". This is correct.

**Support.** "The CUDA 12.9 build, which Paddle publishes for these GPUs ([PaddlePaddle 2026](#references))…" **Yes, with a caveat.**
- "NVIDIA Blackwell-architecture GPUs include, but are not limited to: RTX 5090 RTX 5080 RTX 5070、RTX 5070 Ti RTX 5060、RTX 5060 Ti RTX 5050"
- "# Note that PaddlePaddle for cu129 is being installed here python -m pip install paddlepaddle-gpu == 3 .2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/"
- The Docker images are tagged `latest-nvidia-gpu-sm120`, which is the RTX PRO 6000's architecture.
- Caveat: "PaddleOCR-VL has been verified for accuracy and speed on the RTX 5070. However, due to hardware diversity, compatibility with other NVIDIA Blackwell-architecture GPUs has not yet been confirmed."
- "These GPUs", meaning Blackwell GPUs, is a fair reading.

**Evidence.** Full page.

## R36. vllm2026a: automatic prefix caching (design doc)

**Support.**
- Design page: "Prefix caching kv-cache blocks is a popular optimization in LLM inference to avoid redundant prompt computations. The core idea is simple – we cache the kv-cache blocks of processed requests, and reuse these blocks when a new request comes in with the same prefix as previous requests."
- On images: "The challenge for prefix caching to support this case is we need to differentiate images from the placeholders. To address this problem, we encode the image hash generated by the frontend image processor."
- The feature page adds: "allowing the new query to skip the computation of the shared part".
- This supports "reuses stored computation for a prompt already seen, including its image". "Skipped most of the work" is the benchmark's own observation.

**Evidence.** Full page.

## R37. vllm2026b: tpu-inference README

**Metadata.** The repo vllm-project/tpu-inference still resolves. Its description is now "Moving to vllm-project/vllm-jax. TPU inference for vLLM, with unified JAX and PyTorch support." The README says "`tpu-inference` is now vLLM JAX (`vllm-jax`)". The support matrix reads "Last Updated: 2026-10-22 06:52 AM UTC", not the 2026-08-27 in the earlier notes.

**Support.**
- Release matrix rows:
  - "google/gemma-4-E4B-it | Multimodal | ✅ | ✅ | ✅" (Passing on Unit, Correctness and Performance)
  - "Qwen/Qwen3-VL-8B-Instruct | Multimodal | ❓ | ❓ | ❓"
  - "Qwen/Qwen3.5-9B | Multimodal | ❓ | ❓ | ❓"
  - Legend: "❓ **Untested:** The functionality exists but has not been recently or thoroughly verified."
  - Claim 2 is supported as written ("passing its release tests"). In the nightly matrix, E4B is "✅ Passing | ❌ Failing | ❓ Untested". "Release" is therefore the right qualifier.
- Claim 1 (Details D row): none of the OCR-specific VLMs in this benchmark is in the matrix. The only OCR model listed is deepseek-ai/DeepSeek-OCR (version 1, all ❓). Qwen3.5-2B is not listed. Qwen/Qwen2.5-VL-7B-Instruct is listed (✅ ✅ ❓). "Not supported or not tested by the TPU backend" is a fair summary, reading "not listed" as not tested.

**Evidence.** Full README.

---

## Proposed rewordings (exact current text, then replacement)

**1. R19, engine table row**
- Current: `| Qwen3.5 2B and 9B ([Qwen 2026](#references)) | 2.3, 9.7 | General |`
- Replace: `| Qwen3.5 2B and 9B ([Qwen 2026](#references); [Qwen 2026a](#references); [Qwen 2026b](#references)) | 2.3, 9.7 | General |`
- New references:
  - Qwen (2026a). "Qwen3.5-2B". *Hugging Face model card*. [huggingface.co/Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
  - Qwen (2026b). "Qwen3.5-9B". *Hugging Face model card*. [huggingface.co/Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
  - Keep the blog as the source for "General". If you prefer one fewer reference, cite only the two cards; both describe the model as "Causal Language Model with Vision Encoder".

**2. R22 claim 2 (section 6)**
- Current: "FineBooks did not evaluate multi-column layouts ([Majstorovic & van Strien 2026](#references)), so its leaderboard does not show this effect."
- Replace: "This book is also one of the six FineBooks books, but FineBooks says its leaderboard does not measure multi-column layouts ([Majstorovic & van Strien 2026](#references)), and the leaderboard gives scores by language, not by book ([FineBooks 2026b](#references)), so it does not show this effect on its own."

**3. R27, "Agreement with FineBooks" (section 6)**
- Current: "On the nine engines both studies tested, the order on books agrees closely with the FineBooks leaderboard ([FineBooks 2026b](#references)), with a rank correlation of 0.77. The two large differences come from LightOnOCR-2's repetition, which FineBooks reports separately instead of counting as errors, and from DeepSeek-OCR 2 on the two-column book, which FineBooks did not evaluate."
- Replace: "On the ten engines both studies tested, the order on books agrees closely with the FineBooks leaderboard ([FineBooks 2026b](#references)), with a rank correlation of 0.82. The two large differences come from LightOnOCR-2's repetition, which FineBooks reports separately instead of counting as errors, and from DeepSeek-OCR 2 on the two-column book. That book is also in the FineBooks set, so this study does not explain the second difference."
- The 0.82 is my Spearman of the Space's reading CERs against `figures/data.json` books accuracy, with Chandra OCR 2 added; please recompute with the project code. If you would rather keep nine engines, write "On nine of the ten engines both studies tested … 0.77" and say that Chandra OCR 2 was added to the leaderboard later. Either way, the clause "which FineBooks did not evaluate" must go.

**4. R27, Details C**
- Current: "dots.mocr scored best in the FineBooks blog post; on the later leaderboard, dots.ocr and dots.mocr are statistically tied, followed by OvisOCR2 and a kraken pipeline with PP-OCRv6. FineBooks does not score speed and excludes handwriting and multi-column layouts."
- Replace: "dots.mocr scored best in the FineBooks blog post. On the leaderboard as updated on 30 October 2026, dots.ocr, dots.mocr, and Chandra OCR 2 are first to third, and their 95% intervals overlap, which the leaderboard treats as a tie. FineBooks does not score speed or handwriting, and says its leaderboard does not measure multi-column layouts."
- Also add a Chandra OCR 2 row to the comparison table: FineBooks reading CER 0.0285 (97.2 rounded half up; 97.15 unrounded), and this study's books score, 98.0 (97.0–98.9) per `figures/data.json`.

**5. R29 (section 7, T4 bullet)**
- Current: "It lacks bfloat16, a number format some of these models need, and FlashAttention-2, the attention code that the serving software vLLM ([Kwon et al 2023](#references)) uses by default on newer cards, does not support it ([Dao 2023](#references))."
- Replace: "It lacks bfloat16, a number format some of these models need, and FlashAttention-2 ([Dao 2023](#references)), the attention code that the serving software vLLM ([Kwon et al 2023](#references)) uses by default on the A100 and L4, does not support it ([Dao-AILab 2026](#references); [vLLM 2026c](#references))."
- New references:
  - Dao-AILab (2026). "FlashAttention". *GitHub repository*. [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
  - vLLM (2026c). "Attention Backend Feature Support". *vLLM documentation*. [docs.vllm.ai/en/latest/design/attention_backends/](https://docs.vllm.ai/en/latest/design/attention_backends/)

**6. R30 (section 7, TPU bullet)**
- Current: "The TPU software compiles a separate program for each new input shape ([JAX 2026](#references)), and every handwritten line has a different width."
- Replace: "The TPU software is built on JAX ([vLLM 2026b](#references)), which compiles a separate program for each new input shape ([JAX 2026](#references)), and every handwritten line has a different width."

**7. R34 (Details D, PaddleOCR on the G4)**
- Current: "The RTX PRO 6000 is a Blackwell GPU of compute capability 12.0 ([NVIDIA 2026](#references)), which needs CUDA 12.8 or later ([NVIDIA 2025](#references)), and the benchmark had installed PaddlePaddle's CUDA 12.6 build."
- Replace: "The RTX PRO 6000 is a Blackwell GPU of compute capability 12.0 ([NVIDIA 2026](#references)), which CUDA first supported in version 12.8 ([NVIDIA 2025](#references)), and the benchmark had installed PaddlePaddle's CUDA 12.6 build."

**8. R35 (optional, same bullet)**
- Current: "The CUDA 12.9 build, which Paddle publishes for these GPUs ([PaddlePaddle 2026](#references)), is now installed on any GPU of compute capability 10 or higher; the rerun scored 84.3% on books, matching the other runtimes."
- Replace (optional): "The CUDA 12.9 build, which Paddle's guide for Blackwell GPUs installs ([PaddlePaddle 2026](#references)), is now installed on any GPU of compute capability 10 or higher; the rerun scored 84.3% on books, matching the other runtimes."

**9. R32 (Details B machines table, uncited)**
- Current: "| G4 | RTX PRO 6000 Blackwell Server Edition | 95.6 GB | 48 | 8.71 | 0.870 |"
- Keep it if "Server Edition" and 95.6 GB come from the benchmark's own nvidia-smi log. Otherwise replace with: "| G4 | RTX PRO 6000 (Blackwell) | 95.6 GB | 48 | 8.71 | 0.870 |". McCormick says only "NVIDIA RTX PRO 6000 (Blackwell architecture) with … ~96 GB".
