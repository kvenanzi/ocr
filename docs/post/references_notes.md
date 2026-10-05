# References notes: "OCR benchmark"

Verified 2026-10-05. Each entry: what the source says that matters for the post, then the formatted reference line. Keys match `references.bib`. Where two sources share surname and year, the letter appears in the key and in the reference line's year, e.g. (2025a). The bib `year` field stays plain.

## Classic OCR

### smith2007
Ray Smith's overview of Tesseract (originally HP, open-sourced 2005, then Google): describes its line finding, features/classification and adaptive classifier. Crossref metadata: ICDAR 2007 proceedings vol. 2, pp. 629–633, IEEE. The benchmark ran Tesseract 5.3.4 (modern Tesseract 4+/5 uses an LSTM recogniser, which this 2007 paper predates).
- Smith R (2007). "An Overview of the Tesseract OCR Engine". *Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)*, vol. 2, pp. 629–633. [doi:10.1109/ICDAR.2007.4376991](https://doi.org/10.1109/ICDAR.2007.4376991)

### barlow2026
OCRmyPDF "adds an OCR text layer to scanned PDF files, allowing them to be searched or copy-pasted"; it drives Tesseract and produces PDF/A by default. Copyright holder/maintainer James R. Barlow (README SPDX header; PyPI author). MPL-2.0. Repo created 2013; the benchmark required ocrmypdf>=17 (v17.0.0 released 2026-01-30; latest v17.13.0 on 2026-09-28).
- Barlow JR (2026). "OCRmyPDF". *GitHub repository* (v17). [github.com/ocrmypdf/OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)

### du2020
Original PP-OCR paper (Baidu): ultra-lightweight 3-stage system (DB text detector, direction classifier, CRNN recogniser) with "a bag of strategies"; 3.5 MB model for 6,622 Chinese characters, 2.8 MB for 63 alphanumeric symbols; detector trained on 97K images, classifier on 600K, recogniser on 17.9M.
- Du Y, Li C, Guo R, Yin X, Liu W, Zhou J, Bai Y, Yu Z, Yang Y, Dang Q, Wang H (2020). "PP-OCR: A Practical Ultra Lightweight OCR System". *arXiv preprint arXiv:2009.09941*. [arxiv.org/abs/2009.09941](https://arxiv.org/abs/2009.09941)

### cui2025a
PaddleOCR 3.0 technical report (submitted 2025-07-08): Apache-licensed toolkit introducing PP-OCRv5 (multilingual text recognition), PP-StructureV3 (hierarchical document parsing) and PP-ChatOCRv4 (key information extraction); claims these <100M-parameter models rival billion-parameter VLMs. Cited as Cui et al 2025a (PaddleOCR-VL is 2025b).
- Cui C, Sun T, Lin M, Gao T, Zhang Y, Liu J, Wang X, Zhang Z, Zhou C, Liu H, et al (2025a). "PaddleOCR 3.0 Technical Report". *arXiv preprint arXiv:2507.05595*. [arxiv.org/abs/2507.05595](https://arxiv.org/abs/2507.05595)

### zhang2026b
PP-OCRv6 paper (submitted 2026-06-11, same day as the PaddleOCR v3.7.0 release; cited on the HF model cards). PP-OCRv6 is the newest PP-OCR generation, built on a new PPLCNetV4 backbone (MetaFormer-style blocks with structural reparameterisation), in three tiers: tiny 1.5M / small 7.7M / medium 34.5M parameters (det+rec combined). Medium and small cover 50 languages (Simplified + Traditional Chinese, English, Japanese, 46 Latin-script languages); tiny covers 49 (no Japanese). Official docs claim PP-OCRv6_medium: 86.2% detection Hmean and 83.2% recognition accuracy on Baidu's in-house benchmark, i.e. +4.6% det Hmean and +5.1% rec accuracy over PP-OCRv5_server, and claim to surpass Qwen3-VL-235B and GPT-5.5 on their OCR benchmark. HF card for PP-OCRv6_medium_rec: ~19M params, LCNetV4 backbone + EncoderWithLightSVTR neck + CTC/NRTR multi-head decoder; 91.5% printed Chinese, 94.1% printed English, 90.5% Japanese (vendor-reported).
- Zhang Y, Wang X, Lin M, Zhang Y, Deng P, Sun T, Gao T, Zhang Z, Liu J, Zhou C, et al (2026b). "PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks". *arXiv preprint arXiv:2606.13108*. [arxiv.org/abs/2606.13108](https://arxiv.org/abs/2606.13108)

### paddlepaddle2026a
PaddleOCR v3.7.0 release notes (2026-06-11): "Release PP-OCRv6" -- medium tier +4.6% detection and +5.1% recognition over PP-OCRv5_server with only 34.5M parameters; single model for 50 languages; 5.2x CPU speedup (OpenVINO), 6.1x on Apple M4 (tiny), 0.13 s on A100; tiers tiny 1.5M / small 7.7M / medium 34.5M. This is the release the benchmark used (paddleocr==3.7.0, models PP-OCRv6_medium_det / PP-OCRv6_medium_rec, both Apache-2.0 on HF, created 2026-06-10). Docs page: https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html
- PaddlePaddle (2026a). "PaddleOCR v3.7.0 release notes: Release PP-OCRv6". *GitHub release*. [github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0)

### mindee2021
docTR (Mindee; now maintained by t2k GmbH): two-stage OCR library (text detection + recognition) in PyTorch; README lists implemented architectures including FAST (detection) and PARSeq (recognition), alongside DBNet, LinkNet, CRNN, SAR, MASTER, ViTSTR, VIPTR. Apache-2.0. The project's own recommended citation is `@misc{doctr2021, author={Mindee}, year={2021}}`; benchmark used python-doctr >=1.1 (v1.1.0 released 2026-08-21) with fast_base + parseq.
- Mindee (2021). "docTR: Document Text Recognition". *GitHub repository*. [github.com/mindee/doctr](https://github.com/mindee/doctr)

### chen2021
FAST text detector: "minimalist kernel representation (only has 1-channel output)" for arbitrarily-shaped text plus GPU-parallel post-processing and a text-detection-specific NAS backbone; FAST-T reaches 81.6% F-measure at 152 FPS on Total-Text (600+ FPS with TensorRT). arXiv only (v1 2021-11-03, v2 2023-01-11); no venue listed on the arXiv page. Used as docTR's `fast_base` detector.
- Chen Z, Wang J, Wang W, Chen G, Xie E, Luo P, Lu T (2021). "FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation". *arXiv preprint arXiv:2111.02394*. [arxiv.org/abs/2111.02394](https://arxiv.org/abs/2111.02394)

### bautista2022
PARSeq: permutation language modelling learns an ensemble of internal autoregressive LMs with shared weights, unifying context-free non-AR and context-aware AR inference with iterative refinement; 91.9% word accuracy on STR benchmarks trained on synthetic data, 96.0% trained on real data. ECCV 2022, LNCS pp. 178–196 (Crossref); volume 13688 inferred from the ECVA PDF filename 136880177.pdf. Used as docTR's `parseq` recogniser.
- Bautista D, Atienza R (2022). "Scene Text Recognition with Permuted Autoregressive Sequence Models". *ECCV 2022 (LNCS 13688), pp. 178–196*. [arxiv.org/abs/2207.06966](https://arxiv.org/abs/2207.06966) ([doi:10.1007/978-3-031-19815-1_11](https://doi.org/10.1007/978-3-031-19815-1_11))

### jaidedai2024
EasyOCR: "Ready-to-use OCR with 80+ supported languages". README: detection uses CRAFT (official clovaai implementation and pretrained model); recognition is a CRNN (ResNet/VGG feature extractor, LSTM, CTC decoding) trained with a modified clovaai deep-text-recognition-benchmark. Apache-2.0. Latest release v1.7.2 (2024-09-24), which the benchmark used.
- JaidedAI (2024). "EasyOCR". *GitHub repository* (v1.7.2). [github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)

### baek2019
CRAFT (Clova AI / NAVER): detects text by predicting per-character region scores and inter-character affinity scores, using character-level pseudo-labels for real images produced by an interim model; strong on curved text (TotalText, CTW-1500). Crossref (IEEE) pages 9357–9366; CVF open-access version lists pp. 9365–9374.
- Baek Y, Lee B, Han D, Yun S, Lee H (2019). "Character Region Awareness for Text Detection". *CVPR 2019*, pp. 9357–9366. [arxiv.org/abs/1904.01941](https://arxiv.org/abs/1904.01941) ([doi:10.1109/CVPR.2019.00959](https://doi.org/10.1109/CVPR.2019.00959))

### shi2017
CRNN: unified, end-to-end trainable conv + recurrent network with CTC transcription; handles arbitrary-length sequences without character segmentation, lexicon-free or lexicon-based. arXiv preprint 2015 (1507.05717); journal version TPAMI 39(11):2298–2304, 2017.
- Shi B, Bai X, Yao C (2017). "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition". *IEEE Transactions on Pattern Analysis and Machine Intelligence* 39(11):2298–2304. [doi:10.1109/TPAMI.2016.2646371](https://doi.org/10.1109/TPAMI.2016.2646371) ([arxiv.org/abs/1507.05717](https://arxiv.org/abs/1507.05717))

### rapidai2021
RapidOCR: OCR toolkit running (converted) PaddleOCR models on ONNX Runtime, OpenVINO, MNN, PaddlePaddle, TensorRT and PyTorch backends. Apache-2.0 (converted model artifacts carry their upstream licences). Project's recommended citation: `@misc{RapidOCR 2021, author={RapidAI Team}, year={2021}}`. Benchmark used rapidocr >=3.9 (v3.9.0 released 2026-06-23; latest v3.9.2, 2026-07-21).
- RapidAI (2021). "RapidOCR: OCR Toolbox". *GitHub repository*. [github.com/RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR)

### paruchuri2025
Surya (Datalab): "OCR, layout analysis, reading order, table recognition in 90+ languages". Code Apache-2.0; model weights under a modified AI Pubs OpenRAIL-M licence (free for research, personal use and startups under $5M funding/revenue). The README's own BibTeX is `paruchuri2025surya`, author "Vikas Paruchuri and Datalab Team", 2025 -- so this entry uses a personal first author, not {{Datalab}}. Benchmark pinned surya-ocr 0.17.1 (released 2026-01-30).
- Paruchuri V, Datalab Team (2025). "Surya: A lightweight document OCR and analysis toolkit". *GitHub repository*. [github.com/datalab-to/surya](https://github.com/datalab-to/surya)


## OCR vision-language models

### cui2025b
PaddleOCR-VL (arXiv v1 2025-10-16; v4 current): 0.9B VLM = NaViT-style dynamic-resolution encoder + ERNIE-4.5-0.3B. **Confirmed two-stage**: "The first stage, PP-DocLayoutV2, is responsible for layout analysis, where it localizes semantic regions and predicts their reading order. Subsequently, the second stage, PaddleOCR-VL-0.9B, leverages these layout predictions to perform fine-grained recognition" (PP-DocLayoutV2 = RT-DETR detector + pointer network for reading order); VLM runs on cropped regions.
- Cui C, Sun T, Liang S, Gao T, Zhang Z, Liu J, Wang X, Zhou C, Liu H, Lin M, et al (2025b). "PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model". *arXiv preprint arXiv:2510.14528*. [arxiv.org/abs/2510.14528](https://arxiv.org/abs/2510.14528)

### cui2026
PaddleOCR-VL-1.5 technical report (v1 2026-01-29, v2 2026-04-03): 0.9B multi-task VLM; adds seal recognition and text spotting; introduces Real5-OmniDocBench (scanning, skew, warping, screen-photography, illumination).
- Cui C, Sun T, Liang S, Gao T, Zhang Z, Liu J, Wang X, Zhou C, Liu H, Lin M, et al (2026). "PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing". *arXiv preprint arXiv:2601.21957*. [arxiv.org/abs/2601.21957](https://arxiv.org/abs/2601.21957)

### zhang2026a
PaddleOCR-VL-1.6 technical report (2026-06-02): architecture identical to 1.5; region-aware data optimisation + progressive post-training with RL; 96.33% on OmniDocBench v1.6. **Still two-stage, but the layout model is now PP-DocLayoutV3**: "The full system consists of two models: PP-DocLayoutV3 for layout analysis and PaddleOCR-VL-1.6-0.9B for vision-language understanding."
- Zhang Z, Liu H, Liang S, Zhang Y, Xiang Y, Liu J, Sun T, Lin M, Zhang Y, Zhou C, et al (2026a). "PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training". *arXiv preprint arXiv:2606.03264*. [arxiv.org/abs/2606.03264](https://arxiv.org/abs/2606.03264)

### paddlepaddle2026b
Model card PaddlePaddle/PaddleOCR-VL-1.6 (released 2026-05-28; Apache-2.0). Official usage is the `paddleocr doc_parser --pipeline_version v1.6` / `PaddleOCRVL` pipeline (needs `paddleocr[doc-parser]>=3.6.0`, PaddlePaddle ≥3.2.1); the card's transformers snippet "only supports element-level recognition and text spotting" and the authors "recommend using the official method for inference, as it is faster and supports page-level document parsing".
- PaddlePaddle (2026b). "PaddleOCR-VL-1.6". *Hugging Face model card*. [huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)

### lu2026
OvisOCR2 technical report (2026-07-15): 0.8B end-to-end page parser emitting Markdown in reading order (text, LaTeX formulas, HTML tables, image regions as `<img src="images/bbox_...">`); SFT + RL + OPD; 96.58 on OmniDocBench v1.6, "the first end-to-end model to top this leaderboard previously dominated by pipeline methods".
- Lu S, Li Y, Xia Y, Chen Y, Ji AY, Jiang JP, Chen QG, Zhao J, Lin E, Li H, et al (2026). "OvisOCR2 Technical Report". *arXiv preprint arXiv:2607.13639*. [arxiv.org/abs/2607.13639](https://arxiv.org/abs/2607.13639)

### athmaas2026
Model card ATH-MaaS/OvisOCR2 (Apache-2.0; created 2026-07-13). `base_model: Qwen/Qwen3.5-0.8B`; config architecture is `Qwen3_5ForConditionalGeneration` — i.e. **it is a post-trained Qwen3.5-0.8B, not the Ovis architecture** (same team as Ovis). Card recommends vLLM 0.22.1.
- ATH-MaaS (2026). "OvisOCR2". *Hugging Face model card*. [huggingface.co/ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2)

### lu2024
Ovis (2024-05-31): MLLM with a learnable visual embedding table aligning visual and textual embeddings. Only cite as background (same first author/team); OvisOCR2 does not use this architecture per its card/config.
- Lu S, Li Y, Chen QG, Xu Z, Luo W, Zhang K, Ye HJ (2024). "Ovis: Structural Embedding Alignment for Multimodal Large Language Model". *arXiv preprint arXiv:2405.20797*. [arxiv.org/abs/2405.20797](https://arxiv.org/abs/2405.20797)

### taghadouini2026a
LightOnOCR paper (v1 2026-01-20, v2 2026-06-30): 1B end-to-end multilingual VLM converting document images to naturally ordered text "without brittle OCR pipelines"; distillation mix rich in scans, French and scientific PDFs; RL with IoU rewards for image bounding boxes; "9× smaller and substantially faster than prior best-performing models".
- Taghadouini S, Cavaillès A, Aubertin B (2026a). "LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR". *arXiv preprint arXiv:2601.14251*. [arxiv.org/abs/2601.14251](https://arxiv.org/abs/2601.14251)

### taghadouini2026b
HF blog (2026-01-19): reports LightOnOCR-2-1B at 83.2 ± 0.9 on olmOCR-Bench, with the **headers/footers category excluded** from the overall score.
- Taghadouini S, Cavaillès A, Aubertin B (2026b). "LightOnOCR-2-1B: a lightweight high-performance end-to-end OCR model family". *Hugging Face blog*. [huggingface.co/blog/lightonai/lightonocr-2](https://huggingface.co/blog/lightonai/lightonocr-2)

### lighton2026
Model card lightonai/LightOnOCR-2-1B (Apache-2.0; created 2026-01-16): RLVR-refined flagship variant; claims 5.71 pages/s on one H100 (~493k pages/day), "<$0.01 per 1,000 pages"; requires transformers ≥ v5.
- LightOn (2026). "LightOnOCR-2-1B". *Hugging Face model card*. [huggingface.co/lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)

### duan2026
GLM-OCR technical report (2026-03-11): 0.9B model (CogViT encoder + GLM-0.5B decoder) with Multi-Token Prediction; describes a two-stage pipeline where layout analysis precedes region-level recognition.
- Duan S, Xue Y, Wang W, Su Z, Liu H, Yang S, Gan G, Wang G, Wang Z, Yan S, et al (2026). "GLM-OCR Technical Report". *arXiv preprint arXiv:2603.10910*. [arxiv.org/abs/2603.10910](https://arxiv.org/abs/2603.10910)

### zai2026
Model card zai-org/GLM-OCR (MIT; created 2026-01-30): 94.62 on OmniDocBench v1.5. **Confirmed layout-first SDK**: "the SDK integrates PP-DocLayoutV3 and provides a complete … pipeline for document parsing, including layout analysis"; GitHub README: "The SDK provides the complete pipeline: layout detection, parallel region OCR, and result formatting." Model-only prompts are limited to "Text Recognition:", "Formula Recognition:", "Table Recognition:" or JSON-schema extraction.
- ZAI (2026). "GLM-OCR". *Hugging Face model card*. [huggingface.co/zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)

### ibm2025
Model card ibm-granite/granite-docling-258M (IBM Research; Apache-2.0; release date 2025-09-17): Idefics3-based, SigLIP2-base-patch16-512 encoder + Granite 165M LM; outputs DocTags. **No mention of handwriting; no mention of fp16.** It does say that on GPUs without bfloat16 (e.g. Colab T4) the model "outputs only exclamation marks" and recommends `dtype float32`; vLLM ≤0.10.2 needs `--revision untied`.
- IBM (2025). "granite-docling-258M". *Hugging Face model card*. [huggingface.co/ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)

### nassar2025
SmolDocling (2025-03-14; arXiv preprint, no venue found): 256M VLM introducing **DocTags**; block tags include `<text>`, `<caption>`, `<footnote>`, `<formula>`, `<title>`, `<page_footer>`, `<page_header>`, **`<picture>`** (pictures/figures; may wrap a `<caption>`), `<section_header>`, `<document_index>`, `<code>`, `<otsl>` (tables), `<list_item>`. Handwriting is not discussed.
- Nassar A, Marafioti A, Omenetti M, Lysak M, Livathinos N, Auer C, Morin L, Teixeira de Lima R, Kim Y, Gurbuz AS, et al (2025). "SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion". *arXiv preprint arXiv:2503.11576*. [arxiv.org/abs/2503.11576](https://arxiv.org/abs/2503.11576)

### rednote2025
Model card rednote-hilab/dots.ocr (MIT; released 2025-07-30; HF URL now redirects to dots-studio/dots.ocr): single VLM on a 1.7B LLM unifying layout detection and recognition. No paper or citation block for dots.ocr itself. Same `prompt_ocr` mode as dots.mocr ("Parse text only, except Page-header and Page-footer").
- rednote (2025). "dots.ocr". *Hugging Face model card*. [huggingface.co/rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)

### rednote2026
Model card dots-studio/dots.mocr (MIT; created 2026-03-19): 3B; adds image-to-SVG parsing. **`prompt_ocr` confirmed**: README comment "Parse text only, except Page-header and Page-footer"; in `prompts.py` the comment reads "prompt_ocr: parse ocr text except the Page-header and Page-footer" but the actual prompt string is only "Extract the text content from this image." — so header/footer exclusion is learned behaviour, not an instruction in the prompt. The layout-mode parser also writes a `*_nohf.md` without headers/footers "for compatibility with benchmarks like Omnidocbench and olmOCR-bench", and their olmOCR-bench results "delete the Page-header and Page-footer cells".
- rednote (2026). "dots.mocr". *Hugging Face model card*. [huggingface.co/dots-studio/dots.mocr](https://huggingface.co/dots-studio/dots.mocr)

### zheng2026
dots.mocr paper (2026-03-13): "Multimodal OCR", a compact 3B model parsing documents and structured graphics (charts, UI, figures→SVG). Note: the arXiv author list has "Yuqiu Ji" (19th) while the model-card BibTeX has "Jiyu Qiu" (9th); the arXiv list is used here.
- Zheng H, Li Y, Zhang K, Xin L, Zhao G, Liu H, Chen J, Lou J, Fu Q, Yang R, et al (2026). "Multimodal OCR: Parse Anything from Documents". *arXiv preprint arXiv:2603.13032*. [arxiv.org/abs/2603.13032](https://arxiv.org/abs/2603.13032)


## OCR vision-language models (cont.) and general VLMs

### wei2025
DeepSeek-OCR (arXiv v1 21 Oct 2025). Abstract: two components, "DeepEncoder" and "DeepSeek3B-MoE-A570M as the decoder" (a 3B-parameter mixture-of-experts LM with ~570M active parameters); 97% OCR precision at <10x compression, ~60% at 20x. HF card (MIT license) and GitHub README (news item 2025/10/23: "officially supported in upstream vLLM") recommend vLLM with the custom logits processor `NGramPerReqLogitsProcessor` (`logits_processors=[NGramPerReqLogitsProcessor]`, `ngram_size=30`, `window_size=90`, `whitelist_token_ids={128821, 128822}` = `<td>`, `</td>`) and `enable_prefix_caching=False`; the vLLM recipe page (https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html) says "It's important to use the custom logits processor along with the model for the optimal OCR and markdown generation performance."
- Wei H, Sun Y, Li Y (2025). "DeepSeek-OCR: Contexts Optical Compression". *arXiv preprint arXiv:2510.18234*. [arxiv.org/abs/2510.18234](https://arxiv.org/abs/2510.18234)

### wei2026
DeepSeek-OCR 2 (arXiv 28 Jan 2026; HF card deepseek-ai/DeepSeek-OCR-2, Apache-2.0, ~3B params BF16). Replaces the CLIP part of DeepEncoder with an LM-style encoder ("DeepEncoder V2", instantiated from Qwen2-0.5B) with "causal flow" queries that reorder visual tokens; paper says the decoder is kept the same: "a 3B-parameter MoE structure with about 500M active parameters". Dynamic resolution (0-6)x768x768 + 1x1024x1024 = (0-6)x144 + 256 visual tokens. The HF card's vLLM section only points to GitHub; the GitHub vLLM script (vLLM 0.8.5, custom model code) uses `NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=90, whitelist_token_ids={128821, 128822})`. (Note: the card's NGram recommendation for *upstream* vLLM is documented for DeepSeek-OCR v1, not v2.)
- Wei H, Sun Y, Li Y (2026). "DeepSeek-OCR 2: Visual Causal Flow". *arXiv preprint arXiv:2601.20552*. [arxiv.org/abs/2601.20552](https://arxiv.org/abs/2601.20552)

### datalab2026a
Chandra OCR 2 card (repo created 2026-03-16): "state of the art OCR model from Datalab that outputs markdown, HTML, and JSON"; claims 85.8 ± 0.8 on olmOCR-bench (ArXiv 86.9, Old Scans Math 89.1, Tables 92.1, Old Scans 51.1, Headers/Footers 91.4, Multi column 82.1, Long tiny text 93.7, Base 99.9) and 77.8% on its multilingual bench; vLLM recommended. Weights: modified OpenRAIL-M (free for research, personal use, startups under $2M funding/revenue; "Cannot be used competitively with our API"); code Apache-2.0. Card does not state the parameter count or base model in text (HF safetensors metadata: ~5.3B params; credits list "Qwen 3.5"). No citation block.
- Datalab (2026a). "Chandra OCR 2 (datalab-to/chandra-ocr-2)". *Hugging Face model card*. [huggingface.co/datalab-to/chandra-ocr-2](https://huggingface.co/datalab-to/chandra-ocr-2)

### datalab2026b
Surya OCR 2 card (repo created 2026-05-14): "Surya is a 650M param OCR model"; 83.3% on olmOCR-bench ("top under 3B params"), ~5 pages/s on an RTX 5090; layout, OCR and table recognition share one VLM ("Qwen3.5-style architecture, ~650M params") served by vLLM or llama.cpp, plus a separate EfficientViT-segformer line detector. Weights: modified AI Pubs Open Rail-M (free for research, personal use, startups under $5M); code Apache-2.0. Card's own suggested citation is the GitHub toolkit: `paruchuri2025surya` (Vikas Paruchuri and Datalab Team, 2025, https://github.com/datalab-to/surya).
- Datalab (2026b). "Surya OCR 2 (datalab-to/surya-ocr-2)". *Hugging Face model card*. [huggingface.co/datalab-to/surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2)

### allenai2025a
olmOCR-2-7B-1025-FP8 card (Apache-2.0; created 2025-10-06): FP8 quantization (llmcompressor) of olmOCR-2-7B-1025, fine-tuned from Qwen2.5-VL-7B-Instruct on olmOCR-mix-1025 then GRPO RL. olmOCR-bench with olmOCR pipeline v0.4.0: ArXiv 83.0, Old Scans Math 82.3, Tables 84.9, Old Scans 47.7, Headers and Footers 96.1, Multi column 83.7, Long tiny text 81.9, Base 99.7, Overall 82.4 ± 1.1 (BF16 version: 82.3 ± 1.1). Expects pages rendered at longest side 1288 px.
- AllenAI (2025a). "olmOCR-2-7B-1025-FP8". *Hugging Face model card*. [huggingface.co/allenai/olmOCR-2-7B-1025-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8)

### poznanski2025a
olmOCR paper (arXiv v1 25 Feb 2025, v3 2 Jul 2025; Allen Institute for AI). Introduces olmOCR-Bench: "1,402 distinct PDF documents derived from diverse source repositories, covered by 7,010 unique test cases". Test types: Text Presence (721), Text Absence (823), Natural Reading Order (1,061), Table Accuracy (1,020), Math Formula Accuracy (3,385), plus a per-PDF baseline test that "checks that some plain text output containing alphanumeric characters was actually produced for that page, that such output does not have a string of repeating N grams at the end (longer than 30), and that the output does not contain any characters from the Chinese, Japanese, or Emoji Unicode charsets." Tests per source: arXiv Math 2,927 (formula); Old Scans Math 458 (formula); Tables 1,020; Old Scans 526 (279 presence, 70 absence, 177 order; Library of Congress letters/typewritten documents); Headers Footers 753 (absence); Multi Column 884 (reading order); Long Tiny Text 442 (presence; Internet Archive dense small print). Per-source PDF counts are not given in the paper. Overall score = unweighted mean of per-source pass rates ("Overall score = 1/N ∑ Score(s)"); the bench code reports 95% bootstrap CIs (default 1,000 samples). Math: "We render a reference LaTeX equation using KaTeX in a headless browser and extract all rendered symbols and their (visual) bounding boxes", then "check if a matching collection of symbols, with the same relative orientations, exists anywhere in the final OCR document." The bench README (github.com/allenai/olmocr/tree/main/olmocr/bench) confirms the browser is Chromium via Playwright: "# Configure playwright headless browser to run the math rendering tests / playwright install chromium"; katex/render.py: "...from rendered LaTeX equations using Playwright and KaTeX." Note: arXiv metadata lists "Aman Rangapur" twice; the paper header lists 9 distinct authors (used here).
- Poznanski J, Rangapur A, Borchardt J, Dunkelberger J, Huff R, Lin D, Wilhelm C, Lo K, Soldaini L (2025a). "olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models". *arXiv preprint arXiv:2502.18443*. [arxiv.org/abs/2502.18443](https://arxiv.org/abs/2502.18443)

### poznanski2025b
olmOCR 2 (arXiv 22 Oct 2025): olmOCR-2-7B-1025 trained with RL with verifiable rewards where "our rewards are a diverse set of binary unit tests" (reward = fraction of passing tests); synthetic HTML-derived documents generate the tests. Reports olmOCR-Bench 82.4 ± 1.1, "+14.2 point overall improvement over our initial release". Describes test types: Text Presence, Text Absence ("e.g., headers, footers, or page numbers"), Natural Reading Order, Table Accuracy, "Math Formula Accuracy: Checks that a given math formula visually renders the same way with KaTeX", Baseline Robustness ("long repeated n-grams or non-target language characters do not appear").
- Poznanski J, Soldaini L, Lo K (2025b). "olmOCR 2: Unit Test Rewards for Document OCR". *arXiv preprint arXiv:2510.19817*. [arxiv.org/abs/2510.19817](https://arxiv.org/abs/2510.19817)

### mandal2025
Nanonets-OCR2-3B card (created 2025-10-13): `base_model: Qwen/Qwen2.5-VL-3B-Instruct`; image-to-markdown with semantic tagging (LaTeX, tables, signatures, watermarks, checkboxes, image descriptions). No license is stated anywhere in the card or its metadata (verified 2026-10-05). Card BibTeX lists five personal authors (used here).
- Mandal S, Talewar A, Thakuria S, Ahuja P, Juvatkar P (2025). "Nanonets-OCR2: A model for transforming documents into structured markdown with intelligent content recognition and semantic tagging". *Hugging Face model card*. [huggingface.co/nanonets/Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B)

### bai2025a
Qwen2.5-VL technical report (arXiv 19 Feb 2025, 27 authors): base of olmOCR-2 (7B) and Nanonets-OCR2 (3B); claims "robust structured data extraction from invoices, forms, and tables" and document parsing.
- Bai S, Chen K, Liu X, Wang J, Ge W, Song S, Dang K, Wang P, Wang S, Tang J, et al (2025a). "Qwen2.5-VL Technical Report". *arXiv preprint arXiv:2502.13923*. [arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)

### bai2025b
Qwen3-VL technical report (arXiv 26 Nov 2025, 64 authors): dense 2B/4B/8B/32B and MoE 30B-A3B/235B-A22B variants, native 256K interleaved context. Note the Qwen/Qwen3-VL-8B-Instruct HF card's own citation block cites the Qwen3 (2505.09388) and Qwen2.5-VL reports, not this one.
- Bai S, Cai Y, Chen R, Chen K, Chen X, Cheng Z, Deng L, Ding W, Gao C, Ge C, et al (2025b). "Qwen3-VL Technical Report". *arXiv preprint arXiv:2511.21631*. [arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)

### qwen2026
Qwen3.5 (cards Qwen/Qwen3.5-2B and Qwen/Qwen3.5-9B, Apache-2.0, Feb/Mar 2026): natively multimodal ("Causal Language Model with Vision Encoder"), early-fusion vision-language training, hybrid Gated DeltaNet + Gated Attention layout (2B: 24 layers, 6 x (3 x Gated DeltaNet + 1 x Gated Attention)). No technical report on arXiv found for the base Qwen3.5 family (only Qwen3.5-Omni, arXiv 2604.15804); the cards' citation block cites the blog post (author "Qwen Team", February 2026). The blog page is JavaScript-rendered and could not be read directly; metadata taken from the cards' citation block.
- Qwen (2026). "Qwen3.5: Towards Native Multimodal Agents". *Qwen blog*. [qwen.ai/blog?id=qwen3.5](https://qwen.ai/blog?id=qwen3.5)

### gemma2026
Gemma 4 Technical Report (arXiv v1 2 Jul 2026, revised 24 Jul 2026; 323 authors, first listed "Gemma Team"): open-weight multimodal models 2.3B–31B, dense and MoE, vision and audio encoders, reasoning mode. Cited by the google/gemma-4-E4B-it card.
- Gemma Team, El Abd S, Aggarwal V, Algayres R, Andreev A, Bachem O, Ballantyne I, Brick C, Cărbune V, Casbon M, et al (2026). "Gemma 4 Technical Report". *arXiv preprint arXiv:2607.02770*. [arxiv.org/abs/2607.02770](https://arxiv.org/abs/2607.02770)

### google2026a
google/gemma-4-E4B-it card (Apache-2.0; "Authors: Google DeepMind"): E4B = 4.5B effective parameters (8B with embeddings; Per-Layer Embeddings), 42 layers, 128K context, text/image/audio input, ~150M vision encoder, ~300M audio encoder.
- Google (2026a). "Gemma 4 E4B Instruction-Tuned (google/gemma-4-E4B-it)". *Hugging Face model card*. [huggingface.co/google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)


## Datasets and evaluations

### majstorovic2026
FineBooks blog post (Hugging Face community article under the `finebooks` org, published 10 Aug 2026; authors Sebastian Majstorovic [storytracer] and Daniel van Strien [davanstrien]). FineBooks is a Hugging Face + EleutherAI collaboration with two goals: test whether open OCR models are good enough to re-OCR historical books, then re-process public-domain collections, starting with the Biodiversity Heritage Library (>300,000 items, >64 million pages).
- Ground truth: IMPACT project + BHL-Europe expert transcriptions (2011–2012) of 6 BHL volumes, 2,165 pages, English/French/German/Latin, "roughly one error in two thousand characters", CC-BY; rebuilt as `finebooks/bhl-impact-gt`.
- Setup: 14 open-weight, permissively licensed models; each model ran all 2,165 pages as one Hugging Face Job (per-minute billing, "between a few cents and a few dollars per model"; pinned model revision, container image and script commit).
- Metric: CER (substitutions + deletions + insertions / reference chars), reported in the post as accuracy = 1 − CER. Two lanes: **diplomatic CER** counts modernising ſ→s as an error; **reading CER** counts it as correct. The post's table uses the reading lane. The leaderboard defines reading as "NFKC, case-folded, de-hyphenated, markup stripped, body only" and diplomatic as "NFC, case-sensitive, keeps long-s (ſ), ligatures and diacritics". Extra metrics: recall (word coverage), over-extraction (text not on the page, with blank/illustration pages scored as a separate category), loop rate (pages that repeat until the token limit; looped pages are excluded from the other scores, so the two have to be read together).
- Blog table, reading accuracy / cost per 1,000 pages on HF Jobs: dots.mocr (3B) 97.6% / $1.94; OvisOCR2 (0.9B) 96.9% / $0.46; PaddleOCR-VL-1.6 (1B) 96.1% / $0.34; olmOCR-2 (8.3B) 95.7% / $0.45; LightOnOCR-2 (1B) 95.1% / $0.37; Qwen3.5-9B (9.7B) 94.9% / $0.89; DeepSeek-OCR (3.3B) 93.8% / $0.37. Best: dots.mocr.
- Verdicts: good enough for LLM training corpora like Common Pile ("in most cases, yes"; re-processing all of BHL's public-domain holdings is "a realistic project" at these costs). For libraries replacing legacy OCR, "it depends": the models emit Markdown or plain text, sometimes with region coordinates but never word-level positions, so there is no drop-in ALTO XML. For scholarly transcription, "not quite yet": the models "silently modernise the long-s (ſ), ligatures, and other text features"; focused fine-tuning might fix this.
- Motivation cited: the Talkie project found that an LM trained on OCR text learned "at 30% of the efficiency" of the same model trained on human transcriptions. Common Pile has about 300k public-domain books from older OCR pipelines.
- Scope: antiqua typefaces only; no Fraktur, non-Latin scripts or handwriting; books only (no newspapers or multi-column layouts). Next step: re-OCR about 200,000 public-domain BHL items with a leading model and release them as the first FineBooks dataset.
- Leaderboard update (see finebooks2026b; board baked 2026-09-03): 16 models are ranked. rednote-hilab/dots.ocr now edges ahead (reading CER 0.0235) of dots.mocr (0.0237), and the top two "are statistically identical" (overlapping volume-bootstrap 95% CIs). Tesseract 5 = 0.0642 (rank 14). A kraken + PP-OCRv6-medium line pipeline is rank 4 (0.0338) and **best in the diplomatic lane** (0.0405).

- Majstorovic S, van Strien D (2026). "FineBooks: are open OCR models good enough to unlock historical knowledge?". *Hugging Face blog (community article, FineBooks org), 10 Aug 2026*. [huggingface.co/blog/finebooks/historical-books-ocr-leaderboard](https://huggingface.co/blog/finebooks/historical-books-ocr-leaderboard)

### finebooks2026a
Dataset card for `finebooks/bhl-impact-gt` ("FineBooks BHL IMPACT Ground Truth"): 2,165 pages from 6 natural-history books (1708–1913), license **CC-BY 3.0**, languages de/en/fr/la (plus occasional Cyrillic in one volume's references), about 390 MB of WebP images and metadata. Only one split (train). Columns: image, text, markdown, docling, PAGE XML. Transcription is about 99.95% accurate and "Nothing is normalized" (it keeps ſ, ligatures and curly quotes). No Fraktur. The card's own suggested citation is "IMPACT Centre of Competence & Biodiversity Heritage Library (2012). IMPACT-BHL ground truth", https://github.com/impactcentre/groundtruth-bhl.

- FineBooks (2026a). "FineBooks BHL IMPACT Ground Truth (finebooks/bhl-impact-gt)". *Hugging Face dataset card*. [huggingface.co/datasets/finebooks/bhl-impact-gt](https://huggingface.co/datasets/finebooks/bhl-impact-gt)

### finebooks2026b
BHL OCR Leaderboard Space plus the `finebooks/bhl-ocr-eval` repo (RESULTS.md). Reading-lane CER on all 2,165 pages (428 sparse/blank), 95% CIs from resampling volumes:
1 dots.ocr 0.0235 [0.0156, 0.0318]; 2 dots.mocr 0.0237; 3 OvisOCR2 0.0305; 4 kraken/PP-OCRv6-medium 0.0338; 5 PaddleOCR-VL-1.6 0.0392; 6 olmOCR-2-7B-1025-FP8 0.0432; 7 LightOnOCR-2-1B 0.0489; 8 GLM-OCR 0.0491; 9 Qwen3.5-9B 0.0507; 10 Qianfan-OCR 0.0575; 11 Unlimited-OCR 0.0604; 12 DeepSeek-OCR 0.0617; 13 DeepSeek-OCR-2 0.0620; 14 Tesseract 5 0.0642; 15 SmolDocling-256M-preview 0.0660; 16 Falcon-OCR 0.1425.
Other findings:
- Sparse/blank pages are where models diverge most. GLM-OCR has a content CER of 0.0236 but a sparse CER of 26.93.
- Loop % is a selection effect. PaddleOCR-VL-1.6 loops on 6.47% of pages and dots.ocr on 3.88%.
- Parameter counts are measured from the safetensors files: GLM-OCR is 1.33B and olmOCR-2 is 8.29B.
- Qwen3.5-9B decoding ablation, reading CER: greedy 0.0440, greedy + presence penalty 1.5 (the board row) 0.0507, card sampling 0.0533.
- Throughput is "deliberately" not scored. On an A10G, kraken's pipeline takes 12.3 s/page against 3–5 s for most specialists.
- surya-ocr-2 and PaddlePaddle's PP-OCRv6 are "held for v1.1".

- FineBooks (2026b). "BHL OCR Leaderboard". *Hugging Face Space*. [huggingface.co/spaces/finebooks/bhl-ocr-leaderboard](https://huggingface.co/spaces/finebooks/bhl-ocr-leaderboard)

### bhl2026
Biodiversity Heritage Library, an open-access digital library of biodiversity literature run by a consortium ("Inspiring discovery through free access to biodiversity knowledge"). According to the FineBooks post it has >300,000 items and >64M pages, and it publishes bulk downloads via AWS Open Data. The main site returned HTTP 403 to automated fetches, but about.biodiversitylibrary.org was verified.

- BHL (2026). "Biodiversity Heritage Library". *Website*. [biodiversitylibrary.org](https://www.biodiversitylibrary.org/)

### allenai2025b
Dataset card for `allenai/olmOCR-bench`: 1,403 PDFs and 7,010 unit tests. License **ODC-BY-1.0** ("intended for research and educational use in accordance with AI2's Responsible Use Guidelines").
- Splits and PDFs/tests: arxiv_math 522/2,927; old_scans_math 36/458; table_tests 188/1,020; old_scans 98/526; headers_footers 266/753; multi_column 231/884; long_tiny_text 62/442.
- Test classes: text present 721, text absent 823, reading order 1,061, table 1,020, math 3,385.
- Card created 2025-03-14.

- AllenAI (2025b). "olmOCR-bench". *Hugging Face dataset card*. [huggingface.co/datasets/allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)

### marti2002
The IAM handwriting database: forms of handwritten English text (based on the LOB corpus) for training and testing handwriting recognisers. Verified via Crossref: IJDAR 5(1):39–46, published Nov 2002.

- Marti UV, Bunke H (2002). "The IAM-database: an English sentence database for offline handwriting recognition". *International Journal on Document Analysis and Recognition 5(1):39–46*. [doi.org/10.1007/s100320200071](https://doi.org/10.1007/s100320200071)

### teklia2024
`Teklia/IAM-line` on HF, a line-level version of IAM. Splits: train **6,482**, validation **976**, test **2,915** lines (10,373 total). License (card metadata) **MIT**. All images are resized to a fixed height of 128 px. English. Created 2024-01-12. (The original IAM database has its own non-commercial research terms at fki.tic.heia-fr.ch, which is worth noting next to the MIT tag.)

- Teklia (2024). "IAM – line level (Teklia/IAM-line)". *Hugging Face dataset card*. [huggingface.co/datasets/Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line)

### levenshtein1966
This paper introduced deletion/insertion/reversal-correcting codes and the edit distance underlying CER. English translation in Soviet Physics Doklady 10(8):707–710 (1966), confirmed by multiple citing papers. The Russian original is Doklady Akademii Nauk SSSR 163(4):845–848 (1965), confirmed on Math-Net.Ru. There is no DOI; the URL is the Math-Net.Ru record of the original.

- Levenshtein VI (1966). "Binary codes capable of correcting deletions, insertions, and reversals". *Soviet Physics Doklady 10(8):707–710*. [mathnet.ru/eng/dan31411](https://www.mathnet.ru/eng/dan31411)

### efron1979
This paper introduced the bootstrap, i.e. estimating the sampling distribution of a statistic by resampling the observed data. It shows the jackknife is a linear approximation to the bootstrap. Annals of Statistics 7(1):1–26, verified on Project Euclid and Crossref.

- Efron B (1979). "Bootstrap Methods: Another Look at the Jackknife". *The Annals of Statistics 7(1):1–26*. [doi.org/10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)

### spearman1904
This is the origin of rank correlation (Spearman's ρ) and of the correction for attenuation due to measurement error. Crossref confirms Am J Psychol 15(1), first page 72; pages 72–101 per York U. Classics and other citations. JSTOR stable id 1412159.

- Spearman C (1904). "The Proof and Measurement of Association between Two Things". *The American Journal of Psychology 15(1):72–101*. [doi.org/10.2307/1412159](https://doi.org/10.2307/1412159)

### kesselman2008
M.S. thesis (Applied Intelligence), Mercyhurst College, May 2008; chair Kristan J. Wheaton. It analysed words of estimative probability in NIE key judgments from the 1950s to the 2000s. Of 50 words, only 13 were statistically significant, and "will" was used >700 times. It proposes the "Kesselman List of Estimative Words": 7 terms in roughly 15% bands, except for a 10% middle band ("chances a little better [or less]") and 14% top and bottom bands (Figure 5.2). The numeric ranges in Figure 5.2 are an image and were NOT text-verified.

- Kesselman RF (2008). "Verbal Probability Expressions in National Intelligence Estimates: A Comprehensive Analysis of Trends from the Fifties through Post 9/11". *M.S. thesis, Mercyhurst College*. [files.ethz.ch/isn/55739/kesselman_thesis_final.pdf](https://files.ethz.ch/isn/55739/kesselman_thesis_final.pdf)


## Systems

### kwon2023
vLLM paper. Introduces PagedAttention, which stores the KV cache in fixed-size blocks the way an OS pages memory. This cuts fragmentation and lets requests share KV-cache blocks. The abstract reports "2-4× throughput" over FasterTransformer and Orca at comparable latency. SOSP '23, pp. 611–626.
- Kwon W, Li Z, Zhuang S, Sheng Y, Zheng L, Yu CH, Gonzalez JE, Zhang H, Stoica I (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention". *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23)*, 611–626. [doi.org/10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165) ([arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180))

### vllm2026a
Automatic Prefix Caching (APC): "Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part" (feature page). The design page says each block hash covers the block's tokens, the parent block's hash, and extra keys such as the LoRA ID and multimodal hashes. On images it says: "The challenge for prefix caching to support this case is we need to differentiate images from the placeholders. To address this problem, we encode the image hash generated by the frontend image processor." So an identical image in the same prefix position can hit the cache, while a different image misses even though the placeholder tokens are the same. The page is an undated living doc; the year is the access year. Feature page: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- vLLM (2026a). "Automatic Prefix Caching". *vLLM documentation (design docs)*. [docs.vllm.ai/en/latest/design/prefix_caching/](https://docs.vllm.ai/en/latest/design/prefix_caching/)

### vllm2026b
tpu-inference is "an expressive and powerful new hardware plugin unifying JAX and PyTorch under a single lowering path within the vLLM project"; it powers vLLM TPU (PyPI `vllm-tpu`, 0.30.0 at access). The README support matrix was "Last Updated: 2026-08-27". The release matrix lists Qwen/Qwen2.5-VL-7B-Instruct (Multimodal) as Unit ✅, Correctness ✅, Performance ❓. Qwen/Qwen3.5-397B-A17B (Text) passes all three. Qwen/Qwen3.5-9B and Qwen/Qwen3-VL-8B-Instruct appear only as ❓ Untested ("The functionality exists but has not been recently or thoroughly verified"). Qwen3.5-2B is not listed. Also listed: Gemma 4 26B-A4B, 31B, E2B and E4B (all ✅ in release; in nightly E2B and E4B fail correctness), Gemma 3 27B, Llama 3.1 8B, Llama 3.3 70B, Qwen3 4B/30B-A3B/32B, Qwen3-Coder-480B, Qwen3-Embedding-8B, DeepSeek-R1, Kimi-K2.6 and gpt-oss-120b. deepseek-ai/DeepSeek-OCR, Llama-4-Maverick, Qwen3-Omni and DeepSeek-Math-V2 are ❓ Untested. In the nightly matrix Qwen2.5-VL-7B fails the performance test (❌).
- vLLM (2026b). "tpu-inference: TPU inference for vLLM, with unified JAX and PyTorch". *GitHub repository vllm-project/tpu-inference*. [github.com/vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)

### jax2026
The jit cache is keyed on argument shape and dtype. Sharp Bits: "Subsequent runs with parameters of same type and shape may not show the side-effect / This is because JAX now invokes a cached compilation of the function … JAX re-runs the Python function when the type or shape of the argument changes." The AOT page: "Compiled functions are specialized to a particular set of argument 'types,' such as arrays with a specific shape and element type." The jit page: "When we first invoke f, it will get compiled, and the resulting XLA code will get cached. Subsequent calls of f will reuse the cached code." The pages are undated; the year is the access year.
- JAX (2026). "JAX – The Sharp Bits". *JAX documentation*. [docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) (see also [docs.jax.dev/en/latest/aot.html](https://docs.jax.dev/en/latest/aot.html), [docs.jax.dev/en/latest/jit-compilation.html](https://docs.jax.dev/en/latest/jit-compilation.html))

### dao2023
FlashAttention-2 is about 2× faster than FlashAttention and reaches "50-73% of the theoretical maximum FLOPs/s on A100". GPU requirement from the Dao-AILab/flash-attention README: "Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). For Turing GPUs (T4, RTX 2080), see the separate flash-attention-turing repo, which supports a core subset of FlashAttention features on Turing." Also "bf16 requires Ampere, Ada, or Hopper GPUs". So the official package does not support Turing (T4, sm_75). The README's own citation gives the venue as ICLR 2024.
- Dao T (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning". *arXiv preprint arXiv:2307.08691 (ICLR 2024)*. [arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

### nvidia2025
CUDA 12.8 release notes, New Features: "This release adds compiler support for the following Nvidia Blackwell GPU architectures: SM_100, SM_101, SM_120". SM_120 is compute capability 12.0, the RTX PRO 6000 Blackwell / RTX 50 series. Supporting source: the Blackwell Compatibility Guide (v13.4) says "With versions 12.8 of the CUDA Toolkit, nvcc can generate cubin native to the Blackwell architecture (compute capability 10.0)", but it does not mention 12.0. These two sources support "Blackwell needs CUDA ≥ 12.8"; the release notes are the explicit source for sm_120.
- NVIDIA (2025). "CUDA Toolkit 12.8 Release Notes". *NVIDIA CUDA documentation*. [docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)

### nvidia2026
NVIDIA's compute-capability table lists compute capability 12.0 for "NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000 Blackwell Workstation Edition" and "RTX PRO 6000 Blackwell Max-Q Workstation Edition". The page is undated; the year is the access year.
- NVIDIA (2026). "CUDA GPU Compute Capability". *NVIDIA Developer*. [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)

### paddlepaddle2026c
The Linux pip install page lists paddlepaddle-gpu 3.3.0 wheel indexes per CUDA version. "If you are using CUDA 12.9 … python3 -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/", alongside cu126 and cu130. This page does not mention Blackwell; see paddlepaddle2026d for that.
- PaddlePaddle (2026c). "Install on Linux via PIP". *PaddlePaddle documentation*. [paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html)

### paddlepaddle2026d
The PaddleOCR-VL Blackwell tutorial covers "NVIDIA Blackwell-architecture GPUs include, but are not limited to: RTX 5090 … RTX 5050". It says "please ensure that your NVIDIA driver supports CUDA 12.9 or higher". The install step reads "# Note that PaddlePaddle for cu129 is being installed here: python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/". It was "verified for accuracy and speed on the RTX 5070"; other Blackwell GPUs are "not yet confirmed". The RTX PRO 6000 is not named.
- PaddlePaddle (2026d). "PaddleOCR-VL NVIDIA Blackwell-Architecture GPUs Usage Tutorial". *PaddleOCR documentation*. [paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html)

### google2026b
Official Colab pricing page (`<title>` "Colab Paid Services Pricing"). The plan cards are rendered client-side, so the figures were read from the page's embedded data, not from visible text. That data contains `ccuProductTitle="Pay As You Go"`, `formattedCCU100Price="$9.99"`, `formattedCCU500Price="$49.99"`, `formattedProSubscriptionPrice="$9.99"` and `formattedVeryProSubscriptionPrice="$49.99"` (US). This gives Pay As You Go at $9.99 per 100 compute units, Colab Pro at $9.99/month and Pro+ at $49.99/month. The Colab FAQ confirms that compute-unit balance governs paid access. The FAQ is at https://research.google.com/colaboratory/faq.html.
- Google (2026b). "Colab Paid Services Pricing". *Google Colaboratory*. [colab.research.google.com/signup](https://colab.research.google.com/signup)

### google2024
An official Google source for the monthly unit allotments. Colab Pro: "100 compute units that grant access to additional powerful GPUs, memory, features, and productivity enhancements enabled with AI assistance". Colab Pro+: "400 compute units for a total of 500 per month". Posted 2024-06-25, about Workspace editions; I found no newer official statement of the consumer Pro allotment.
- Google (2024). "Introducing Colab Pro and Colab Pro+ for Google Workspace". *Google Workspace Updates blog*. [workspaceupdates.googleblog.com/2024/06/google-workspace-colab-pro-and-colab-pro-plus.html](https://workspaceupdates.googleblog.com/2024/06/google-workspace-colab-pro-and-colab-pro-plus.html)

### mccormick2024
A third-party measured table of compute units per hour (updated March 2026). T4 1.19 ($0.12/h), L4 1.71 ($0.17/h), A100 40GB 5.40 ($0.54/h), A100 80GB 7.52 ($0.75/h), G4 8.71 ($0.87/h); H100 "?". Units are priced at "$10/100 units". Thunder Compute (Peterson C, 2025, https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025) reproduces the table, attributing it to mccormickml.com: "not officially published by Google". It identifies G4 as the RTX PRO 6000. Note that Thunder Compute's "Pro+ 600 CU" conflicts with Google's 500.
- McCormick C (2024). "Colab GPUs Features & Pricing". *mccormickml.com (updated March 2026)*. [mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)

### wolf2020
The Hugging Face Transformers library paper: "carefully engineered state-of-the art Transformer architectures under a unified API". EMNLP 2020 System Demonstrations, pp. 38–45.
- Wolf T, Debut L, Sanh V, Chaumond J, Delangue C, Moi A, Cistac P, Rault T, Louf R, Funtowicz M, et al (2020). "Transformers: State-of-the-Art Natural Language Processing". *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38–45. [doi.org/10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)
