"""Verbatim OCR prompts from each model's card / reference code.

Prompts are part of a model's training contract; paraphrasing them usually
costs accuracy, so they are copied exactly (including odd spacing).
"""

OVIS_OCR2 = (
    "\nExtract all readable content from the image in natural human reading order and output the result as a "
    "single Markdown document. For charts or images, represent them using an HTML image tag: "
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."
)

NANONETS = (
    "Extract the text from the above document as if you were reading it naturally. Return the tables in html "
    "format. Return the equations in LaTeX representation. If there is an image in the document and image caption "
    "is not present, add a small description of the image inside the </img> tag; otherwise, add the image caption "
    "inside </img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers "
    "should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer "
    "using ☐ and ☑ for check boxes."
)

# olmocr.prompts.build_no_anchoring_v4_yaml_prompt (olmOCR-2)
OLMOCR_V4 = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)

_CHANDRA_TAGS = ["math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p", "th", "div", "pre",
                 "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li", "input", "a", "span", "img", "hr", "tbody", "small",
                 "caption", "strong", "thead", "big", "code", "chem"]
_CHANDRA_ATTRS = ["class", "colspan", "rowspan", "display", "checked", "type", "border", "value", "style", "href",
                  "alt", "align", "data-bbox", "data-label"]
_CHANDRA_ENDING = f"""
Only use these tags {_CHANDRA_TAGS}, and these attributes {_CHANDRA_ATTRS}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

# chandra.prompts.OCR_PROMPT
CHANDRA_OCR = f"OCR this image to HTML.\n\n{_CHANDRA_ENDING}"

# surya.inference.prompts.HIGH_ACCURACY_BBOX_PROMPT (surya 0.22 full-page mode)
SURYA2_FULL_PAGE = ("OCR this image to HTML. Each block is a div with data-label and data-bbox "
                    "(x0 y0 x1 y1, normalized 0-1000).")

# For general-purpose VLMs (no official OCR prompt). Starts from the Qwen VL cookbook's
# "Read all the text in the image." and asks for Markdown tables so table tests can pass.
GENERAL_OCR = ("Read all the text in the image. Transcribe it exactly as written, in natural reading order, "
               "as Markdown. Format tables as Markdown tables. Do not add any commentary.")

PROMPTS = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, str) and not k.startswith("_")}
