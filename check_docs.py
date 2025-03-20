import logging
import os


logging.basicConfig(format="[%(asctime)s] (%(levelname)s): %(message)s", level=logging.DEBUG)

logging.debug("Check that all .rst files compile to .html.")

DOCS_PATH = os.path.join(os.path.dirname(__file__), "docs")
RSTS_PATH = os.path.join(DOCS_PATH, "generated")
HTML_PATH = os.path.join(DOCS_PATH, os.path.join("_build", "html", "generated"))

if not os.path.exists(RSTS_PATH):
    os.makedirs(RSTS_PATH)
if not os.path.exists(HTML_PATH):
    os.makedirs(HTML_PATH)

html_filenames = [os.path.splitext(name)[0] + ".html" for name in os.listdir(RSTS_PATH) if ".rst" in name]
html_filenames = sorted(html_filenames)
logging.debug(f".rst filenames: {html_filenames}")

for fname in html_filenames:
    fpath = os.path.join(HTML_PATH, fname)
    logging.debug(f"Check {fname}")
    assert os.path.exists(fpath), f"File {fpath} doesn`t exist."

logging.debug("All files exists.")
