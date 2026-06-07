"""First-import welcome / citation notice for CXAS.

Shown once per machine. A marker file is written to the CXAS home directory
(``$CXAS_PATH/.cxas`` or ``$HOME/.cxas``) so the message does not reappear on
subsequent imports. Set ``CXAS_NO_WELCOME=1`` to disable it entirely.
"""

import os

_MESSAGE = r"""
============================================================================
 Thank you for using CXAS (Chest X-Ray Anatomy Segmentation)!

 CXAS segments 159 anatomical structures in chest X-rays and extracts
 clinically relevant features (e.g. Cardio-Thoracic Ratio) using PyTorch.

 If you use this work or dataset, please cite both:

 @inproceedings{Seibold_2022_BMVC,
   author    = {Constantin Marc Seibold and Simon Reiss and M. Saquib Sarfraz
                and Matthias A. Fink and Victoria Mayer and Jan Sellner and
                Moon Sung Kim and Klaus H. Maier-Hein and Jens Kleesiek and
                Rainer Stiefelhagen},
   title     = {Detailed Annotations of Chest X-Rays via CT Projection for
                Report Understanding},
   booktitle = {33rd British Machine Vision Conference 2022, BMVC 2022,
                London, UK, November 21-24, 2022},
   publisher = {BMVA Press},
   year      = {2022},
   url       = {https://bmvc2022.mpi-inf.mpg.de/0058.pdf}
 }

 @article{seibold2023accurate,
   title   = {Accurate fine-grained segmentation of human anatomy in
              radiographs via volumetric pseudo-labeling},
   author  = {Seibold, Constantin and Jaus, Alexander and Fink, Matthias A
              and Kim, Moon and Reiss, Simon and Herrmann, Ken and
              Kleesiek, Jens and Stiefelhagen, Rainer},
   journal = {arXiv preprint arXiv:2306.03934},
   year    = {2023}
 }

 (This notice is shown only once. Set CXAS_NO_WELCOME=1 to disable it.)
============================================================================
"""


def _store_path() -> str:
    """Return the CXAS home directory, matching cxas.models path logic."""
    base = os.environ.get("CXAS_PATH", os.environ.get("HOME", os.path.expanduser("~")))
    return os.path.join(base, ".cxas")


def show_welcome_once() -> None:
    """Print the welcome/citation notice the first time CXAS is imported.

    A marker file in the CXAS home directory records that the message has been
    shown. Any error (e.g. read-only filesystem) is swallowed so importing CXAS
    never fails because of this notice.
    """
    if os.environ.get("CXAS_NO_WELCOME", "").strip() not in ("", "0", "false", "False"):
        return
    try:
        store_path = _store_path()
        marker = os.path.join(store_path, ".welcome_shown")
        if os.path.isfile(marker):
            return
        print(_MESSAGE)
        os.makedirs(store_path, exist_ok=True)
        with open(marker, "w") as f:
            f.write("shown\n")
    except Exception:
        # Never let the welcome notice break an import.
        pass
