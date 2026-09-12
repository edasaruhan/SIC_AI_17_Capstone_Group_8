"""Apply the documented interpretation erratum without changing legacy metrics.

Requires optional PyMuPDF, not a runtime analysis dependency. Run with:
uv run --no-project --with pymupdf python scripts/correct_sprint2_pdf.py
The unchanged original is recoverable from main commit ab7a3de.
"""

from pathlib import Path

import pymupdf

PATH = Path("reports/sprint-2/Sprint2_Modelleme_Raporu.pdf")
FONT = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
TEXT = (
    "Marka adları maskelendiğinde vekil modelin PR-AUC değeri 0,714'ten 0,362'ye düştü. "
    "Bu fark nedensel katkı yüzdesi, tanınırlık/içerik payına alt veya üst sınır ya da "
    "bütçe oranı olarak yorumlanamaz."
)


def main() -> None:
    import subprocess

    font = (
        FONT
        if FONT.exists()
        else Path(
            subprocess.check_output(["fc-match", "-f", "%{file}", "DejaVu Sans"], text=True).strip()
        )
    )
    if not font.is_file():
        raise RuntimeError("DejaVu Sans font required for Turkish glyphs")
    document = pymupdf.open(PATH)
    page_count = len(document)
    page = document[8]
    blocks = page.get_text("blocks")
    selected = [
        b
        for b in blocks
        if any(
            s in b[4]
            for s in (
                "Savunabildiğimiz ifade:",
                "Bu, tanınırlık payının üst sınırı",
                "bir bütçe oranı olarak değil.",
            )
        )
    ]
    if not selected and "Bu fark nedensel katkı" in page.get_text():
        print("Erratum already applied")
        return
    if len(selected) != 3:
        raise RuntimeError("Report layout changed; review the correction before applying")
    bounds = pymupdf.Rect(
        min(b[0] for b in selected),
        min(b[1] for b in selected),
        max(b[2] for b in selected),
        max(b[3] for b in selected),
    )
    page.add_redact_annot(bounds, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0)
    page.insert_font(fontname="ErratumFont", fontfile=str(font))
    spare = page.insert_textbox(
        bounds, TEXT, fontname="ErratumFont", fontsize=10, color=(0.1, 0.1, 0.1)
    )
    if spare < 0:
        raise RuntimeError("Correction did not fit; original PDF was not changed")
    temporary = PATH.with_suffix(".corrected.tmp")
    document.save(temporary, garbage=4, deflate=True)
    document.close()
    check = pymupdf.open(temporary)
    if "Bu, tanınırlık payının üst sınırı" in check[8].get_text() or len(check) != page_count:
        raise RuntimeError("Correction validation failed; original PDF was not changed")
    check.close()
    temporary.replace(PATH)
    print("Corrected interpretation on page 9; numerical result tables unchanged")


if __name__ == "__main__":
    main()
