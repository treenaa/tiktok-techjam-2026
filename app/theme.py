"""Visual identity for the Spectral Evidence demo.

The palette is not an arbitrary colour scheme. `RAMP` is the colour ramp used to
render log-FFT magnitude maps in `app/spectral.py`, and the same ramp colours the
probability bar and the drift chart -- so the accent the interface is built from
is sampled from the data the detector actually reads.

Styling deliberately goes through our own wrapper classes (`sx-*`) rather than
Streamlit's internal class names, which change between releases.
"""

from __future__ import annotations

from typing import List, Tuple

# -- palette ---------------------------------------------------------------
VOID = "#0A0F1A"       # ground: the field a spectrum is plotted against
PANEL = "#141C2B"      # card surface
PANEL_HI = "#1B2536"   # raised surface
RULE = "#253247"       # hairlines
INK = "#E8EDF5"
MUTED = "#8A9AB2"
FAINT = "#5C6B82"

STABLE = "#3FB98A"     # class survived every transform
FLIPPED = "#F0603A"    # class changed under at least one transform

#: Spectral energy ramp, low frequency/energy -> high.
RAMP: List[Tuple[int, int, int]] = [
    (0x1B, 0x1E, 0x6B),
    (0x6B, 0x2E, 0x8F),
    (0xC0, 0x3A, 0x5B),
    (0xF0, 0x70, 0x20),
    (0xFF, 0xD9, 0x8A),
]

ACCENT = "#C03A5B"
ACCENT_WARM = "#F07020"

FONT_IMPORT = (
    "https://fonts.googleapis.com/css2"
    "?family=Instrument+Serif:ital@0;1"
    "&family=IBM+Plex+Mono:wght@400;500;600"
    "&family=IBM+Plex+Sans:wght@400;500;600"
    "&display=swap"
)


def ramp_css(direction: str = "90deg") -> str:
    """The spectral ramp as a CSS gradient."""
    stops = ", ".join(
        "rgb(%d,%d,%d) %d%%" % (r, g, b, round(100 * i / (len(RAMP) - 1)))
        for i, (r, g, b) in enumerate(RAMP)
    )
    return "linear-gradient(%s, %s)" % (direction, stops)


def css() -> str:
    """The single injected style block."""
    return """
<style>
@import url('%(font)s');

:root{
  --void:%(void)s; --panel:%(panel)s; --panel-hi:%(panel_hi)s; --rule:%(rule)s;
  --ink:%(ink)s; --muted:%(muted)s; --faint:%(faint)s;
  --stable:%(stable)s; --flipped:%(flipped)s;
  --accent:%(accent)s; --accent-warm:%(accent_warm)s;
  --ramp:%(ramp)s;
}

/* Set the family on the root only and let it inherit. Overriding bare `span`
   and `div` also hits Streamlit's Material icon glyphs, which are selected by
   font-family -- the icons then render as their literal ligature text
   ("uploadUpload" on the file uploader). */
.stApp{
  background:var(--void);
  font-family:'IBM Plex Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  color:var(--ink);
}
.stApp p, .stApp li, .stApp label{ color:var(--ink); }
/* Streamlit's toolbar is position:absolute and 60px tall, so it paints over the
   top of the page rather than pushing it down. Streamlit's own default top
   padding exists to clear it; anything smaller clips the first element. Keep
   this comfortably above 60px. */
.block-container{ padding-top:5rem; padding-bottom:5rem; max-width:1180px; }
#MainMenu, footer{ visibility:hidden; }
/* Let the ground show through the toolbar strip, and stop it swallowing clicks
   on the masthead beneath it. */
[data-testid="stHeader"]{ background:transparent; pointer-events:none; }
[data-testid="stHeader"] button,
[data-testid="stHeader"] a{ pointer-events:auto; }

/* -- masthead -------------------------------------------------------- */
.sx-mast{ display:flex; flex-direction:column; gap:.55rem;
  padding-bottom:1.1rem; border-bottom:1px solid var(--rule); margin-bottom:1.6rem; }
.sx-word{ font-family:'Instrument Serif',Georgia,serif; font-size:clamp(2.4rem,5.4vw,3.9rem);
  line-height:.96; letter-spacing:-.015em; color:var(--ink); margin:0; }
.sx-word em{ font-style:italic;
  background:var(--ramp); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; }
.sx-thesis{ color:var(--muted); font-size:1.02rem; max-width:62ch; margin:0; line-height:1.5; }

.sx-eyebrow{ font-family:'IBM Plex Mono',monospace; font-size:.68rem; font-weight:600;
  letter-spacing:.16em; text-transform:uppercase; color:var(--faint); margin:0; }

/* -- provenance strip ------------------------------------------------ */
.sx-prov{ display:flex; flex-wrap:wrap; gap:.45rem .5rem; margin-top:.35rem; }
.sx-tag{ font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--muted);
  background:var(--panel); border:1px solid var(--rule); border-radius:2px;
  padding:.26rem .55rem; white-space:nowrap; }
.sx-tag b{ color:var(--ink); font-weight:500; }

/* -- section heads --------------------------------------------------- */
.sx-h{ font-family:'Instrument Serif',Georgia,serif; font-size:1.75rem; line-height:1.15;
  margin:0 0 .15rem 0; color:var(--ink); letter-spacing:-.01em; }
.sx-sub{ color:var(--muted); font-size:.92rem; margin:0 0 1rem 0; max-width:70ch; line-height:1.5; }
.sx-rule{ height:1px; background:var(--rule); margin:2.4rem 0 1.5rem 0; border:0; }

/* -- verdict --------------------------------------------------------- */
.sx-verdict{ background:var(--panel); border:1px solid var(--rule); border-radius:4px;
  padding:1.4rem 1.6rem; display:flex; flex-direction:column; gap:.9rem; }
.sx-prob{ font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;
  font-size:clamp(2.8rem,7vw,4.4rem); font-weight:500; line-height:1; letter-spacing:-.03em;
  color:var(--ink); }
.sx-label{ font-family:'Instrument Serif',Georgia,serif; font-size:1.65rem; line-height:1.1; }
.sx-bar{ height:9px; border-radius:5px; background:#0E1626; border:1px solid var(--rule);
  overflow:hidden; position:relative; }
.sx-bar i{ display:block; height:100%%; background:var(--ramp); }
.sx-thresh{ position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--ink); opacity:.85; }
.sx-note{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:var(--faint); }

.sx-stat{ display:flex; flex-direction:column; gap:.18rem; }
.sx-stat .k{ font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.13em;
  text-transform:uppercase; color:var(--faint); }
.sx-stat .v{ font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;
  font-size:1.5rem; font-weight:500; color:var(--ink); line-height:1.1; }
.sx-stat .v.ok{ color:var(--stable); }
.sx-stat .v.bad{ color:var(--flipped); }

/* -- ladder cards ---------------------------------------------------- */
.sx-card{ background:var(--panel); border:1px solid var(--rule); border-radius:4px;
  padding:.7rem .8rem .8rem; height:100%%; }
.sx-card.flip{ border-color:var(--flipped); }
.sx-card.clean{ border-color:var(--accent-warm); }
.sx-cap{ display:flex; align-items:baseline; justify-content:space-between; gap:.4rem;
  margin-bottom:.45rem; }
.sx-name{ font-family:'IBM Plex Mono',monospace; font-size:.76rem; font-weight:600; color:var(--ink); }
.sx-delta{ font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;
  font-size:.72rem; color:var(--muted); }
.sx-delta.bad{ color:var(--flipped); }
.sx-delta.ok{ color:var(--stable); }
.sx-p{ font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;
  font-size:1.28rem; font-weight:500; color:var(--ink); line-height:1.1; margin-top:.4rem; }
/* Fixed block so cards in a row keep the same height when a parameter string
   wraps to a second or third line. */
.sx-meta{ font-family:'IBM Plex Mono',monospace; font-size:.66rem; color:var(--faint);
  margin-top:.1rem; min-height:2.5em; line-height:1.25; }
.sx-flag{ font-family:'IBM Plex Mono',monospace; font-size:.6rem; font-weight:600;
  letter-spacing:.1em; text-transform:uppercase; color:var(--flipped); }

.sx-img{ display:block; width:100%%; height:auto; border-radius:2px;
  border:1px solid var(--rule); }

/* -- pair labels under images ---------------------------------------- */
.sx-pair{ display:flex; gap:.3rem; font-family:'IBM Plex Mono',monospace;
  font-size:.6rem; letter-spacing:.1em; text-transform:uppercase; color:var(--faint);
  margin-top:.25rem; }
.sx-pair span{ flex:1; }

/* -- evidence table -------------------------------------------------- */
.sx-tbl{ width:100%%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace;
  font-size:.8rem; font-variant-numeric:tabular-nums; }
.sx-tbl th{ text-align:left; font-size:.63rem; letter-spacing:.12em; text-transform:uppercase;
  color:var(--faint); font-weight:600; padding:.5rem .6rem; border-bottom:1px solid var(--rule); }
.sx-tbl td{ padding:.5rem .6rem; border-bottom:1px solid var(--rule); color:var(--ink); }
.sx-tbl tr:last-child td{ border-bottom:none; }
.sx-tbl .up{ color:var(--stable); }
.sx-tbl .dn{ color:var(--flipped); }

.sx-foot{ color:var(--faint); font-size:.82rem; line-height:1.6; max-width:70ch; }
.sx-foot b{ color:var(--muted); font-weight:500; }

/* -- streamlit chrome, via stable hooks ------------------------------ */
[data-testid="stFileUploaderDropzone"]{
  background:var(--panel); border:1px dashed var(--rule); border-radius:4px; }
[data-testid="stSidebar"]{ background:#0C121E; border-right:1px solid var(--rule); }
div[data-baseweb="tab-list"]{ gap:.3rem; border-bottom:1px solid var(--rule); }
.stAlert{ border-radius:4px; }
</style>
""" % {
        "font": FONT_IMPORT,
        "void": VOID,
        "panel": PANEL,
        "panel_hi": PANEL_HI,
        "rule": RULE,
        "ink": INK,
        "muted": MUTED,
        "faint": FAINT,
        "stable": STABLE,
        "flipped": FLIPPED,
        "accent": ACCENT,
        "accent_warm": ACCENT_WARM,
        "ramp": ramp_css(),
    }
