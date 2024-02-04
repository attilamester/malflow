R2_ERROR_MESSAGES = [
    (
        """digraph any from
        here""",
        """digraph any from
        here"""
    ),
    (
        """Error: message
Warning:
helper test

digraph any
 any
  thing
  --
from here
        """,
        """digraph any
 any
  thing
  --
from here
        """
    )
]

DOT_LINES = [
    (
        ' any line here having [label="should escape these \' \" " URL=" " URL="the real URL value"];',
        ' any line here having [label="should escape these \' \' \' URL=\' "];'
    ),
    (
        '"0x00401400" [label="dbg.runtime.strhash" URL="dbg.runtime.strhash/0x00401400"];',
        '"0x00401400" [label="dbg.runtime.strhash"];'
    ),
    (
        '  "0x008bf630" [label="dbg.type..hash.struct { En string "json:\"en\""; Ru string "json:\"ru\"" }" URL="dbg.type..hash.struct { En string "json:\"en\""; Ru string "json:\"ru\"" }/0x008bf630"];',
        '  "0x008bf630" [label="dbg.type..hash.struct { En string \'json:\'en\'\'; Ru string \'json:\'ru\'\' }"];',
    )
]
