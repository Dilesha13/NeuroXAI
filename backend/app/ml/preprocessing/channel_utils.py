def canonicalize_ch_name(ch: str) -> str:
    c = ch.strip()
    if c.lower().startswith('eeg '):
        c = c[4:].strip()
    for suf in ['-REF', '-Ref', '-ref']:
        if c.endswith(suf):
            c = c[:-len(suf)].strip()
    return c
