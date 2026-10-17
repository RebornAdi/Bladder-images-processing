import cv2, re, pytesseract

def ocr_printed_values(image, table_region=(0.0, 0.70, 0.25, 1.0)):
    """OCR the burned-in measurement table (bottom-left by default).
    Returns {d1,d2,d3,vol} in mm/ml (None if unread) + 'flag' note.

    Strategy: first try to match by label (d1/d2/d3/vol). For any value
    still missing, fall back to POSITIONAL order — the table always
    lists d1, d2, d3, Vol top-to-bottom, so the first four plausible
    mm-range numbers in order fill the gaps. Every value is flagged if
    label-matching failed so you know which to eyeball."""
    h, w = image.shape[:2]
    fx0, fy0, fx1, fy1 = table_region
    crop = image[int(h*fy0):int(h*fy1), int(w*fx0):int(w*fx1)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(up, 120, 255, cv2.THRESH_BINARY)
    txt = pytesseract.image_to_string(th, config='--psm 6')

    out = {'d1': None, 'd2': None, 'd3': None, 'vol': None, 'raw': txt, 'flag': ''}
    label_hits = set()

    # pass 1: label-based
    for line in txt.splitlines():
        low = line.lower()
        nums = re.findall(r'(\d{1,3}\.\d{1,2})', line)
        if not nums:
            continue
        v = float(nums[0])
        if 'vol' in low and 5 < v < 5000:
            out['vol'] = v; label_hits.add('vol')
        elif 'd1' in low and 5 < v < 200:
            out['d1'] = v; label_hits.add('d1')
        elif 'd2' in low and 5 < v < 200:
            out['d2'] = v; label_hits.add('d2')
        elif 'd3' in low and 5 < v < 200:
            out['d3'] = v; label_hits.add('d3')

    # pass 2: positional fallback for whatever's still missing
    if any(out[k] is None for k in ('d1','d2','d3','vol')):
        seq = []
        for line in txt.splitlines():
            for m in re.findall(r'(\d{1,3}\.\d{1,2})', line):
                seq.append(float(m))
        # d1,d2,d3 are mm (5-200); vol is ml (can be larger)
        mm_vals = [v for v in seq if 5 < v < 200]
        for key, val in zip(('d1','d2','d3'), mm_vals[:3]):
            if out[key] is None and val is not None:
                out[key] = val
        if out['vol'] is None:
            vols = [v for v in seq if 5 < v < 5000 and v not in mm_vals[:3]]
            if vols:
                out['vol'] = vols[0]

    missing_labels = [k for k in ('d1','d2','d3','vol') if k not in label_hits]
    if missing_labels:
        out['flag'] = 'CHECK:' + ','.join(missing_labels)
    return out

if __name__ == "__main__":
    import sys
    im = cv2.imread(sys.argv[1])
    r = ocr_printed_values(im)
    print(f"d1={r['d1']} d2={r['d2']} d3={r['d3']} vol={r['vol']}  {r['flag']}")