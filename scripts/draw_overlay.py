import json, os, sys, cv2
base = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 30
d = json.load(open(os.path.join(base, "labels.json")))
os.makedirs(os.path.join(base, "overlay"), exist_ok=True)
col = {"pass": (0, 200, 0), "missing_cap": (0, 140, 255), "crooked_cap": (200, 0, 200),
       "damaged_vial": (0, 0, 255), "no_vial": (255, 120, 0)}
for rec in d["images"]:
    img = cv2.imread(os.path.join(base, "images", rec["image"] + ".png"))
    for n, (u, v) in d["positions"].items():
        c = col[rec["labels"][n]]
        cv2.circle(img, (int(u), int(v)), R, c, 2)
        cv2.putText(img, n[1:], (int(u) - 12, int(v) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(base, "overlay", rec["image"] + ".png"), img)
    print("wrote", rec["image"], img.shape)
