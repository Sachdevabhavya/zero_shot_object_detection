import h5py, numpy as np, os
from PIL import Image
from scipy.io import loadmat

MATFILE  = "data/nyu_depth_v2_labeled.mat"
OUT_IMGS  = "data/nyu_prompt_331/images"
OUT_DEPTH = "data/nyu_prompt_331/depth_maps"
OUT_MASKS = "data/nyu_prompt_331/masks"
OUT_SEG13 = "data/nyu_prompt_331/seg13"
OUT_SEG40 = "data/nyu_prompt_331/seg40"

print("Loading mappings...")
mapping40 = loadmat('data/classMapping40.mat')['mapClass'][0]
mapping40 = np.insert(mapping40, 0, 0)
mapping13 = loadmat('data/class13Mapping.mat')['classMapping13'][0][0][0][0]
mapping13 = np.insert(mapping13, 0, 0)
splits = loadmat('data/splits.mat')
idxs = splits['trainNdxs'].reshape(-1)[:331]

with h5py.File(MATFILE, "r") as f:
    for count, i in enumerate(idxs):
        raw_idx = i - 1
        img   = np.transpose(f["images"][raw_idx],  (2, 1, 0))
        depth = np.transpose(f["depths"][raw_idx])
        label = np.transpose(f["labels"][raw_idx])

        lbl_40 = mapping40[label]
        lbl_13 = mapping13[lbl_40].astype('uint8')
        lbl_40 = lbl_40.astype('uint8')
        mask   = (label > 0).astype(np.uint8) * 255

        Image.fromarray(img).save(f"{OUT_IMGS}/{i:05d}.jpg")
        np.save(f"{OUT_DEPTH}/{i:05d}.npy", depth)
        Image.fromarray(mask).save(f"{OUT_MASKS}/{i:05d}.png")
        Image.fromarray(lbl_13).save(f"{OUT_SEG13}/{i:05d}.png")
        Image.fromarray(lbl_40).save(f"{OUT_SEG40}/{i:05d}.png")

        if (count + 1) % 50 == 0:
            print(f"  Saved {count+1}/331")
print("[*] Extraction complete.")
