import os

img_dir = "data/nyu_prompt_331/images"
lbl_dir = "data/nyu_prompt_331/labels"

os.makedirs(lbl_dir, exist_ok=True)

# Generate simple dummy targets for ALL images
count = 0
for filename in os.listdir(img_dir):
    if filename.endswith(".jpg"):
        idx = filename.replace('.jpg', '')
        with open(os.path.join(lbl_dir, f"{idx}.txt"), 'w') as f:
            f.write("0 0.5 0.5 0.5 0.5\n")
        count += 1

print(f"Generated {count} label files.")
