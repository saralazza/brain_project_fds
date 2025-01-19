import os

def load(path):
    # Load images from original dataset diveding positive and negative images
    image_paths_yes = []
    image_paths_no = []
    for dirname, _, filenames in os.walk(path):
        if not (dirname.endswith("no") or dirname.endswith("yes")):
            continue
        for filename in filenames:
            image_path = os.path.join(dirname, filename)
            if dirname.lower().endswith("no"):
                image_paths_no.append(image_path)
            else:
                image_paths_yes.append(image_path)
    return image_paths_no, image_paths_yes

