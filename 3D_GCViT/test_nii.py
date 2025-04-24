import nibabel as nib

# Datei laden
img = nib.load('/home/wiebketeetz/ct_clip_data/train_1_a_1.nii')

# Header anzeigen
print("=== Header ===")
print(img.header)

# Affine Matrix anzeigen (zeigt die räumliche Zuordnung)
print("\n=== Affine Matrix ===")
print(img.affine)

# Die tatsächlichen Bilddaten als NumPy-Array
data = img.get_fdata()
print("\n=== Datenform ===")
print(data.shape)

# Beispiel: Wertebereich prüfen
print(f"\nMin: {data.min()}, Max: {data.max()}")