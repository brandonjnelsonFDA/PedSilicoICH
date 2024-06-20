# %%
from pathlib import Path
from io import StringIO

import pandas as pd
import nibabel as nib
import numpy as np

from notebooks.utils import CTobj, add_random_sphere_lesion

phantom_dir = Path('MIDA Head Phantom')
img = nib.load(phantom_dir/'MIDA_v1.nii')
phantom = np.array(img.get_fdata()).transpose(1, 0, 2)

with open(phantom_dir / 'MIDA_v1.txt', 'rb') as data:
    df = pd.read_csv(StringIO(data.read().decode(errors='replace')), sep='\t', names=['grayscale','c1', 'c2', 'c3', 'material'])
    material_lut = df.iloc[:-8]
    temp = df.iloc[-8:, :2].set_index('grayscale').T
    nx, ny, nz, dx, dy, dz = temp.nx.item(), temp.ny.item(), temp.nz.item(), temp.dx.item()*1000, temp.dy.item()*1000, temp.dz.item()*1000

material_lut = pd.read_csv('notebooks/material_lut.csv')
# %%
phantom[phantom == 50] = -1000 # air
for idx, row in material_lut[~material_lut['CT Number [HU]'].isna()].iterrows():
    phantom[phantom==row.grayscale] = row['CT Number [HU]']
# %%
nonair_projections = np.argwhere(phantom.mean(axis=1)[::-1].mean(axis=1) > -775).squeeze()
start_idx, end_idx = nonair_projections[0], nonair_projections[-1]
phantom = phantom[start_idx:end_idx]
# %%
radius = 10
contrast = 20
material = 'white_matter'

ground_truth_image = phantom

brain_mask = ground_truth_image==material_lut.loc[material_lut['xcist material']==material]['CT Number [HU]'].iloc[1]
              
img_w_lesion, lesion_image, lesion_coords = add_random_sphere_lesion(ground_truth_image, brain_mask, radius=radius, contrast=contrast)
# %%
ct = CTobj(img_w_lesion, spacings=(dz, dx, dy), patientname='MIDA Head',
                      studyname='full volume long scan', output_dir='MIDA Head')
# %%
ct.scout_view(startZ=-95, endZ=50)
# %%
ct.run_scan(startZ=-95, endZ=50, views=1000)
# %%
dicom_path = Path(ct.patientname)/ 'simulations' / f'{ct.patientid}' / 'dicoms'
ct.write_to_dicom(dicom_path / 'MIDA_head_full.dcm')
# %%