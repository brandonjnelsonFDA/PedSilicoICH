import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pydicom
from datetime import datetime

from dxcist.xcist_sims import get_projection_data, get_reconstructed_data, install_path, convert_to_dicom, voxelize_ground_truth
from gecatsim.reconstruction.pyfiles import recon

import gecatsim as xc

# https://radiopaedia.org/articles/windowing-ct?lang=us
display_settings = {
    'brain': (80, 40),
    'subdural': (300, 100),
    'stroke': (40, 40),
    'temporal bones': (2800, 600),
    'soft tissues': (400, 50),
    'lung': (1500, -600),
    'liver': (150, 30),
}

def ctshow(img, window='soft tissues', fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax=plt.subplots()
    # Define some specific window settings here
    if isinstance(window, str):
        if window not in display_settings:
            raise ValueError(f"{window} not in {display_settings}")
        ww, wl = display_settings[window]
    elif isinstance(window, tuple):
        ww = window[0]
        wl = window[1]
    else:
        ww = 6.0 * img.std()
        wl = img.mean()

    if img.ndim == 3: img = img[0].copy()

    ax.imshow(img, cmap='gray', vmin=wl-ww/2, vmax=wl+ww/2)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.imshow(img, cmap='gray', vmin=wl-ww/2, vmax=wl+ww/2)

def add_sphere(phantom:np.ndarray, center:tuple|None=None, radius:tuple|None=None):
    '''
    Returns binary sphere mask based on input phantom array and center coordinates and radii parameters
    
    sphere defined as r^2 = z^2 + x^2 + y^2

    :param phantom: 3D array to add sphere to 
    '''
    center = center or [dim//2 for dim in phantom.shape]
    radius = radius or [dim//10 for dim in phantom.shape]
    z, x, y = np.meshgrid(range(phantom.shape[0]), range(phantom.shape[1]), range(phantom.shape[2]))
    distance_matrix = (z - center[0])**2 + (x-center[1])**2 + (y-center[2])**2
    return np.where(distance_matrix > radius**2, False, True)


def add_random_sphere_lesion(vol, mask, radius=20, contrast=-100):
    r = radius
    volume = (4/3*np.pi*r**3)*0.95
    lesion_vol = np.zeros_like(vol)
    counts = 0
    while np.sum(mask & (lesion_vol==contrast)) < volume: #can increase threshold to size of lesion
        lesion_vol = np.zeros_like(vol)
        z, x, y = np.argwhere(mask)[np.random.randint(0, mask.sum())]
        if mask[z].sum() < np.pi*r**2:
            continue
        counts += 1
        sphere = add_sphere(vol, center=(z, x, y), radius=radius).transpose(1, 0, 2)
        lesion_vol[sphere]=contrast
        if counts > 20:
            raise ValueError("Failed to insert lesion into mask")

    img_w_lesion = vol + lesion_vol
    return img_w_lesion, lesion_vol, (z, x, y)

    
def initialize_xcist(ground_truth_image, spacings=(1,1,1), output_dir='default', phantom_id='default', kVp=120):
    '''
    :param fov: in mm
    :param spacings: z, x, y in mm
    '''
    print('Initializing Scanner object...')
    print(''.join(10*['-']))
    
    # load defaults
    ct = xc.CatSim(install_path/'defaults/Phantom_Default',
                   install_path/'defaults/Physics_Default',
                   install_path/'defaults/Protocol_Default',
                   install_path/'defaults/Recon_Default',
                   install_path/'defaults/Scanner_Default')
    
    ct.cfg.waitForKeypress=False
    ct.cfg.do_Recon = True
    

    # prepare directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    phantom_path = output_dir / 'phantoms' / f'{phantom_id}'
    phantom_path.mkdir(exist_ok=True, parents=True)
    ct.cfg.phantom.filename = str(phantom_path / f'{phantom_id}.json')

    # prepare material density arrays from ground truth phantom
    if ground_truth_image.ndim == 2:
        ground_truth_image = ground_truth_image[None]
    
    dicom_path = phantom_path / 'dicom'
    for slice_id, img in enumerate(ground_truth_image):
        dicom_filename = dicom_path / f'1-{slice_id:03d}.dcm'    
        convert_to_dicom(img, dicom_filename, spacings=spacings)

    voxelize_ground_truth(dicom_path, phantom_path)
    print('Scanner Ready')
    return ct

class CTobj():
    """
        A class to hold CT simulation data and run simulations

        :param phantom: phantom object to be scanned
        :param framework: Optional, CT simulation framework options include `['CATSIM'] <https://github.com/JeffFessler/mirt>`_
        :returns: None
        
        See also <https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L19>
    """
    def __init__(self, phantom, spacings, kVp=120, patientname="default", patientid=0, age=0, studyname="default", studyid=0, seriesname="default", seriesid=0, framework='CATSIM', output_dir=None) -> None:
        """Constructor method
        """
        output_dir =  output_dir or f'{patientname}'
        self.output_dir = Path(output_dir)
        self.phantom=phantom
        self.spacings=spacings
        self.kVp=kVp
        self.age=age
        self.patientname=patientname
        self.patientid=patientid
        self.studyname=studyname or self.patientname
        self.studyid=studyid
        self.seriesname=seriesname
        self.seriesid=seriesid
        self.framework=framework
        self.ndetectors=900
        self.nangles=1000
        self.detector_size=1
        self.recon=None
        self.projections=None
        self.groundtruth=None
        self.patient_diameter = 18
        
        self.xcist = initialize_xcist(self.phantom, self.spacings, output_dir=self.output_dir,
                                      phantom_id=patientid, kVp=self.kVp)
        self.start_positions = self.calculate_start_positions()

    def calculate_start_positions(self, slice_thickness=5):
        # determine number of axial scans required to cover the phantom
        detector_width = self.xcist.scanner.detectorRowCount * self.xcist.scanner.detectorRowSize
        magnification = self.xcist.scanner.sdd / self.xcist.scanner.sid
        detector_width_at_isocenter = detector_width / magnification
        
        safe_width_at_isocenter = detector_width_at_isocenter - 2*self.xcist.scanner.detectorRowSize
        valid_slices = int(safe_width_at_isocenter // slice_thickness) # remove one at each end to avoid cone beam artifacts
        
        self.scan_width = valid_slices*slice_thickness 
        self.total_scan_length = self.spacings[0]*self.phantom.shape[0]
        return np.arange(-self.total_scan_length/2, self.total_scan_length/2, self.scan_width)

    def scout_view(self, startZ=None, endZ=None, slice_thickness=5):
        '''
        Preview radiograph useful for determining scan range startZ and endZ
        :param startZ: optional starting table position in mm of the scan, see self.start_positions
        :param endZ: optional last position of scan in mm, see self.start_positions
        '''
        self.start_positions = self.calculate_start_positions(slice_thickness)
        start_positions = self.calculate_start_positions(slice_thickness)
        if startZ is not None:
            if startZ < start_positions.min():
                raise ValueError(f'startZ is outside the range of valid start positions: {self.start_positions}')
            start_positions = start_positions[start_positions>startZ]
        if endZ is not None:
            if endZ > self.start_positions.max():
                raise ValueError(f'startZ is outside the range of valid start positions: {self.start_positions}')
            start_positions = start_positions[start_positions<endZ]

        plt.imshow(self.phantom.sum(axis=1)[::-1], cmap='gray', origin='upper', extent=[-self.phantom.shape[0]*self.spacings[0]/2, self.phantom.shape[0]*self.spacings[0]/2,
                                                                                        self.start_positions[0], self.start_positions[0]+self.total_scan_length])
        plt.hlines(y=start_positions[0], xmin=-self.phantom.shape[0]*self.spacings[0]/2, xmax=self.phantom.shape[0]*self.spacings[0]/2, color='red')
        plt.annotate('Start', (0, start_positions[0]-10), horizontalalignment='center')
        
        plt.hlines(y=start_positions[-1], xmin=-self.phantom.shape[0]*self.spacings[0]/2, xmax=self.phantom.shape[0]*self.spacings[0]/2, color='red')
        plt.annotate('Stop', (0, start_positions[-1]+10), horizontalalignment='center')

        plt.annotate(f'{len(start_positions)} scans required', xy=(0, (start_positions[0]+start_positions[-1])/2), horizontalalignment='center')
        plt.annotate('', xy=(40, start_positions[0]), xytext=(40, start_positions[-1]), arrowprops=dict(facecolor='black', shrink=0.05))

        plt.ylabel('scan z position [mm]')
        plt.xlabel('scan x position [mm]')

    def __repr__(self) -> str:
        repr = f'{self.__class__} {self.seriesname}'
        if self.recon is None:
            return repr
        repr += f'\nRecon: {self.recon.shape} {self.xcist.cfg.recon.fov/10} cm fov'
        if self.projections is None:
            return repr
        repr += f'\nProjections: {self.projections.shape}'
        return repr

    def run_scan(self, mA=200, kVp=120, startZ=None, endZ=None, views=None, verbose=False, slice_thickness=5):
        """
            Runs the CT simulation using the stored parameters.

            :param mA: x-ray source milliamps, increases x-ray flux linearly, $noise \propto 1/sqrt(mA)$
            :param kVp: x-ray source potential, increases x-ray flux nonlinearly and reduces contrast as increased
            :param startZ: optional starting table position in mm of the scan, see self.start_positions
            :param endZ: optional last position of scan in mm, see self.start_positions
            :param views: number of angular views, for testing this can be reduced but will produced aliasing streaks
            :param verbose: optional boolean, if True prints out status updates, if False they are suppressed.
        """
            # update parameters and raise Value Errors if needd
        self.xcist.cfg.protocol.mA = mA
        kVp_options = [80, 90, 100, 110, 120, 130, 140]
        if kVp not in kVp_options:
            raise ValueError(f'Selected kVP [{kVp}] not available, please choose from {kVp_options}')
        self.xcist.cfg.protocol.spectrumFilename = f'tungsten_tar7.0_{kVp}_filt.dat'
        
        self.start_positions = self.calculate_start_positions(slice_thickness)
        start_positions = self.start_positions
        
        if startZ:
            if startZ < start_positions.min():
                raise ValueError(f'startZ is outside the range of valid start positions: {self.start_positions}')
            start_positions = start_positions[start_positions>startZ]
        if endZ:
            if endZ > start_positions.max():
                raise ValueError(f'startZ is outside the range of valid start positions: {self.start_positions}')
            start_positions = start_positions[start_positions<endZ]

        if views:
            self.xcist.cfg.protocol.viewCount = views
            self.xcist.protocol.stopViewId = self.xcist.cfg.protocol.startViewId+self.xcist.cfg.protocol.viewCount-1
            self.xcist.cfg.protocol.viewsPerRotation =views
        
        self.results_dir = self.output_dir / 'simulations' / f'{self.patientid}'
        self.results_dir.mkdir(exist_ok=True, parents=True)    
        self.xcist.resultsName = str(self.results_dir / f'{self.patientid}_{mA}mA_{kVp}kV')
        self.xcist.protocol.spectrumFilename = f"tungsten_tar7.0_{int(kVp)}_filt.dat" # name of the spectrum file
        self.xcist.cfg.experimentDirectory = str(self.results_dir)
        
        recons = []
        for idx, table_position in enumerate(start_positions):
            print(f'scan: {idx+1}/{len(start_positions)}')
            self.xcist.resultsName = str(self.results_dir / f'{idx:03d}_{mA}mA_{kVp}kV') #keep projection data from each scan
            self.xcist.protocol.startZ = table_position
            self.xcist.run_all()
            self.run_recon(sliceThickness=slice_thickness)
            recons.append(self.recon)
        self.recon = np.concatenate(recons, axis=0)
        return self

    def run_recon(self, fov=None, sliceThickness=None, sliceCount=None, mu_water=None, preview=False):
        if sliceThickness:
            self.xcist.recon.sliceThickness = sliceThickness
        if mu_water:
            self.xcist.cfg.recon.mu = mu_water
        if not sliceCount:
            detector_width = self.xcist.scanner.detectorRowCount * self.xcist.scanner.detectorRowSize
            magnification = self.xcist.scanner.sdd / self.xcist.scanner.sid
            detector_width_at_isocenter = detector_width / magnification
            safe_width_at_isocenter = detector_width_at_isocenter - 2*self.xcist.scanner.detectorRowSize
            valid_slices = int(safe_width_at_isocenter // self.xcist.recon.sliceThickness)
            self.xcist.cfg.recon.sliceCount = valid_slices
        else:
            self.xcist.cfg.recon.sliceCount = sliceCount
        if fov:
            self.xcist.cfg.recon.fov = fov
    
        print(f'fov size: {self.xcist.cfg.recon.fov}')

        self.xcist.cfg.recon.displayImagePictures = preview
        recon.recon(self.xcist.cfg)
        self.recon = get_reconstructed_data(self.xcist)
        self.projections = get_projection_data(self.xcist)
        self.groundtruth = None
        self.I0 = self.xcist.cfg.protocol.mA
        self.nsims = 1
        return self
    
    def write_to_dicom(self, fname:str|Path, groundtruth=False) -> list[Path]:
        """
            write ct data to DICOM file, returns list of written dicom file names

            :param fname: filename to save image to (preferably with '.dcm` or related extension)
            :param groundtruth: Optional, whether to save the ground truth phantom image (no noise, blurring, or other artifacts).
                If True, 'self.groundtruth` is saved, if False (default) `self.recon` which contains blurring (and noise if 'add_noise`True)
            :returns: list[Path]

            Adapted from <https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144>
        """
        fpath = pydicom.data.get_testdata_file("CT_small.dcm")
        ds = pydicom.dcmread(fpath)
        # update meta info
        ds.Manufacturer = 'GE (simulated)'
        ds.ManufacturerModelName = 'LightSpeed 16 (simulated)'
        time = datetime.now()
        ds.InstanceCreationDate = time.strftime('%Y%m%d')
        ds.InstanceCreationTime = time.strftime('%H%M%S')
        ds.InstitutionName = 'FDA/CDRH/OSEL/DIDSR'
        ds.StudyDate = ds.InstanceCreationDate
        ds.StudyTime = ds.InstanceCreationTime
        
        ds.PatientName = self.patientname
        ds.SeriesNumber = self.seriesid
        
        ds.PatientAge = f'{int(self.age):03d}Y'
        ds.PatientID = f'{int(self.patientid):03d}'
        del(ds.PatientWeight)
        del(ds.ContrastBolusRoute)
        del(ds.ContrastBolusAgent)
        ds.ImageComments = f"effctive diameter [cm]: {self.patient_diameter/10}"
        ds.ScanOptions = self.xcist.cfg.protocol.scanTrajectory.upper()
        ds.ReconstructionDiameter = self.xcist.cfg.recon.fov
        ds.ConvolutionKernel = self.xcist.cfg.recon.kernelType
        ds.Exposure = self.xcist.cfg.protocol.mA
        
        # load image data
        ds.StudyDescription = f"{self.I0} photons " + self.seriesname + " " + ds.ConvolutionKernel + self.xcist.cfg.recon.reconType
        if self.recon.ndim == 2: self.recon = self.recon[None]
        nslices, ds.Rows, ds.Columns = self.recon.shape

        ds.SpacingBetweenSlices = ds.SliceThickness
        ds.DistanceSourceToDetector = self.xcist.cfg.scanner.sdd
        ds.DistanceSourceToPatient = self.xcist.cfg.scanner.sid
        
        ds.PixelSpacing = [self.xcist.cfg.recon.fov/self.xcist.cfg.recon.imageSize, self.xcist.cfg.recon.fov/self.xcist.cfg.recon.imageSize]
        ds.SliceThickness = ds.PixelSpacing[0]

        ds.KVP = self.kVp
        ds.StudyID = str(self.studyid)
        # series instance uid unique for each series
        end = ds.SeriesInstanceUID.split('.')[-1]
        new_end = str(int(end) + self.studyid)
        ds.SeriesInstanceUID = ds.SeriesInstanceUID.replace(end, new_end)
        
        # study instance uid unique for each series
        end = ds.StudyInstanceUID.split('.')[-1]
        new_end = str(int(end) + self.studyid)
        ds.StudyInstanceUID = ds.StudyInstanceUID.replace(end, new_end)
        ds.AcquisitionNumber = self.studyid

        fname = Path(fname)
        fname.parent.mkdir(exist_ok=True, parents=True)
        # saveout slices as individual dicom files
        fnames = []
        vol = self.groundtruth if groundtruth else self.recon
        if vol.ndim == 2: vol = vol[None]
        for slice_idx, array_slice in enumerate(vol):
            ds.InstanceNumber = slice_idx + 1 # image number
            # SOP instance UID changes every slice
            end = ds.SOPInstanceUID.split('.')[-1]
            new_end = str(int(end) + slice_idx + self.studyid + self.seriesid)
            ds.SOPInstanceUID = ds.SOPInstanceUID.replace(end, new_end)
            # MediaStorageSOPInstanceUID changes every slice
            end = ds.file_meta.MediaStorageSOPInstanceUID.split('.')[-1]
            new_end = str(int(end) + slice_idx + self.studyid + self.seriesid)
            ds.file_meta.MediaStorageSOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID.replace(end, new_end)
            # slice location and image position changes every slice
            ds.SliceLocation = self.nsims//2*ds.SliceThickness + slice_idx*ds.SliceThickness
            ds.ImagePositionPatient[-1] = ds.SliceLocation
            ds.ImagePositionPatient[0] = -ds.Rows//2*ds.PixelSpacing[0]
            ds.ImagePositionPatient[1] = -ds.Columns//2*ds.PixelSpacing[1]
            ds.ImagePositionPatient[2] = ds.SliceLocation
            ds.PixelData = array_slice.copy(order='C').astype('int16') - int(ds.RescaleIntercept)
            dcm_fname = fname.parent / f'{fname.stem}_{slice_idx:03d}{fname.suffix}' if nslices > 1 else fname
            fnames.append(dcm_fname)
            pydicom.write_file(dcm_fname, ds)
        return fnames

def center_crop(img, thresh=-800, rows=True, cols=True):
    cropped = img[img.mean(axis=1)>thresh, :]
    cropped = cropped[:, img.mean(axis=0)>thresh]
    return cropped

def center_crop_like(img, ref, thresh=-800):
    cropped = img[ref.mean(axis=1)>thresh, :]
    cropped = cropped[:, ref.mean(axis=0)>thresh]
    return cropped

from ipywidgets import interact, IntSlider

def scrollview(phantom):
    interact(lambda idx: ctshow(phantom[idx]), idx=IntSlider(value=phantom.shape[0]//2, min=0, max=phantom.shape[0]-1))