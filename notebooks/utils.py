import matplotlib.pyplot as plt

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

def ctshow(img, window='soft tissues'):
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

    plt.imshow(img, cmap='gray', vmin=wl-ww/2, vmax=wl+ww/2)
    plt.xticks([])
    plt.yticks([])
    return

class CTobj():
    """
        A class to hold CT simulation data and run simulations

        :param phantom: phantom object to be scanned
        :param mA: Optional float, tube current
        :param fov: Optional float, reconstructed field of view (FOV) units mm
        :param framework: Optional, CT simulation framework options include `['CATSIM'] <https://github.com/JeffFessler/mirt>`_
        :returns: None
        
        See also <https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L19>
    """
    def __init__(self, phantom, mA=200, kVp=120, fov=None, patientname="default", patientid=0, age=0, studyname="default", studyid=0, seriesname="default", seriesid=0, framework='CATSIM') -> None:
        """Constructor method
        """
        self.phantom=phantom
        self.mA=mA
        self.kVp=kVp
        self.fov=fov
        self.age=age
        self.patientname=patientname or f'{patient_diameter/10} cm {phantom}'
        self.patientid=patientid
        self.studyname=studyname or self.patientname
        self.studyid=studyid
        self.seriesname=seriesname or f'{self.patientname} I0: {I0}'
        self.seriesid=seriesid
        self.framework=framework
        self.ndetectors=900
        self.nangles=1000
        self.detector_size=1
        self.sid = 540
        self.sdd = 950
        self.recon=None
        self.projections=None
        self.groundtruth=None
        self.patient_diameter = 18


    def __repr__(self) -> str:
        repr = f'{self.__class__} {self.seriesname}'
        if self.recon is None:
            return repr
        repr += f'\nRecon: {self.recon.shape} {self.fov/10} cm FOV'
        if self.projections is None:
            return repr
        repr += f'\nProjections: {self.projections.shape}'
        return repr

    def run(self, verbose=False):
        """
            Runs the CT simulation using the stored parameters.

            :param verbose: optional boolean, if True prints out status updates, if False they are suppressed. 
        """

        ct = run_simulation(ground_truth_image=self.phantom, output_dir=f'{self.patientname}', phantom_id=f'{self.patientid}', mA=self.mA, kVp=self.kVp, FOV=self.fov)
        # mirt_sim(phantom=phantom, patient_diameter=patient_diameter, reference_diameter=reference_diameter, reference_fov=reference_fov,
        #                    I0=I0, nb=nb, na=na, ds=ds, sid=sid, sdd=sdd, offset_s=offset_s, down=down, has_bowtie=has_bowtie,
        #                    add_noise=add_noise, aec_on=aec_on, nx=nx, fov=fov, fbp_kernel=fbp_kernel, nsims=nsims, lesion_diameter=lesion_diameter, verbose=verbose)
        self.recon = get_reconstructed_data(ct).transpose(2, 0, 1)
        self.projections = get_projection_data(ct)
        self.groundtruth = None
        self.I0 = ctobj.mA
        self.nsims = 1
        self.matrix_size = ctobj.recon.shape[-1]
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
        ds.Manufacturer = 'Siemens (simulated)'
        ds.ManufacturerModelName = 'Definition AS+ (simulated)'
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
        ds.ScanOptions = 'AXIAL MODE'
        ds.ReconstructionDiameter = self.fov
        ds.ConvolutionKernel ='fbp D45'
        ds.Exposure = self.mA
        
        # load image data
        ds.StudyDescription = f"{self.I0} photons " + self.seriesname + " " + ds.ConvolutionKernel
        if self.recon.ndim == 2: self.recon = self.recon[None]
        nslices, ds.Rows, ds.Columns = self.recon.shape
        assert nslices == self.nsims
        ds.SpacingBetweenSlices = ds.SliceThickness
        ds.DistanceSourceToDetector = self.sdd
        ds.DistanceSourceToPatient = self.sid
        
        ds.PixelSpacing = [self.fov/self.matrix_size, self.fov/self.matrix_size]
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