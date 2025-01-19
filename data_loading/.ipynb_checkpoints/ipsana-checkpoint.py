import numpy as np
import logging
import psana
from psana import setOption
from psana import EventId
from PSCalib.GeometryAccess import GeometryAccess

logger = logging.getLogger(__name__)

class PsanaImg:
    """
    It serves as an image accessing layer based on the data management system
    psana in LCLS.  
    """

    def __init__(self, exp, run, mode, detector_name):

        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()
        # Set up detector
        self.detector = psana.Detector(detector_name)
        self.detector_name = detector_name

        # Set image reading mode
        self.read = { "raw"   : self.detector.raw,
                      "calib" : self.detector.calib,
                      "image" : self.detector.image,
                      "mask"  : self.detector.mask, }

    def __len__(self):
        return len(self.timestamps)


    def get(self, event_num, id_panel = None, mode = "calib"): 

        if event_num >= self.__len__():
            raise ValueError(f"Event number {event_num} is out of range!!!")
        
        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[event_num]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Only two modes are supported...
        assert mode in ("raw", "calib", "image"), \
            f"Mode {mode} is not allowed!!!  Only 'raw', 'calib' and 'image' are supported."

        # Fetch image data based on timestamp from detector...
        data = self.read[mode](event)
        img  = data[int(id_panel)] if id_panel is not None else data

        return img
    
    def get_timestamp(self, event_num):
        return self.timestamps[event_num]


    def assemble(self, multipanel = None, mode = "image", fake_event_num = 0):
        # Set up a fake event_num...
        event_num = fake_event_num

        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Fetch image data based on timestamp from detector...
        img = self.read[mode](event, multipanel)

        return img


    def create_bad_pixel_mask(self):
        return self.read["mask"](self.run_current, calib       = True,
                                                   status      = True,
                                                   edges       = True,
                                                   central     = True,
                                                   unbond      = True,
                                                   unbondnbrs  = True,
                                                   unbondnbrs8 = False).astype(np.uint16)

#### Miscellaneous functions ####

def retrieve_pixel_index_map(geom):
    """
    Retrieve a pixel index map that specifies the relative arrangement of
    pixels on an LCLS detector.

    Parameters
    ----------
    geom : string or GeometryAccess Object
        if str, full path to a psana *-end.data file
        else, a PSCalib.GeometryAccess.GeometryAccess object

    Returns
    -------
    pixel_index_map : numpy.ndarray, 4d or 5d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)
                           shape (pidx1, pidx2, fs_shape, ss_shape, 2)
    """
    if type(geom) == str:
        geom = GeometryAccess(geom)

    temp_index = [np.asarray(t) for t in geom.get_pixel_coord_indexes()]
    pixel_index_map = np.zeros((np.array(temp_index).shape[2:]) + (2,))
    pixel_index_map[...,0] = temp_index[0][0]
    pixel_index_map[...,1] = temp_index[1][0]
    
    return pixel_index_map.astype(np.int64)

def assemble_image_stack_batch(image_stack, pixel_index_map):
    """
    Assemble the image stack to obtain a 2D pattern according to the index map.
    Either a batch or a single image can be provided. Modified from skopi.

    Parameters
    ----------
    image_stack : numpy.ndarray, 3d or 4d
        stack of images, shape (n_images, n_panels, fs_panel_shape, ss_panel_shape)
        or (n_panels, fs_panel_shape, ss_panel_shape)
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)

    Returns
    -------
    images : numpy.ndarray, 3d
        stack of assembled images, shape (n_images, fs_panel_shape, ss_panel_shape)
        of shape (fs_panel_shape, ss_panel_shape) if ony one image provided
    """
    multiple_panel_dimensions = False
    if len(image_stack.shape) == 3:
        image_stack = np.expand_dims(image_stack, 0)

    if len(pixel_index_map.shape) == 5:
        multiple_panel_dimensions = True
        
    # get boundary
    index_max_x = np.max(pixel_index_map[..., 0]) + 1
    index_max_y = np.max(pixel_index_map[..., 1]) + 1
    # get stack number and panel number
    stack_num = image_stack.shape[0]

    # set holder
    images = np.zeros((stack_num, index_max_x, index_max_y))

    if multiple_panel_dimensions:
        pdim1 = pixel_index_map.shape[0]
        pdim2 = pixel_index_map.shape[1]
        for i in range(pdim1):
            for j in range(pdim2):
                x = pixel_index_map[i, j, ..., 0]
                y = pixel_index_map[i, j, ..., 1]
                idx = i*pdim2 + j
                images[:, x, y] = image_stack[:, idx]
    else:
        panel_num = image_stack.shape[1]
        # loop through the panels
        for l in range(panel_num):
            x = pixel_index_map[l, ..., 0]
            y = pixel_index_map[l, ..., 1]
            images[:, x, y] = image_stack[:, l]

    if images.shape[0] == 1:
        images = images[0]

    return images

def disassemble_image_stack_batch(images, pixel_index_map):
    """
    Diassemble a series of 2D diffraction patterns into their consituent panels. 
    Function modified from skopi.

    Parameters
    ----------
    images : numpy.ndarray, 3d
        stack of assembled images, shape (n_images, fs_panel_shape, ss_panel_shape)
        of shape (fs_panel_shape, ss_panel_shape) if ony one image provided
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)

    Returns
    -------
    image_stack_batch : numpy.ndarray, 3d or 4d 

        stack of images, shape (n_images, n_panels, fs_panel_shape, ss_panel_shape)
        or (n_panels, fs_panel_shape, ss_panel_shape)
    """
    multiple_panel_dimensions = False
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)

    if len(pixel_index_map.shape) == 5:
        multiple_panel_dimensions = True

    if multiple_panel_dimensions:
        ishape = images.shape[0]
        (pdim1, pdim2, fs_shape, ss_shape) = pixel_index_map.shape[:-1]
        image_stack_batch = np.zeros((ishape, pdim1*pdim2, fs_shape, ss_shape))
        for i in range(pdim1):
            for j in range(pdim2):
                x = pixel_index_map[i, j, ..., 0]
                y = pixel_index_map[i, j, ..., 1]
                idx = i*pdim2 + j
                image_stack_batch[:, idx] = images[:, x, y]
    else:
        image_stack_batch = np.zeros((images.shape[0],) + pixel_index_map.shape[:3])
        for panel in range(pixel_index_map.shape[0]):
            idx_map_1 = pixel_index_map[panel, :, :, 0]
            idx_map_2 = pixel_index_map[panel, :, :, 1]
            image_stack_batch[:, panel] = images[:, idx_map_1, idx_map_2]

    if image_stack_batch.shape[0] == 1:
        image_stack_batch = image_stack_batch[0]

    return image_stack_batch

#### binning methods ####

def bin_data(arr, bin_factor, det_shape=None):
    """
    Bin detector data by bin_factor through averaging.
    Retrieved from
    https://github.com/apeck12/cmtip/blob/main/cmtip/prep_data.py

    :param arr: array shape (n_images, n_panels, panel_shape_x, panel_shape_y)
      or if det_shape is given of shape (n_images, 1, n_pixels_per_image)
    :param bin_factor: how may fold to bin arr by
    :param det_shape: tuple of detector shape, optional
    :return arr_binned: binned data of same dimensions as arr
    """
    # reshape as needed
    if det_shape is not None:
        arr = np.array([arr[i].reshape(det_shape) for i in range(arr.shape[0])])

    n, p, y, x = arr.shape

    # ensure that original shape is divisible by bin factor
    assert y % bin_factor == 0
    assert x % bin_factor == 0

    # bin each panel of each image
    binned_arr = (
        arr.reshape(
            n,
            p,
            int(y / bin_factor),
            bin_factor,
            int(x / bin_factor),
            bin_factor,
        )
        .mean(-1)
        .mean(3)
    )

    # if input data were flattened, reflatten
    if det_shape is not None:
        flattened_size = np.prod(np.array(binned_arr.shape[1:]))
        binned_arr = binned_arr.reshape((binned_arr.shape[0], 1) + (flattened_size,))

    return binned_arr

def bin_pixel_index_map(arr, bin_factor):
    """
    Bin pixel_index_map by bin factor.
    Retrieved from
    https://github.com/apeck12/cmtip/blob/main/cmtip/prep_data.py

    :param arr: pixel_index_map of shape (n_panels, panel_shape_x, panel_shape_y, 2)
    :param bin_factor: how may fold to bin arr by
    :return binned_arr: binned pixel_index_map of same dimensions as arr
    """
    arr = np.moveaxis(arr, -1, 0)
    if bin_factor > 1:
        arr = np.minimum(arr[..., ::bin_factor, :], arr[..., 1::bin_factor, :])
        arr = np.minimum(arr[..., ::bin_factor], arr[..., 1::bin_factor])
        arr = arr // bin_factor

    return np.moveaxis(arr, 0, -1)
