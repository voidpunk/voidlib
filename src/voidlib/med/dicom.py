from matplotlib import pyplot
import pydicom
import numpy
import cv2


def read_dicom(
        img_path: str,
        lut: bool = False
    ) -> numpy.ndarray:
    # read the dicom
    dcm = pydicom.dcmread(img_path)
    # apply the normalized LUT
    if lut:
        img = apply_lut(dcm.pixel_array, dcm)
    else:
        img = dcm.pixel_array
    return img


def read_raw(
        img_path: str
    ) -> numpy.ndarray | None:
    # read the raw file
    with open(img_path, "rb") as buffer:
        data_raw = buffer.read()
        # check if it's a valid file
        if len(data_raw) <= 512:
            return None
        # separate the metadata from the image data
        meta, vec_raw = data_raw[:256], data_raw[256:-256]
    # get cols & rows number
    cols = meta[12] + (meta[13] << 8)
    rows = meta[16] + (meta[17] << 8)
    # create an uint16 array from the binary data
    img_raw = numpy.frombuffer(vec_raw, numpy.uint16).reshape((rows, cols))
    return img_raw


def apply_lut(
        img: numpy.ndarray,
        dcm: pydicom.dataset.FileDataset
    ) -> numpy.ndarray:
    # apply LUT in DICOM
    img = pydicom.pixel_data_handlers.util.apply_voi_lut(img, dcm)
    # normalize the image
    img = (img - img.min()) / (img.max())
    # invert the image if not yet
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img
    # convert back to full uint16 range
    img = (img * (2**16-1)).astype("uint16")
    return img


def plot_histogram(
        input_img: numpy.ndarray | str,
        show_image: bool = True
    ) -> None:
    # check if it's a path or an array
    if isinstance(input_img, str):
        img = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
    else:
        img = input_img
    # check the image dtype to determine the histogram bins number
    if img.dtype == "uint16":
        max_val = 2**16-1
    elif img.dtype == "uint8":
        max_val = 2**8-1
    # calculate the histogram
    hist = cv2.calcHist([img], [0], None, [max_val], [0, max_val]).ravel()
    if show_image:
        # show the histogram & image
        fig, axs = pyplot.subplots(1, 2, figsize=(12,6))
        axs[0].plot(hist)
        axs[1].imshow(img, cmap="gray")
    else:
        # show only the histogram
        pyplot.figure(figsize=(6,6))
        pyplot.plot(hist)
    pyplot.show()
