import pydicom
import numpy


def read_dicom(img_path: str, lut: bool = False) -> numpy.ndarray:
    dcm = pydicom.dcmread(img_path)
    if lut:
        img = apply_lut(dcm.pixel_array, dcm)
    else:
        img = dcm.pixel_array
    return img


def read_raw(img_path: str) -> numpy.ndarray | None:
    with open(img_path, "rb") as buffer:
        data_raw = buffer.read()
        if len(data_raw) <= 512:
            return None
        meta, vec_raw = data_raw[:256], data_raw[256:-256]
    cols = meta[12] + (meta[13] << 8)
    rows = meta[16] + (meta[17] << 8)
    img_raw = numpy.frombuffer(vec_raw, numpy.uint16).reshape((rows, cols))
    return img_raw


def apply_lut(img: numpy.ndarray, dcm: pydicom.dataset.FileDataset):
    dtype = img.dtype
    img = pydicom.pixel_data_handlers.util.apply_voi_lut(img, dcm)
    img = (img - img.min()) / (img.max())
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img
    img = (img * (2**16-1)).astype(dtype)
    return img
