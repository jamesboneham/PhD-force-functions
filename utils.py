import numpy as np
import xarray as xr
import numba as nb
import bottleneck as bn

def sortdims(da):
    """
    Rearrange dimensions on input DataArray da to allow for consistency when
    converting to and from numpy arrays.
    """
    return da.transpose(*sorted(da.dims, key=lambda x:{"T":1,"Z":2,"t":2,"Y":3,"X":4}[x[0]]))

def xr2np(da):
    """
    Convert a DataArray into a numpy array, with consistent ordering of dimensions.
    """
    return da.pipe(sortdims).values

def invert(arr):
    """
    Calculate 1/arr, ignoring zeros without throwing an error or warning.
    """
    return np.divide(1,arr,where=(arr!=0))

def npshift(array, num, axis=-1, fill_value=np.nan):
    """
    Shift numpy array: analogous to xarray DataArray.shift method.
    """
    result = np.empty_like(array)
    arr = np.swapaxes(array, axis, -1)
    res = np.swapaxes(result, axis, -1)
    if num > 0:
        res[..., :num] = fill_value
        res[..., num:] = arr[..., :-num]
    elif num < 0:
        res[..., num:] = fill_value
        res[..., :num] = arr[..., -num:]
    else:
        res[..., :] = arr
    return result

def getHFac(array_or_shape, grd, option="min"):
    """
    Get the fractional size of cells for a given array or shape consistent with
    grid descriptors in DataArray grd.
    """
    if option == "full":
        return getHFac(array_or_shape, grd, option="min").where(lambda x: x == 1, 0.)
    else:
        pass
    Z, Zp1, Y, Yp1, X, Xp1 = (xr2np(grd[i]) for i in ("Z", "Zp1", "Y", "Yp1", "X", "Xp1"))
    try:
        arr = xr2np(array_or_shape.isel(T=0) if "T" in array_or_shape.dims else array_or_shape)
    except AttributeError:
        try:
            arr = array_or_shape[0, ...] if array_or_shape.ndim == 4 else array_or_shape
        except AttributeError:
            try:
                arr = np.empty(array_or_shape)
            except TypeError:
                arr = np.empty([len(grd[i]) for i in array_or_shape])
    HC = xr2np(grd.Depth)
    if arr.shape[-3] == len(Z):
        hC = grd.HFacC
    elif arr.shape[-3] == len(Zp1):
        drF, drC = (xr2np(dr)[:, None, None] for dr in (grd.drF, grd.drC))
        rF = xr2np(grd.RF)[:, None, None]
        rC = np.pad(xr2np(grd.RC), ((1, 1)), constant_values=(rF.max(), rF.min()))[:, None, None]
        dzC = np.diff(np.minimum(-rC, HC), axis=-3)
        hC = dzC/drC
    else:
        raise ValueError
    opt_func = {"min": lambda x: np.minimum.reduce(x),
                "max": lambda x: np.maximum.reduce(x),
                "avg": lambda x: (1/len(x))*np.add.reduce(x)}[option]
    loc_func = {
        (len(Y), len(X)): lambda x: x,
        (len(Yp1), len(X)): lambda x: np.pad(opt_func((x[:, 1:, :], x[:, :-1, :])),
                                             ((0, 0), (1, 1), (0, 0))),
        (len(Y), len(Xp1)): lambda x: np.pad(opt_func((x[:, :, 1:], x[:, :, :-1])),
                                             ((0, 0), (0, 0), (1, 1))),
        (len(Yp1), len(Xp1)): lambda x: np.pad(
            opt_func((x[:, 1:, 1:], x[:, :-1, 1:], x[:, 1:, :-1], x[:, :-1, :-1])),
            ((0, 0), (1, 1), (1, 1))
        )}[arr.shape[-2:]]
    hfac = loc_func(hC)
    zdim = {len(Z): "Z", len(Zp1): "Zp1"}[hfac.shape[-3]]
    ydim = {len(Y): "Y", len(Yp1): "Yp1"}[hfac.shape[-2]]
    xdim = {len(X): "X", len(Xp1): "Xp1"}[hfac.shape[-1]]
    return xr.DataArray(hfac, dims=(zdim, ydim, xdim),
                        coords={zdim: grd[zdim], ydim: grd[ydim], xdim: grd[xdim]})


def getH(array_or_shape, grd, option="min"):
    """
    Get DataArray of depth for a given array or shape consistent with grid
    descriptors in DataArray grd.
    """
    Y, Yp1, X, Xp1 = (xr2np(grd[i]) for i in ("Y", "Yp1", "X", "Xp1"))
    try:
        arr = xr2np(array_or_shape.isel(T=0) if "T" in array_or_shape.dims else array_or_shape)
    except AttributeError:
        try:
            arr = array_or_shape[0, ...] if array_or_shape.ndim == 4 else array_or_shape
        except AttributeError:
            try:
                arr = np.empty(array_or_shape)
            except TypeError:
                arr = np.empty([len(grd[i]) for i in array_or_shape])
    ylen, xlen = arr.shape[-2:]
    HC = xr2np(grd.Depth)
    opt_func = {"min": lambda x: np.minimum.reduce(x),
                "max": lambda x: np.maximum.reduce(x),
                "avg": lambda x: (1/len(x))*np.add.reduce(x)}[option]
    loc_func = {
        (len(Y), len(X)): lambda x: x,
        (len(Yp1), len(X)): lambda x: np.pad(opt_func((x[1:, :], x[:-1, :])),
                                             ((1, 1), (0, 0))),
        (len(Y), len(Xp1)): lambda x: np.pad(opt_func((x[:, 1:], x[:, :-1])),
                                             ((0, 0), (1, 1))),
        (len(Yp1), len(Xp1)): lambda x: np.pad(
            opt_func((x[1:, 1:], x[:-1, 1:], x[1:, :-1], x[:-1, :-1])),
            ((1, 1), (1, 1))
        )}[(ylen, xlen)]
    H = loc_func(HC)
    ydim = {len(Y): "Y", len(Yp1): "Yp1"}[ylen]
    xdim = {len(X): "X", len(Xp1): "Xp1"}[xlen]
    return xr.DataArray(H, dims=(ydim, xdim), coords={ydim: grd[ydim], xdim: grd[xdim]})

def pushnan(arr, axis, limit=None, bothdir=False):
    """
    Fill missing values up to limit along axis using preceding non-missing value.
    """
    axis = axis if axis >= 0 else arr.ndim+axis
    limit = arr.shape[axis] if limit is None else limit
    if bothdir is True:
        out = pushnan(pushnan(arr, axis, limit=limit), axis, limit=-limit)
    elif limit < 0:
        out = np.flip(pushnan(np.flip(arr, axis=axis), axis, limit=-limit), axis=axis)
    else:
        out = bn.push(arr, n=limit, axis=axis)
    return out

def bar(arr, axis=-1, mask=True, expand=False):
    """
    MITgcm-consistent averaging operator
    """
    axis = axis if axis >= 0 else arr.ndim+axis
    if expand is True:
        pad_widths = [(0,1) if i == axis else (0,0) for i in range(arr.ndim)]
        out = bn.move_mean(np.pad(np.where(mask, arr, np.nan), pad_widths, mode="edge"),
                           window=2, min_count=1, axis=axis)
    else:
        axis = arr.ndim + axis if axis < 0 else axis
        sl = tuple([slice(1 if i == axis else None, None, None)
                    for i in range(arr.ndim)])
        out = bn.move_mean(np.where(mask, arr, np.nan),
                           window=2, min_count=1, axis=axis)[sl]
    return out

def delta(arr, axis=-1, mask=True, expand=False, extrapolatebc=False):
    """
    MITgcm-consistent differencing operator
    """
    axis = axis if axis >= 0 else arr.ndim+axis
    a = np.where(mask, arr, np.nan)
    expandkw = {"prepend":np.nan, "append":np.nan} if expand is True else {}
    if extrapolatebc is True:
        out = pushnan(np.diff(a, axis=axis, **expandkw), axis, limit=1, bothdir=True)
        m = np.logical_and(np.isfinite(delta(arr, axis=axis, mask=mask,
                                             expand=expand, extrapolatebc=False)),
                           np.isnan(out))
        out[m] = 0.
    elif extrapolatebc is False:
        out = np.diff(pushnan(a, axis, limit=1, bothdir=True), axis=axis, **expandkw)
    else:
        raise ValueError
    return out

def fillnan(arr, n=0, inplace=False):
    """
    Fill missing values with n.
    """
    if inplace is True:
        out = bn.replace(arr, np.nan, n)
    else:
        arr_copy = np.empty_like(arr)
        arr_copy[:] = arr[:]
        out = bn.replace(arr_copy, np.nan, n)
    return out

def nanfill(arr, n=0, inplace=False):
    """
    Fill values equal to n with np.nan.
    """
    if inplace is True:
        out = bn.replace(arr, n, np.nan)
    else:
        arr_copy = np.empty_like(arr)
        arr_copy[:] = arr[:]
        out = bn.replace(arr_copy, n, np.nan)
    return out
