import numpy as np
import xarray as xr
import numba as nb
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from utils import (xr2np, invert, npshift, getHFac,
                   getH, sortdims, pushnan, bar,
                   delta, fillnan, nanfill)

def getF_rho(ds, **kwargs):
    """
    Given an MITgcm output xarray dataset, returns an xarray dataset containing
    viscous, Coriolis, wind-stress, inertial, and buoyancy forces in the x, y,
    and z directions.
    """
    f_ds = xr.Dataset(coords={**ds.UVEL.coords, **ds.VVEL.coords, **ds.Zp1.coords})
    guZero = xr.zeros_like(ds.UVEL).assign_attrs(units='m/s^2')
    gvZero = xr.zeros_like(ds.VVEL).assign_attrs(units='m/s^2')
    gwZero = xr.zeros_like(ds.T * ds.Zp1 * ds.Depth).assign_attrs(units='m/s^2')
    f_ds["viscx"] = ((ds.Um_Diss + ds.Um_ImplD)
                     .rename("Gu_Viscosity")
                     .assign_attrs(description='U momentum tendency from Dissipation',
                                   units='m/s^2'))
    f_ds["viscy"] = ((ds.Vm_Diss + ds.Vm_ImplD)
                     .rename("Gv_Viscosity")
                     .assign_attrs(description='V momentum tendency from Dissipation',
                                   units='m/s^2'))
    f_ds["corix"] = (ds.Um_Cori
                     .rename("Gu_Coriolis Force")
                     .assign_attrs(description='U momentum tendency from Coriolis',
                                   units='m/s^2'))
    f_ds["coriy"] = (ds.Vm_Cori
                     .rename("Gv_Coriolis Force")
                     .assign_attrs(description='V momentum tendency from Coriolis',
                                   units='m/s^2'))
    f_ds["windx"] = (ds.Um_Ext
                     .rename("Gu_Wind Stress")
                     .assign_attrs(description='U momentum tendency from Wind',
                                   units='m/s^2'))
    f_ds["windy"] = (ds.Vm_Ext
                     .rename("Gv_Wind Stress")
                     .assign_attrs(description='V momentum tendency from Wind',
                                   units='m/s^2'))
    f_ds["inrtlx"] = ((ds.Um_Advec - ds.Um_Cori)
                      .rename("Gu_Inertial Forces")
                      .assign_attrs(description='U momentum tendency from Inertia',
                                    units='m/s^2'))
    f_ds["inrtly"] = ((ds.Vm_Advec - ds.Vm_Cori)
                      .rename("Gv_Inertial Forces")
                      .assign_attrs(description='V momentum tendency from Inertia',
                                    units='m/s^2'))
    f_ds["buoyx"] = (guZero.rename("Gu_Buoyancy Force")
                     .assign_attrs(description="U momentum tendency from Buoyancy Force"))
    f_ds["buoyy"] = (gvZero.rename("Gv_Buoyancy Force")
                     .assign_attrs(description="V momentum tendency from Buoyancy Force"))
    f_ds["buoyzC"] = ((-ds.RHOAnoma.where(ds.HFacC) * ds.f_data.gravity / ds.f_data.rhonil)
                      .fillna(0.)
                      .rename("Gw_Buoyancy Force")
                      .assign_attrs(description="W momentum tendency from Buoyancy Force"))
    fbuoy = bar(xr2np(-ds.RHOAnoma.where(ds.HFacC) * ds.f_data.gravity / ds.f_data.rhonil),
                axis=-3, expand=True)
    f_ds["buoyzF"] = (gwZero.pipe(sortdims)
                      .pipe(lambda x: x + fillnan(fbuoy, n=0, inplace=True))
                      .rename("Gw_Buoyancy Force")
                      .assign_attrs(description="W momentum tendency from Buoyancy Force"))
    f_ds["phix"] = (ds.Um_dPhiX
                    .rename("Gu_Pressure")
                    .assign_attrs(description='U momentum tendency from Pressure',
                                  units='m/s^2'))
    f_ds["phiy"] = (ds.Vm_dPhiY
                    .rename("Gv_Pressure")
                    .assign_attrs(description='V momentum tendency from Pressure',
                                  units='m/s^2'))
    f_ds["totx"] = (ds.TOTUTEND.pipe(lambda x: x/86400.)
                    .rename("Gu_Total Force")
                    .assign_attrs(description='Total U momentum tendency',
                                  units='m/s^2'))
    f_ds["toty"] = (ds.TOTVTEND.pipe(lambda x: x/86400.)
                    .rename("Gv_Total Force")
                    .assign_attrs(description='Total V momentum tendency',
                                  units='m/s^2'))
    return f_ds

def getphiBT_closure(ds, h_opt="max", **kw):
    """
    Takes an MITgcm output xarray dataset, as well as h_opt (which controls how
    to handle cell boundaries). Returns a closure which calculates the barotropic
    pressure from a given force dataset, force name and timestep.
    """
    def get_locals():
        """
        Calculates and returns the LU factorisation of the domain matrix, as
        well as the grid data required by the solver.
        """
        dyG, dxG, dyC, dxC = [xr2np(ds[i]) for i in ("dyG", "dxG", "dyC", "dxC")]
        drF = xr2np(ds.drF)[:,None,None]
        dzS = xr2np(getHFac(("Z", "Yp1", "X"), ds, option=h_opt))*drF
        dzW = xr2np(getHFac(("Z", "Y", "Xp1"), ds, option=h_opt))*drF
        dzC_0 = xr2np(getHFac(("Z", "Y", "X"), ds, option=h_opt))*drF
        dzC = xr2np(getHFac(("Zp1", "Y", "X"), ds, option=h_opt)*ds.drC)
        HS = dzS.sum(axis=0)
        HW = dzW.sum(axis=0)
        H = dzC.sum(axis=0)
        z = -dzC.cumsum(axis=0)[:-1,:,:]
        HSdx_dy = HS*dxG/dyC
        HWdy_dx = HW*dyG/dxC
        m0 = H!=0
        mW = np.logical_and(m0, npshift(m0, 1, axis=-1, fill_value=False))
        mE = np.logical_and(m0, npshift(m0, -1, axis=-1, fill_value=False))
        mS = np.logical_and(m0, npshift(m0, 1, axis=-2, fill_value=False))
        mN = np.logical_and(m0, npshift(m0, -1, axis=-2, fill_value=False))
        mNaN = ~m0
        mFx = xr2np(getH(dxC.shape, ds, option="max")) != 0
        mFy = xr2np(getH(dyC.shape, ds, option="max")) != 0
        cW = np.where(mW, HWdy_dx[:,:-1], 0)
        cE = np.where(mE, HWdy_dx[:,1:], 0)
        cS = np.where(mS, HSdx_dy[:-1, :], 0)
        cN = np.where(mN, HSdx_dy[1:, :], 0)
        c0 = np.where(m0, -(cW+cE+cS+cN), 1)
        Nj, Ni = c0.shape
        diags = (
            cS.flatten()[Ni:], cW.flatten()[1:], c0.flatten(),
            cE.flatten()[:-1], cN.flatten()[:-Ni]
        )
        offsets = (-Ni, -1, 0, 1, Ni)
        A = sp.diags(diags, offsets=offsets, format="csc")
        A_dtype = A.dtype
        LU = spl.factorized(A)
        return dyG, dxG, dzS, dzW, mNaN, mFx, mFy, LU, z, dzC_0, H, dxC, dyC, A_dtype
    dyG, dxG, dzS, dzW, mNaN, mFx, mFy, LU, z, dzC_0, H, dxC, dyC, A_dtype = get_locals()
    def get_bc_component(fz_np):
        """
        Calculates the baroclinic component from the vertical force.
        """
        zfz_int = (z*fz_np*dzC_0).sum(axis=-3)
        mC = dzC_0.sum(axis=-3) != 0
        term_x = (delta(zfz_int, mask=mC, axis=-1, expand=True)
                  - dzW.sum(axis=-3)*delta(zfz_int*invert(H), mask=mC, axis=-1, expand=True))*dyG*invert(dxC)
        term_y = (delta(zfz_int, mask=mC, axis=-2, expand=True)
                  - dzS.sum(axis=-3)*delta(zfz_int*invert(H), mask=mC, axis=-2, expand=True))*dxG*invert(dyC)
        return term_x, term_y
    def solve(fx_np, fy_np, fz_np):
        """
        Given arrays of the forces in the x, y, and z directions, returns a
        numpy array of the barotropic pressure field arising from that force.
        """
        zfz_x, zfz_y = get_bc_component(fz_np)
        Fxdy, Fydx = ((fx_np*dzW).sum(axis=-3)*dyG, (fy_np*dzS).sum(axis=-3)*dxG)
        RHS = fillnan(delta(Fxdy - zfz_x, mask=mFx, axis=-1)
                      + delta(Fydx - zfz_y, mask=mFy, axis=-2))
        data = LU(RHS.astype(A_dtype).flatten()).reshape(RHS.shape)
        data[mNaN] = np.nan
        return data
    def closure(force_ds, force, T=None):
        """
        Takes a force dataset, a force name, and a timestep. Returns an xarray
        dataArray of the pressure field at that time (or range of times).
        """
        if T is None:
            fx, fy = force_ds[f"{force}x"], force_ds[f"{force}y"]
            fz = force_ds.get(f"{force}zC", xr.zeros_like(force_ds["buoyzC"]))
        else:
            fx, fy = force_ds[f"{force}x"].isel(T=T), force_ds[f"{force}y"].isel(T=T)
            fz = force_ds.get(f"{force}zC", xr.zeros_like(force_ds["buoyzC"])).isel(T=T)
            pass
        out = (xr.apply_ufunc(solve, fx, fy, fz,
                             input_core_dims=[["Z", "Y", "Xp1"], ["Z", "Yp1", "X"],
                                              ["Z", "Y", "X"]],
                             output_core_dims=[("Y", "X")],
                             vectorize=True, dask="parallelized")
               .rename(r"$\phi_{\mathrm{BT}}$" + f" ({fx.name[3:]})")
               .assign_attrs(description=("Barotropic Pressure due to " + f"{fx.name[3:]}"),
                             units=r"m$^{2}$s$^{-2}$"))
        return out
    if kw.get("npclosure", False) is True:
        outfun = solve
    else:
        outfun = closure
        pass
    return outfun

def getAOTx_closure(ds):
    """
    Takes an MITgcm output xarray dataset. Returns a closure which calculates
    the meridional overturning force function.
    """
    def get_locals():
        """
        Calculates and returns the LU factorisation of the domain matrix, as
        well as the grid data required by the solver.
        """
        HS = getH(("Yp1", "X"),ds,"min").data
        dyU = xr2np(ds.dyU)
        dxV = xr2np(ds.dxV)
        dyF = xr2np(ds.dyF)
        dxF = xr2np(ds.dxF)
        rAs = 1/xr2np(ds.rAs)
        # hfc = xr2np(ds.HFacC)
        dzf = np.diff(np.minimum(np.pad(ds.drF.data, (1,0))[:,None,None].cumsum(axis=0), HS),
                    axis=0)
        dzc = np.diff(np.minimum(np.pad(ds.drC.data, (1,0))[:,None,None].cumsum(axis=0), HS),
                    axis=0)
        dzfkm1 = np.pad(dzf, ((1,0),(0,0),(0,0)))
        dzfk = np.pad(dzf, ((0,1),(0,0),(0,0)))
        dyC = xr2np(ds.dyC)
        zinds = np.indices(dzc.shape)[0]
        bottom_inds = fillnan(pushnan(np.where(dzc==0, zinds-1, np.nan)[::-1,...], axis=0)[-1,...],
                            n=dzc.shape[0]-1)
        m0 = ~np.logical_or(zinds<=0, zinds>=bottom_inds)
        mNaN = dzc==0
        c_up = np.where(m0, invert(dzfkm1*dzc), 0)
        c_dn = np.where(m0, invert(dzfk*dzc), 0)
        c_0 = np.where(m0, -(c_up + c_dn), 1)
        _, Nj, Ni = c_0.shape
        diags = (
            c_up.flatten()[Ni*Nj:], c_0.flatten(), c_dn.flatten()[:-Ni*Nj]
        )
        offsets = (-Ni*Nj, 0, Ni*Nj)
        A = sp.diags(diags, offsets=offsets, format="csc")
        A_dtype = A.dtype
        LU = spl.factorized(A)
        return dyU, dxV, dyF, dxF, rAs, dzf, dzc, dyC, m0, mNaN, LU, A_dtype
    dyU, dxV, dyF, dxF, rAs, dzf, dzc, dyC, m0, mNaN, LU, A_dtype = get_locals()
    xrdims = ("Zp1", "Yp1", "X")
    xrcoords = {i:ds.coords[i] for i in xrdims}
    def getRHS(fy, fz):
        """
        Calculates the RHS of the elliptic equation given the meridional and
        vertical forces.
        """
        fy = np.where(dzf!=0, fy, np.nan)
        fz = nanfill(fz, n=0)
        RHS = -(fillnan(delta(fz, axis=-2, expand=True))*invert(dyC)
                + fillnan(delta(fy, axis=-3, expand=True))*invert(dzc))
        return RHS
    def hLap(a):
        """
        Calculates the finite-volume horizontal Laplacian consistent with the
        (non-uniform) grid.
        """
        out = rAs*(delta(dyU*delta(a, axis=-1, expand=True)/dxV, axis=-1, expand=False)
                + delta(dxF*delta(a, axis=-2, expand=False)/dyF, axis=-2, expand=True))
        return fillnan(out)*m0
    def iterate(RHS, Aprev=None, nmax=50, rtol=1E-13, n=0):
        """
        Recursively improve the solution (to account for the small contribution
        of the horizontal Laplacian).
        """
        Aprev = LU(RHS.flatten()).reshape(RHS.shape) if Aprev is None else Aprev
        Anew = LU((RHS-hLap(Aprev)).flatten()).reshape(RHS.shape)
        r2 = np.sqrt((Anew-Aprev)[m0]**2).sum()*invert(np.sqrt(Anew[m0]**2).sum())
        if n >= nmax:
            print(f"Failed to converge past rtol={r2} after n={n} iterations")
            return Anew
        elif r2 > rtol:
            return iterate(RHS, Aprev=Anew, nmax=nmax, rtol=rtol, n=(n+1))
        else:
            print(f"Converged to rtol={r2} in n={n} iterations")
            pass
        return Anew
    def solve(fy, fz, precise=False, nmax=50, rtol=1E-13):
        """
        Given arrays of the forces in the y and z directions, returns a numpy
        array of the meridional overturning force function arising from that
        force.
        """
        RHS = getRHS(fy, fz).astype(A_dtype)
        data = LU(RHS.flatten()).reshape(RHS.shape)
        if precise is True:
            data = iterate(RHS, Aprev=data, nmax=nmax, rtol=rtol)
        else:
            pass
        data[mNaN] = np.nan
        return data
    def closure(force_ds, force, T=None, precise=False, nmax=50, rtol=1E-13):
        """
        Takes a force dataset, a force name, and a timestep. Precise determines
        whether to account for the horizontal Laplacian, and nmax and rtol
        control the iterative solver. Returns an xarray dataArray of the
        meridional overturning force function field at that time (or range of
        times).
        """
        if T is None:
            fy = force_ds[f"{force}y"]
            fz = force_ds.get(f"{force}zF", xr.zeros_like(force_ds["buoyzF"]))
        else:
            fy = force_ds[f"{force}y"].isel(T=T)
            fz = force_ds.get(f"{force}zF", xr.zeros_like(force_ds["buoyzF"])).isel(T=T)
            pass
        kw = {"precise":precise, "nmax":nmax, "rtol":rtol}
        out = (xr.apply_ufunc(solve, fy, fz, kwargs=kw,
                             input_core_dims=[["Z", "Yp1", "X"], ["Zp1", "Y", "X"]],
                             output_core_dims=[("Zp1", "Yp1", "X")],
                             vectorize=True, dask="parallelized")
               .rename(r"$\phi_{\mathrm{BT}}$" + f" ({fy.name[3:]})")
               .assign_attrs(description=("Barotropic Pressure due to " + f"{fy.name[3:]}"),
                             units=r"m$^{2}$s$^{-2}$"))
        return out
    return closure

def getAOTy_closure(ds):
    """
    Takes an MITgcm output xarray dataset. Returns a closure which calculates
    the zonal overturning force function.
    """
    def get_locals():
        """
        Calculates and returns the LU factorisation of the domain matrix, as
        well as the grid data required by the solver.
        """
        HW = getH(("Y", "Xp1"),ds,"min").data
        dyU = xr2np(ds.dyU)
        dxV = xr2np(ds.dxV)
        dyF = xr2np(ds.dyF)
        dxF = xr2np(ds.dxF)
        rAw = 1/xr2np(ds.rAw)
        hfc = xr2np(ds.HFacC)
        dzf = np.diff(np.minimum(np.pad(ds.drF.data, (1,0))[:,None,None].cumsum(axis=0), HW), axis=0)
        dzc = np.diff(np.minimum(np.pad(ds.drC.data, (1,0))[:,None,None].cumsum(axis=0), HW), axis=0)
        dzfkm1 = np.pad(dzf, ((1,0),(0,0),(0,0)))
        dzfk = np.pad(dzf, ((0,1),(0,0),(0,0)))
        dxC = xr2np(ds.dxC)
        zinds = np.indices(dzc.shape)[0]
        bottom_inds = fillnan(pushnan(np.where(dzc==0, zinds-1, np.nan)[::-1,...], axis=0)[-1,...],
                            n=dzc.shape[0]-1)
        m0 = ~np.logical_or(zinds<=0, zinds>=bottom_inds)
        mNaN = dzc==0
        c_up = np.where(m0, invert(dzfkm1*dzc), 0)
        c_dn = np.where(m0, invert(dzfk*dzc), 0)
        c_0 = np.where(m0, -(c_up + c_dn), 1)
        _, Nj, Ni = c_0.shape
        diags = (
            c_up.flatten()[Ni*Nj:], c_0.flatten(), c_dn.flatten()[:-Ni*Nj]
        )
        offsets = (-Ni*Nj, 0, Ni*Nj)
        A = sp.diags(diags, offsets=offsets, format="csc")
        A_dtype = A.dtype
        LU = spl.factorized(A)
        return dyU, dxV, dyF, dxF, rAw, dzf, dzc, dxC, m0, mNaN, LU, A_dtype
    dyU, dxV, dyF, dxF, rAw, dzf, dzc, dxC, m0, mNaN, LU, A_dtype = get_locals()
    def getRHS(fx, fz):
        """
        Calculates the RHS of the elliptic equation given the zonal and vertical
        forces.
        """
        fx = np.where(dzf!=0, fx, np.nan)
        fz = nanfill(fz, n=0)
        RHS = -(fillnan(delta(fz, axis=-1, expand=True))*invert(dxC)
                + fillnan(delta(fx, axis=-3, expand=True))*invert(dzc))
        return RHS
    def hLap(a):
        """
        Calculates the finite-volume horizontal Laplacian consistent with the
        (non-uniform) grid.
        """
        out = rAw*(delta(dxV*delta(a, axis=-2, expand=True)/dyU, axis=-2, expand=False)
                + delta(dyF*delta(a, axis=-1, expand=False)/dxF, axis=-1, expand=True))
        return fillnan(out)*m0
    def iterate(RHS, Aprev=None, nmax=50, rtol=1E-13, n=0):
        """
        Recursively improve the solution (to account for the small contribution
        of the horizontal Laplacian).
        """
        Aprev = LU(RHS.flatten()).reshape(RHS.shape) if Aprev is None else Aprev
        Anew = LU((RHS-hLap(Aprev)).flatten()).reshape(RHS.shape)
        r2 = np.sqrt((Anew-Aprev)[m0]**2).sum()*invert(np.sqrt(Anew[m0]**2).sum())
        if n >= nmax:
            print(f"Failed to converge past rtol={r2} after n={n} iterations")
            return Anew
        elif r2 > rtol:
            return iterate(RHS, Aprev=Anew, nmax=nmax, rtol=rtol, n=(n+1))
        else:
            print(f"Converged to rtol={r2} in n={n} iterations")
            pass
        return Anew
    def solve(fx, fz, precise=False, nmax=50, rtol=1E-13):
        """
        Given arrays of the forces in the x and z directions, returns a numpy
        array of the zonal overturning force function arising from that force.
        """
        RHS = getRHS(fx, fz).astype(A_dtype)
        data = LU(RHS.flatten()).reshape(RHS.shape)
        if precise is True:
            data = iterate(RHS, Aprev=data, nmax=nmax, rtol=rtol)
        else:
            pass
        data[mNaN] = np.nan
        return data
    def closure(force_ds, force, T=None, precise=False, nmax=50, rtol=1E-13):
        """
        Takes a force dataset, a force name, and a timestep. Precise determines
        whether to account for the horizontal Laplacian, and nmax and rtol
        control the iterative solver. Returns an xarray dataArray of the zonal
        overturning force function field at that time (or range of times).
        """
        if T is None:
            fx = force_ds[f"{force}x"]
            fz = force_ds.get(f"{force}zF", xr.zeros_like(force_ds["buoyzF"]))
        else:
            fx = force_ds[f"{force}x"].isel(T=T)
            fz = force_ds.get(f"{force}zF", xr.zeros_like(force_ds["buoyzF"])).isel(T=T)
            pass
        kw = {"precise":precise, "nmax":nmax, "rtol":rtol}
        out = (xr.apply_ufunc(solve, fx, fz, kwargs=kw,
                             input_core_dims=[["Z", "Y", "Xp1"], ["Zp1", "Y", "X"]],
                             output_core_dims=[("Zp1", "Y", "Xp1")],
                             vectorize=True, dask="parallelized")
               .rename(r"$\phi_{\mathrm{BT}}$" + f" ({fx.name[3:]})")
               .assign_attrs(description=("Barotropic Pressure due to " + f"{fx.name[3:]}"),
                             units=r"m$^{2}$s$^{-2}$"))
        return out
    return closure

def getHAbt_closure(ds, h_opt="avg"):
    """
    Takes an MITgcm output xarray dataset. Returns a closure which calculates
    the depth-integrated force function
    """
    HZ, HW, HS = [xr2np(getH(shp, ds, option=h_opt)) for shp in
                  (("Yp1", "Xp1"), ("Y", "Xp1"), ("Yp1", "X"))]
    hfW, hfS, hfC = [xr2np(getHFac(shp, ds, option=h_opt)) for shp in
                     (("Z", "Y", "Xp1"), ("Z", "Yp1", "X"), ("Z", "Y", "X"))]
    rAz, dyC, dxC, dyG, dxG = [xr2np(ds[i]) for i in ("rAz", "dyC", "dxC", "dyG", "dxG")]
    rAz_r = invert(rAz)
    dy_dxH = np.pad(dyC*invert(HS*dxG), ((0,0),(1,1)))
    dx_dyH = np.pad(dxC*invert(HW*dyG), ((1,1),(0,0)))
    z = xr2np(ds.Z)[:,None,None]
    drF = xr2np(ds.drF)[:,None,None]
    HZ_r = invert(HZ)
    m0 = getH(("Yp1", "Xp1"), ds, option="min").data == 0
    mIn = ~m0
    mOut = np.logical_or.reduce([np.logical_and.accumulate(m0, axis=0),
                                 np.logical_and.accumulate(m0[::-1,:], axis=0)[::-1,:],
                                 np.logical_and.accumulate(m0, axis=1),
                                 np.logical_and.accumulate(m0[:,::-1], axis=1)[::-1,:]])
    mIce = np.logical_xor(m0, mOut)
    mBC = m0*np.logical_or.reduce([npshift(mIn,i,axis=j,fill_value=False)
                                for i,j in ((1,0), (-1,0), (1,1), (-1,1))])
    mBCOut = mBC*mOut
    mBCIce = mBC*mIce
    mNaN = np.logical_xor(m0, mBC)
    def getLUDecomp():
        """
        Generate the sparse matrix representing the elliptic equation from the
        grid descriptors, and return its LU factorisation.
        """
        c0 = np.ones_like(mIn, dtype=np.float64)
        cE, cW, cN, cS = (np.zeros_like(c0) for i in range(4))
        np.copyto(cE, -rAz_r*dy_dxH[:, 1:], where=mIn)
        np.copyto(cW, -rAz_r*dy_dxH[:, :-1], where=mIn)
        np.copyto(cN, -rAz_r*dx_dyH[1:, :], where=mIn)
        np.copyto(cS, -rAz_r*dx_dyH[:-1, :], where=mIn)
        np.copyto(c0, -(cE + cW + cN + cS), where=mIn)
        Nj, Ni = c0.shape
        diags = (
            cS.flatten()[Ni:], cW.flatten()[1:], c0.flatten(),
            cE.flatten()[:-1], cN.flatten()[:-Ni]
        )
        offsets = (-Ni, -1, 0, 1, Ni)
        A = sp.diags(diags, offsets=offsets, format="csc")
        return spl.factorized(A), A.dtype
    LU, A_dtype = getLUDecomp()
    getphibt = getphiBT_closure(ds, h_opt=h_opt, npclosure=True)
    def getRHS(fx, fy, fz, comp="tot"):
        """
        Calculates the RHS of the elliptic equation given the arrays of the
        forces in the x, y, and z directions. comp controls which component of
        the force function to calculate.
        """
        if comp.startswith(("c", "C")):
            return np.zeros_like(HZ_r)
        elif comp.startswith(("d",)):
            return get_phi_bot(fx, fy, fz)
        else:
            pass
        zfz = z*fz
        di_zfz = fillnan(delta(zfz, mask=hfC, axis=-1, expand=True)*hfW*drF).sum(axis=-3)
        dj_zfz = fillnan(delta(zfz, mask=hfC, axis=-2, expand=True)*hfS*drF).sum(axis=-3)
        fxdx = fillnan(fx*dxC*hfW*drF).sum(axis=-3)
        fydy = fillnan(fy*dyC*hfS*drF).sum(axis=-3)
        term1 = rAz_r*(delta(fydy, axis=-1, mask=HS, expand=True)
                       - delta(fxdx, axis=-2, mask=HW, expand=True))
        term2_x = invert(HW)*(fxdx + di_zfz)
        term2_y = invert(HS)*(fydy + dj_zfz)
        term2 = HZ*rAz_r*(delta(term2_y, axis=-1, mask=HS, expand=True)
                          - delta(term2_x, axis=-2, mask=HW, expand=True))
        term3 = -rAz_r*(delta(fydy + dj_zfz, axis=-1, mask=HS, expand=True)
                        - delta(fxdx + di_zfz, axis=-2, mask=HW, expand=True))
        sw1 = 0 if comp.startswith(("b", "B")) else 1
        sw2 = 0 if comp.startswith(("a", "A")) else 1
        return fillnan(HZ_r*(sw1*term1 + sw2*(term2 + term3)))
    def get_phi_bot(fx, fy, fz):
        """
        Calculate the bottom pressure torque from the arrays of the forces in
        the x, y, and z directions.
        """
        phi = getphibt(fx, fy, fz)
        t1 = delta((HS*dxG/dyC)*delta(phi, axis=-2, expand=True), axis=-1, expand=True)
        t2 = -delta((HW*dyG/dxC)*delta(phi, axis=-1, expand=True), axis=-2, expand=True)
        return -rAz_r*fillnan(t1+t2)*invert(HZ)
    def getBC(fx, fy, fz):
        """
        Calculate the Dirichlet boundary condition on HABT on the Iceland
        boundary.
        """
        Fx = (xr2np(fx*ds.HFacW)*drF).sum(axis=-3)
        Fy = (xr2np(fy*ds.HFacS)*drF).sum(axis=-3)
        phi = getphibt(fx, fy, fz)
        HW = xr2np(getH(("Y", "Xp1"), ds, option=h_opt))
        HS = xr2np(getH(("Yp1", "X"), ds, option=h_opt))
        HWdpdx = fillnan(HW*delta(phi, axis=-1, expand=True)/dxC)
        HSdpdy = fillnan(HS*delta(phi, axis=-2, expand=True)/dyC)
        dyHAbt = np.pad((Fx-HWdpdx)*dyG, ((1,0),(0,0)))
        dxHAbt = -np.pad((Fy-HSdpdy)*dxG, ((0,0),(1,0)))
        dyHAbt[mNaN] = 0
        HA_fromdy = dyHAbt.cumsum(axis=-2)
        dxHAbt[mNaN] = 0
        HA_fromdx = dxHAbt.cumsum(axis=-1)
        HAbcdy = HA_fromdy[mBCIce]
        HAbcdx = HA_fromdx[mBCIce]
        HAbc = 0.5*(HAbcdy.mean() + HAbcdx.mean())
        return HAbc
    def solve(fx, fy, fz, comp="tot", bcIce=None, returnA=False, returnRHS=False):
        """
        Given arrays of the forces in the x, y, and z directions, return a numpy
        array of the depth-integrated force function. comp controls which
        components of the force function to solve for, bcIce allows for
        specification of the boundary condition on the Iceland boundary, returnA
        controls whether to return the barotropic force function instead of
        depth-integrated, and returnRHS allows for the function to simply return
        the RHS.
        """
        bc = getBC(fx, fy, fz) if bcIce is None else bcIce
        RHS = getRHS(fx, fy, fz, comp=comp).astype(A_dtype)
        RHS[mBCOut] = 0.
        RHS[mBCIce] = bc
        if returnRHS is True:
            return RHS
        HAbt = LU(RHS.flatten()).reshape(RHS.shape)
        HAbt[mNaN] = np.nan
        if returnA is True:
            data = HAbt*invert(HZ)
        else:
            data = HAbt
            pass
        return data
    def closure(force_ds, force, T=None, comp="tot", bcIce=None, returnA=False, returnRHS=False):
        """
        Takes a force dataset, a force name, and a timestep, and returns an
        xarray dataset of the depth-integrated force function. comp controls which
        components of the force function to solve for, bcIce allows for
        specification of the boundary condition on the Iceland boundary, returnA
        controls whether to return the barotropic force function instead of
        depth-integrated, and returnRHS allows for the function to simply return
        the RHS.
        """
        if T is None:
            fx, fy = force_ds[f"{force}x"], force_ds[f"{force}y"]
            fz = force_ds.get(f"{force}zC", xr.zeros_like(force_ds["buoyzC"]))
            bcIce = (getBC(force_ds["phix"], force_ds["phiy"], xr.zeros_like(force_ds["buoyzC"]))
                     if force=="buoy" else bcIce)
        else:
            fx, fy = force_ds[f"{force}x"].isel(T=T), force_ds[f"{force}y"].isel(T=T)
            fz = force_ds.get(f"{force}zC", xr.zeros_like(force_ds["buoyzC"])).isel(T=T)
            bcIce = (getBC(force_ds["phix"].isel(T=T), force_ds["phiy"].isel(T=T),
                           xr.zeros_like(force_ds["buoyzC"].isel(T=T)))
                     if force=="buoy" else bcIce)
            pass
        bcIce = 0 if comp.startswith(("d", "D", "b", "B", "a", "A")) else bcIce
        kw = {"comp":comp, "bcIce":bcIce, "returnA":returnA, "returnRHS":returnRHS}
        out = (xr.apply_ufunc(solve, fx, fy, fz, kwargs=kw,
                              input_core_dims=[["Z", "Y", "Xp1"],
                                               ["Z", "Yp1", "X"],
                                               ["Z", "Y", "X"]],
                              output_core_dims=[("Yp1", "Xp1")],
                              vectorize=True, dask="parallelized")
               .rename(r""))
        return out
    return closure

def getHAbt_residual(ds, h_opt="avg"):
    """
    Takes an MITgcm output xarray dataset. Returns a closure which calculates
    the depth-integrated force function, in this case by calculating the
    divergent component and inferring the rotational component as a residual.
    """
    def get_locals():
        """
        Calculates and returns any local variables required by the solver.
        """
        HW = xr2np(getH(("Y", "Xp1"), ds, option=h_opt))
        HS = xr2np(getH(("Yp1", "X"), ds, option=h_opt))
        hfW, hfS = [xr2np(getHFac(shp, ds, option=h_opt)) for shp in
                    (("Z", "Y", "Xp1"), ("Z", "Yp1", "X"))]
        dyC, dxC, dyG, dxG = [xr2np(ds[i]) for i in ("dyC", "dxC", "dyG", "dxG")]
        # z = -xr2np(getHFac(("Zp1", "Y", "X"), ds, option=h_opt)*ds.drC).cumsum(axis=0)[:-1,:,:]
        drF = xr2np(ds.drF)[:,None,None]
        m0 = getH(("Yp1", "Xp1"), ds, option="min").data == 0
        mIn = ~m0
        mOut = np.logical_or.reduce([np.logical_and.accumulate(m0, axis=0),
                                    np.logical_and.accumulate(m0[::-1,:], axis=0)[::-1,:],
                                    np.logical_and.accumulate(m0, axis=1),
                                    np.logical_and.accumulate(m0[:,::-1], axis=1)[::-1,:]])
        mIce = np.logical_xor(m0, mOut)
        mBC = m0*np.logical_or.reduce([npshift(mIn,i,axis=j,fill_value=False)
                                    for i,j in ((1,0), (-1,0), (1,1), (-1,1))])
        # mBCOut = mBC*mOut
        # mBCIce = mBC*mIce
        mNaN = np.logical_xor(m0, mBC)
        getphibt = getphiBT_closure(ds, h_opt=h_opt, npclosure=True)
        return HW, HS, hfW, hfS, dyC, dxC, dyG, dxG, drF, mNaN, getphibt
    HW, HS, hfW, hfS, dyC, dxC, dyG, dxG, drF, mNaN, getphibt = get_locals()
    def get_phiBC_bot(T=None):
        """
        Calculate the baroclinic component of bottom pressure
        """
        _ds = ds if T is None else ds.isel(T=T)
        phibcbot = ((_ds.PHIHYD*_ds.HFacC*_ds.drF).sum("Z")/_ds.Depth).where(_ds.Depth) # - _ds.PHIBOT
        return phibcbot.values
    def solve(fx, fy, fz, isbuoy=False, T=None):
        """
        Given arrays of the forces in the x, y, and z directions, return a numpy
        array of the depth-integrated force function. isbuoy specifies that the
        force is the buoyancy force, which must be treated differently.
        """
        Fx = ((fx*hfW)*drF).sum(axis=-3)
        Fy = ((fy*hfS)*drF).sum(axis=-3)
        phi = getphibt(fx, fy, fz)
        if isbuoy is True:
            phiBCbot = get_phiBC_bot(T=T)
            phi -= phiBCbot
        else:
            pass
        HWdpdx = fillnan(HW*delta(phi, axis=-1, expand=True)/dxC)
        HSdpdy = fillnan(HS*delta(phi, axis=-2, expand=True)/dyC)
        dyHAbt = np.pad((Fx-HWdpdx)*dyG, ((1,0),(0,0)))
        dxHAbt = -np.pad((Fy-HSdpdy)*dxG, ((0,0),(1,0)))
        dyHAbt1 = -np.pad((Fx-HWdpdx)*dyG, ((0,1),(0,0)))
        dxHAbt1 = np.pad((Fy-HSdpdy)*dxG, ((0,0),(0,1)))
        HA_fromdy = dyHAbt.cumsum(axis=-2)
        HA_fromdy1 = dyHAbt1[::-1, :].cumsum(axis=-2)[::-1, :]
        HA_fromdx = dxHAbt.cumsum(axis=-1)
        HA_fromdx1 = dxHAbt1[:, ::-1].cumsum(axis=-1)[:, ::-1]
        HA_fromdx[mNaN] = np.nan
        HA_fromdy[mNaN] = np.nan
        HA_fromdx1[mNaN] = np.nan
        HA_fromdy1[mNaN] = np.nan
        # data = (HA_fromdx + HA_fromdy + HA_fromdx1 + HA_fromdy1)/4
        data = HA_fromdy
        return data
    def closure(force_ds, force, T=None):
        """
        Takes a force dataset, a force name, and a timestep, and returns an
        xarray dataset of the depth-integrated force function.
        """
        if T is None:
            fx, fy = force_ds[f"{force}x"], force_ds[f"{force}y"]
            fz = force_ds.get(f"{force}zC", xr.zeros_like(force_ds["buoyzC"]))
        else:
            fx, fy = force_ds[f"{force}x"].isel(T=T), force_ds[f"{force}y"].isel(T=T)
            fz = force_ds.get(f"{force}zC", xr.zeros_like(force_ds["buoyzC"])).isel(T=T)
            pass
        out = (xr.apply_ufunc(solve, fx, fy, fz, kwargs={"isbuoy":force=="buoy", "T":T},
                              input_core_dims=[["Z", "Y", "Xp1"], ["Z", "Yp1", "X"],
                                               ["Z", "Y", "X"]],
                              output_core_dims=[("Yp1", "Xp1")],
                              vectorize=True, dask="parallelized")
               .rename(r"$\phi_{\mathrm{BT}}$" + f" ({fx.name[3:]})")
               .assign_attrs(description=("Barotropic Pressure due to " + f"{fx.name[3:]}"),
                             units=r"m$^{2}$s$^{-2}$"))
        return out
    return closure

def getPsiBT_closure(ds):
    """
    Takes an MITgcm output xarray dataset. Returns a closure which calculates
    the depth-integrated streamfunction.
    """
    def get_locals():
        """
        Calculates and returns any local variables required by the solver.
        """
        hfW, hfS = (ds.HFacW, ds.HFacS)
        dyG, dxG = [xr2np(ds[i]) for i in ("dyG", "dxG")]
        drF = xr2np(ds.drF)[:,None,None]
        m0 = getH(("Yp1", "Xp1"), ds, option="min").data == 0
        mIn = ~m0
        mOut = np.logical_or.reduce([np.logical_and.accumulate(m0, axis=0),
                                    np.logical_and.accumulate(m0[::-1,:], axis=0)[::-1,:],
                                    np.logical_and.accumulate(m0, axis=1),
                                    np.logical_and.accumulate(m0[:,::-1], axis=1)[::-1,:]])
        mIce = np.logical_xor(m0, mOut)
        mBC = m0*np.logical_or.reduce([npshift(mIn,i,axis=j,fill_value=False)
                                    for i,j in ((1,0), (-1,0), (1,1), (-1,1))])
        mBCOut = mBC*mOut
        mBCIce = mBC*mIce
        mNaN = np.logical_xor(m0, mBC)
        return hfW, hfS, dyG, dxG, drF, mBCOut, mBCIce, mNaN
    hfW, hfS, dyG, dxG, drF, mBCOut, mBCIce, mNaN = get_locals()
    def solve(u, v):
        """
        Given zonal and meridional velocities u and v, calculates the barotropic
        streamfunction by integrating from each of the boundaries, and then
        averaging.
        """
        U = (u*hfW*drF).sum(axis=-3)
        V = (v*hfS*drF).sum(axis=-3)
        dyPsi = -np.pad(U*dyG, ((1,0),(0,0)))
        dxPsi = np.pad(V*dxG, ((0,0),(1,0)))
        dyPsi1 = np.pad(U*dyG, ((0,1),(0,0)))
        dxPsi1 = -np.pad(V*dxG, ((0,0),(0,1)))
        Psi_fromdy = dyPsi.cumsum(axis=-2)
        Psi_fromdy1 = dyPsi1[::-1, :].cumsum(axis=-2)[::-1, :]
        Psi_fromdx = dxPsi.cumsum(axis=-1)
        Psi_fromdx1 = dxPsi1[:, ::-1].cumsum(axis=-1)[:, ::-1]
        data = 1E-6 * (Psi_fromdy + Psi_fromdy1 + Psi_fromdx + Psi_fromdx1)/4
        data[mNaN] = np.nan
        data[mBCOut] = 0
        data[mBCIce] = data[mBCIce].mean()
        return data
    def closure(ds, T=None):
        """
        Takes a MITgcm output xarray dataset and an optional timestep, and
        returns an xarray DataArray of the depth-integrated streamfunction.
        """
        if T is None:
            u, v = ds.UVEL, ds.VVEL
        else:
            u, v = ds.UVEL.isel(T=T), ds.VVEL.isel(T=T)
            pass
        out = (xr.apply_ufunc(solve, u, v,
                              input_core_dims=[["Z", "Y", "Xp1"], ["Z", "Yp1", "X"]],
                              output_core_dims=[("Yp1", "Xp1")],
                              vectorize=True, dask="parallelized")
               .rename(r"$\psi_{\mathrm{BT}}$")
               .assign_attrs(description=("Barotropic Streamfunction"),
                             units=r"Sv"))
        return out
    return closure

def decompose_v(ds, T=None, delta_ek=50, h_opt="max", verbose=False):
    """
    Given an MITgcm output xarray DataSet, decomposes the meridional velocity
    into geostrophic, ekman, and bottom components and returns a DataSet
    containing these.
    """
    rho0 = ds.f_data.rhonil
    g = ds.f_data.gravity
    hfc = xr2np(ds.HFacC)
    hfs = xr2np(ds.HFacS)
    f = xr2np(ds.fCori.isel(X=0))[None, :, None]
    dxC = xr2np(ds.dxC)
    drF = xr2np(ds.drF)[:,None,None]
    hfwmax = xr2np(getHFac(("Z", "Y", "Xp1"), ds, option=h_opt))
    hfsmax = xr2np(getHFac(("Z", "Y", "Xp1"), ds, option=h_opt))
    def v2u(vvel, maskedoutput=True):
        """
        Averages meridional velocity onto the u-point
        """
        v_weighted = vvel*hfs
        v_c = bar(fillnan(v_weighted, n=0), axis=-2, expand=False)*invert(hfc)
        v_u = bar(v_c, axis=-1, mask=(hfc!=0), expand=True)
        return v_u if maskedoutput is True else fillnan(v_u)
    def get_vshear(rho, extrapolatebc=True):
        """
        Calculates geostrophic shear velocity from density field
        """
        _ = [print("Getting vshear") if verbose else None]
        dzF = hfwmax*drF
        drho_dx = fillnan(delta(rho, axis=-1, mask=hfc, expand=True,
                                extrapolatebc=extrapolatebc)
                          *invert(dxC))
        dv = np.pad(fillnan(-g*invert(f*rho0)*drho_dx*dzF),
                    ((0,0),)*(drho_dx.ndim - 3) + ((0,1),(0,0),(0,0)))
        v_sh = bar(dv[...,::-1,:,:].cumsum(axis=-3)[...,::-1,:,:], axis=-3)
        return v_sh
    def get_vek(tau_x):
        """
        Calculates meridional Ekman velocity from zonal wind stress.
        """
        _ = [print("Getting vek") if verbose else None]
        dzF = hfsmax*drF
        zF = np.pad(dzF.cumsum(axis=-3), ((1,0),(0,0),(0,0)))
        HS = zF[-1,...]
        indicator = delta(np.minimum(delta_ek, zF), axis=-3)*invert(dzF)
        v_ek = -indicator*tau_x*invert(np.minimum(delta_ek, HS)*f*rho0)
        return v_ek
    def get_vbot(v_upoint, v_shear, v_ek):
        """
        Calculates meridional bottom velocity from other components using
        non-divergence of the flow.
        """
        _ = [print("Getting vbot") if verbose else None]
        dzF = hfsmax*drF
        HS = dzF.sum(axis=-3)
        v_bot = (fillnan(v_upoint - v_shear - v_ek)*dzF).sum(axis=-3)*invert(HS)
        v_bot = v_bot[:,None,:,:] if v_bot.ndim == 3 else v_bot
        return np.broadcast_to(v_bot, v_upoint.shape)
    if T is None:
        rho, tau_x, vvel = [xr2np(ds[i]) for i in ("RHOAnoma", "oceTAUX", "VVEL")]
    else:
        rho, tau_x, vvel = [xr2np(ds[i].isel(T=T)) for i in ("RHOAnoma", "oceTAUX", "VVEL")]
        pass
    tau_x = tau_x[:,None,:,:] if tau_x.ndim == 3 else tau_x
    xrdims = ("T", "Z", "Y", "Xp1") if vvel.ndim == 4 else ("Z", "Y", "Xp1")
    xrcoords = {i: ds.coords[i] for i in xrdims}
    _ = [print("Getting vtot") if verbose else None]
    v = xr.Dataset({"tot": (xrdims, v2u(vvel, maskedoutput=False)),
                    "ek": (xrdims, get_vek(tau_x)),
                    "sh": (xrdims, get_vshear(rho, extrapolatebc=True))},
                   coords=xrcoords)
    v["bot"] = xr.DataArray(get_vbot(v.tot, v.sh, v.ek), dims=xrdims, coords=xrcoords)
    v["tw"] = xr.DataArray(v.tot - (v.ek + v.bot), dims=xrdims, coords=xrcoords)
    for comp in ("tot", "bot", "tw", "sh", "ek"):
        v[comp] = v[comp].where(hfwmax)
        continue
    return v

def get_mocz_closure(ds):
    """
    Takes an MITgcm output xarray dataset. Returns a closure which calculates
    the zonally-integrated overturning streamfunction in depth space.
    """
    hfw = getHFac(("Z", "Y", "Xp1"), ds, option="avg")
    dxdz = xr2np(hfw*ds.drF*ds.dxC)
    mnan = (getHFac(("Zp1", "Y", "Xp1"), ds, option="max")
            .sum("Xp1").pipe(lambda x: xr.where(x==0, True, False)))
    def correct_v(v):
        """
        Applies a correction to the velocity field such that there is no local
        depth-integrated meridional transport.
        """
        vbar = (fillnan(v)*dxdz).sum(axis=(-3, -1)) * invert(dxdz.sum(axis=(-3, -1)))
        return v-vbar[None,:,None]
    def solve(v, m, correct=False):
        """
        Given ndArrays of meridional velocity v, and a mask m to indicate
        topography, calculates the meridional overturning streamfunction in
        depth space by integrating from the floor up. Optional boolean flag
        correct determines whether to apply to above correction to the velocity
        field.
        """
        if correct is True:
            v = correct_v(v)
        else:
            pass
        data = np.pad((fillnan(v)*dxdz).sum(axis=-1)[::-1,:].cumsum(axis=-2)[::-1,:],
                      ((0,1),(0,0)))
        data[m] = np.nan
        return -1E-6*data
    def closure(v, T=None, corrected=False):
        """
        Given xarray DataArray of meridional velocity v, calculates the
        meridional overturning streamfunction in depth space by integrating from
        the floor up. Optional boolean flag corrected determines whether to
        apply to local correction to the velocity field.
        """
        if T is None:
            v = v
        else:
            v = v.isel(T=T)
            pass
        out = (xr.apply_ufunc(solve, v, mnan, kwargs={"correct": corrected},
                              input_core_dims=[["Z", "Y", "Xp1"], ["Zp1", "Y"]],
                              output_core_dims=[("Zp1", "Y")],
                              vectorize=True, dask="parallelized")
               .rename(r"$\mathrm{MOC}_{z}$")
               .assign_attrs(description=("Overturning Streamfunction in Depth Space"),
                             units=r"Sv"))
        return out
    return closure


@nb.guvectorize(
    [(nb.float64[:], nb.float64[:], nb.float64[:],
      nb.float64[:], nb.float64[:], nb.float64[:]),
     (nb.float32[:], nb.float32[:], nb.float32[:],
      nb.float32[:], nb.float32[:], nb.float32[:]),],
    "(n),(n),(n),(m),(m)->(m)",
    nopython=True,
)
def _interp_1d_conservative_kernel(phi, theta_1, theta_2, theta_hat_1, theta_hat_2, output):
    """
    Compiled numba gufunc to transform array phi from current coordinates to
    those given by theta_hat_1 and theta_hat_2 by binning.
    """
    output[:] = 0
    n = len(theta_1)
    m = len(theta_hat_1)
    for i in range(n):
        # handle missing values
        if np.isnan(theta_1[i]) and np.isnan(theta_2[i]):
            continue
        # in the next two cases, we are effectively applying a boundary condition
        # by assuming that theta is homogenous over the cell
        elif np.isnan(theta_1[i]):
            theta_min = theta_max = theta_2[i]
        elif np.isnan(theta_2[i]):
            theta_min = theta_max = theta_1[i]
        # handle non-monotonic stratification
        elif theta_1[i] < theta_2[i]:
            theta_min = theta_1[i]
            theta_max = theta_2[i]
        else:
            theta_min = theta_2[i]
            theta_max = theta_1[i]
            continue
        for j in range(m):
            if (theta_hat_1[j] > theta_max) or (theta_hat_2[j] < theta_min):
                # there is no overlap between the cell and the bin
                continue
            elif theta_max == theta_min:
                output[j] += phi[i]
            else:
                # from here on there is some overlap
                theta_hat_min = max(theta_min, theta_hat_1[j])
                theta_hat_max = min(theta_max, theta_hat_2[j])
                alpha = (theta_hat_max - theta_hat_min) / (theta_max - theta_min)
                # now assign based on this weight
                output[j] += alpha * phi[i]
                continue
            continue
        continue
    pass

def get_z2rho_transform(ds, dims, thetapipe=(lambda x: x), **theta_inds):
    """
    Given an xarray DataSet and set of dimensions, with optional transformations
    on the target theta, return a closure that will transform a given DataArray
    into theta coordinates.
    """
    def get_locals(dims):
        """
        Calculates and returns the LU factorisation of the domain matrix, as
        well as the grid data required by the solver.
        """
        dims = sorted(dims, key=lambda x: {"T":0, "Z":1, "Y":2, "X":3}[x[0]])
        theta = ds.THETA.where(ds.HFacC).pipe(thetapipe).isel(**theta_inds).pipe(sortdims)
        x_ax, y_ax, z_ax = [theta.get_axis_num(i) if i in map(lambda x: x[0], dims) else None
                        for i in ("X", "Y", "Z")]
        theta = xr2np(theta)
        theta = bar(theta, axis=x_ax, expand=True) if "Xp1" in dims else theta
        theta = bar(theta, axis=y_ax, expand=True) if "Yp1" in dims else theta
        theta = bar(theta, axis=z_ax, expand=True)
        theta = bar(theta, axis=z_ax, expand=True) if "Zp1" in dims else theta
        nanwhere = np.any(np.isfinite(theta), axis=z_ax, keepdims=True)
        theta_min = fillnan(np.fmin.reduce(theta, axis=z_ax, keepdims=True, where=nanwhere,
                                           initial=np.nan))
        theta_max = fillnan(np.fmax.reduce(theta, axis=z_ax, keepdims=True, where=nanwhere,
                                           initial=np.nan))
        coredim = "Zp1" if "Z" in dims else "Zp1p1"
        theta = xr.DataArray(theta, dims=[coredim if i.startswith("Z") else i for i in dims])
        return dims, theta, theta_min, theta_max, coredim
    dims, theta, theta_min, theta_max, coredim = get_locals(dims)
    def z2rho_transform(da, theta_bins, **inds):
        """
        Closure taking a DataArray and returning a new one transformed into the
        theta coordinates specified by theta_bins.
        """
        if type(theta_bins) == int:
            bins = np.linspace(np.floor(theta.min()), np.ceil(theta.max()), theta_bins)
        elif callable(theta_bins) is True:
            bins = theta_bins(theta_min, theta_max)
        else:
            bins = theta_bins
            pass
        bins = np.array(bins)
        assert all([(i==j[:-2]) if i.startswith("Z") else i==j
                    for i,j in zip(da.dims, theta.dims)])
        assert bins.ndim == 1
        if all(np.diff(bins) > 0):
            pass
        else:
            raise ValueError("Target values are not monotonic")
        arr = da.isel(**inds).fillna(0.)
        theta_arr = theta.isel(**inds)
        t_sl = tuple([inds.get(d, slice(None)) for d in theta.dims])
        t_min = theta_min[t_sl]
        t_max = theta_max[t_sl]
        theta_1 = theta_arr.isel({coredim:np.s_[:-1]})
        theta_2 = theta_arr.isel({coredim:np.s_[1:]})
        theta_hat_1 = bins[:-1]
        theta_hat_2 = bins[1:]
        out = xr.apply_ufunc(_interp_1d_conservative_kernel, arr,
                             theta_1, theta_2, theta_hat_1, theta_hat_2,
                             input_core_dims=[(coredim[:-2],), (coredim,), (coredim,),
                                              ("thetaC",), ("thetaC",)],
                             output_core_dims=[("thetaC",)],
                             output_dtypes=[arr.dtype],
                             join="left",
                             dask="parallelized")
        sl = tuple([slice(None) if i == 1 else None for i in theta_min.shape])
        mask = xr.DataArray(np.where(np.logical_and(theta_hat_2[sl] > t_min,
                                                    theta_hat_1[sl] < t_max),
                                     True, False),
                            dims=["thetaC" if i.startswith("Z") else i for i in dims])
        return (out.assign_coords(thetaC=bar(bins))
                .assign_attrs(thetaF=bins)
                .where(mask).squeeze().pipe(sortdims))
    return z2rho_transform

def get_moctheta_closure(ds):
    """
    Takes an MITgcm output xarray dataset. Returns a closure which calculates
    the zonally-integrated overturning streamfunction in density space.
    """
    def correct_v(v):
        """
        Applies a correction to the velocity field such that there is no local
        depth-integrated meridional transport.
        """
        vbar = (v.fillna(0).sum(["thetaC", "Xp1"])
                / xr.ones_like(v).where(np.isfinite(v)).sum(["thetaC", "Xp1"]))
        return v-vbar
    def solve(v, m):
        """
        Given ndArrays of meridional velocity v, and a mask m to indicate
        topography, calculates the meridional overturning streamfunction in
        depth space by integrating from the floor up.
        """
        data = np.pad(fillnan(v).sum(axis=-1).cumsum(axis=-2), ((1,0),(0,0)))
        data[m] = np.nan
        return -1E-6*data
    def closure(v_trans_theta, T=None, t_bins=None, corrected=False):
        """
        Given xarray DataArray of transformed meridional velocity v, calculates
        the meridional overturning streamfunction in density space by
        integrating from the floor up. T determines the timestep, and t_bins
        allows for specification of temperature levels in output. Optional
        boolean flag corrected determines whether to apply to local correction
        to the velocity field.
        """
        v_trans = v_trans_theta if T is None else v_trans_theta.isel(T=T)
        if corrected is True:
            v_trans = correct_v(v_trans)
        else:
            pass
        thetaF = v_trans_theta.thetaF
        msk = np.all(np.isnan(v_trans), axis=-1).data
        msk = bar(msk, axis=-2, expand=True) == 1
        out = (xr.apply_ufunc(solve, v_trans, msk,
                              input_core_dims=[("thetaC", "Y", "Xp1"), ("thetaF", "Y")],
                              output_core_dims=[("thetaF", "Y")],
                              vectorize=True, dask="parallelized")
               .assign_coords(thetaF=thetaF).pipe(sortdims)
               .rename(r"$\mathrm{MOC}_{\theta}$")
               .assign_attrs(description=("Overturning Streamfunction in Density Space"),
                             units=r"Sv"))
        return out
    return closure

class ArcherExperiment(SimpleNamespace):
    """
    Class for loading and analysing the large datasets created by MITgcm output.
    Intended for interactive use.
    """
    def __init__(self, ncfile, chunks=None, **kwargs):
        """
        Creates a temporary directory to store convenience files, and opens netCDF
        file specified by ncfile. If chunks is specified, uses the dask capability of
        xarray to load the dataset with the specified chunk size.
        """
        super().__init__(**kwargs)
        self.data_file = Path(ncfile).resolve()
        self.data_dir = TemporaryDirectory(prefix=self.data_file.stem,
                                           suffix="_temp",
                                           dir=Path("~/temporary_files/").expanduser().resolve())
        self.ds = xr.open_dataset(ncfile, chunks=chunks, cache=False)
        self.verbose = kwargs.get("verbose", False)
        return
    def load_forces(self, chunks=None, **kwargs):
        """
        Calculates forces from dataset, and stores output in a temporary file to
        avoid holding too much data in memory. This output file is then stored as
        an open DataSet in self.f.
        """
        verbose = kwargs.get("verbose", self.verbose)
        def write_force_nc():
            fname = f"{self.data_dir.name}/forces.nc"
            _ = [print("Calculating forces") if verbose else None]
            fds = getF_rho(self.ds, **kwargs)
            _ = [print("Writing forces to temporary file") if verbose else None]
            fds.astype("f4").to_netcdf(fname, compute=True)
            fds.close()
            return fname
        ffile = write_force_nc()
        _ = [print("Re-opening force dataset") if verbose else None]
        self.f = xr.open_dataset(ffile, chunks=chunks, cache=False)
        _ = [print("Done") if verbose else None]
        return
    def load_closures(self, phi_opt="max", ha_opt="avg", hares_opt="avg", **kwargs):
        """
        Load the relevant closures for calculating streamfunctions and force
        functions, storing only data required for closures, and releasing the
        rest.
        """
        verbose = kwargs.get("verbose", self.verbose)
        _ = [print("Generating pressure solver") if verbose else None]
        phi_closure = getphiBT_closure(self.ds, h_opt=phi_opt)
        _ = [print("Generating A^{OTx} solver") if verbose else None]
        AOTx_closure = getAOTx_closure(self.ds)
        _ = [print("Generating A^{OTy} solver") if verbose else None]
        AOTy_closure = getAOTy_closure(self.ds)
        _ = [print("Generating HA^{BT} direct solver") if verbose else None]
        HAbt_closure = getHAbt_closure(self.ds, h_opt=ha_opt)
        _ = [print("Generating HA^{BT} residual solver") if verbose else None]
        HAbt_res_closure = getHAbt_residual(self.ds, h_opt=hares_opt)
        _ = [print("Generating barotropic streamfunction solver") if verbose else None]
        psiBT_closure = getPsiBT_closure(self.ds)
        def get_phiBT(force, T=None):
            return phi_closure(self.f, force, T=T)
        def get_AOTx(force, T=None, precise=False, nmax=50, rtol=1E-13):
            return AOTx_closure(self.f, force, T=T,
                                precise=precise, nmax=nmax, rtol=rtol)
        def get_AOTy(force, T=None, precise=False, nmax=50, rtol=1E-13):
            return AOTy_closure(self.f, force, T=T,
                                precise=precise, nmax=nmax, rtol=rtol)
        def get_HAbt(force, comp="tot", T=None, bcIce=None, returnA=False, returnRHS=False):
            return HAbt_closure(self.f, force, comp=comp, T=T,
                                bcIce=bcIce, returnA=returnA, returnRHS=False)
        def get_HAbt_res(force, T=None):
            return HAbt_res_closure(self.f, force, T=T)
        def get_psiBT(T=None):
            return psiBT_closure(self.ds, T=T)
        self.get_phiBT = get_phiBT
        self.get_AOTx = get_AOTx
        self.get_AOTy = get_AOTy
        self.get_HAbt = get_HAbt
        self.get_HAbt_res = get_HAbt_res
        self.get_psiBT = get_psiBT
        _ = [print("Done") if verbose else None]
        return
    def get_transform(self, da_meta, **theta_inds):
        """
        Creates a closure to transform from z-coordinates to theta-coordinates.
        """
        transform_closure = get_z2rho_transform(self.ds, da_meta.dims, **theta_inds)
        def transform(da, t_bins, **inds):
            return transform_closure(da, t_bins, **inds)
        return transform
    def load_psiBT_closure(self):
        """
        Loads barotropic streamfunction closure
        """
        psiBT_closure = getPsiBT_closure(self.ds)
        def get_psiBT(T=None):
            return psiBT_closure(self.ds, T=T)
        self.get_psiBT = get_psiBT
    def load_vdecompz(self, T=None, delta_ek=50, h_opt="max", chunks=None, **kwargs):
        """
        Calculates v decomposition in depth space, stores in a temporary file,
        and opens this file as a DataSet.
        """
        verbose = kwargs.get("verbose", self.verbose)
        def write_vz_nc():
            fname = f"{self.data_dir.name}/vdecomp_z.nc"
            fpath = Path(fname).resolve()
            _ = [print("Decomposing velocities in z space") if verbose else None]
            if fpath.exists():
                self.v.close()
                fpath.unlink()
            else:
                pass
            vds = decompose_v(self.ds, T=T, delta_ek=delta_ek, h_opt=h_opt)
            _ = [print("Writing velocities to temporary file") if verbose else None]
            vds.astype("f4").to_netcdf(fname, compute=True)
            vds.close()
            return fname
        vzfile = write_vz_nc()
        _ = [print("Re-opening velocity dataset") if verbose else None]
        self.v = xr.open_dataset(vzfile, chunks=chunks, cache=False)
        _ = [print("Done") if verbose else None]
        return
    def load_vdecompt(self, t_bins, T=None, chunks=None, **kwargs):
        """
        Calculates v decomposition in density space, stores in a temporary file,
        and opens this file as a DataSet.
        """
        verbose = kwargs.get("verbose", self.verbose)
        _ = [print("Loading transformation function") if verbose else None]
        if T is None:
            trans = self.get_transform(self.v.tot)
        else:
            trans = self.get_transform(self.v.tot, T=T)
            pass
        def write_vt_nc():
            fname = f"{self.data_dir.name}/vdecomp_t.nc"
            fpath = Path(fname).resolve()
            _ = [print("Transforming velocities to t space") if verbose else None]
            dxdz = ((self.ds.dxC * self.ds.drF * getHFac(("Z", "Y", "Xp1"), self.ds, option="avg"))
                    .pipe(sortdims))
            if fpath.exists():
                self.vt.close()
                fpath.unlink()
            else:
                pass
            vds = xr.Dataset({comp: [print(f"> transforming {comp}"),
                                     trans(self.v[comp].fillna(0)*dxdz, t_bins=t_bins)][-1]
                              for comp in self.v.data_vars})
            _ = [print("Writing velocities to temporary file") if verbose else None]
            vds.astype("f4").to_netcdf(fname, compute=True)
            vds.close()
            return fname
        vtfile = write_vt_nc()
        _ = [print("Re-opening velocity dataset") if verbose else None]
        self.vt = xr.open_dataset(vtfile, chunks=chunks, cache=False)
        _ = [print("Done") if verbose else None]
        return
    def load_moc_closures(self, **kwargs):
        """
        Load closures to calculate the meridional overturning streamfunction in
        depth and density space.
        """
        verbose = kwargs.get("verbose", self.verbose)
        _ = [print("Loading mocz closure") if verbose else None]
        mocz_closure = get_mocz_closure(self.ds)
        _ = [print("Loading moct closure") if verbose else None]
        moct_closure = get_moctheta_closure(self.ds)
        def get_mocz(component, T=None, corrected=False):
            try:
                return mocz_closure(self.v[component], T=T, corrected=corrected)
            except KeyError:
                raise ValueError(f"component must be in\n{[i for i in self.v]}")
        def get_moct(component, T=None, corrected=False):
            try:
                return moct_closure(self.vt[component], T=T, corrected=corrected)
            except KeyError:
                raise ValueError(f"component must be in\n{[i for i in self.vt]}")
        self.get_mocz = get_mocz
        self.get_moct = get_moct
        _ = [print("Done") if verbose else None]
        return
    def getdMOC_dt_intgrnd(self, force, ot=True, bt=True, T=None, h_opt="avg"):
        """
        Calculate the integrand for the torque on the meridional overturning
        streamfunction.
        """
        ds = self.ds
        intgrnd = 0
        if ot is True:
            aot = self.get_AOTx(force, T=T)
            aotdims = [i[0] if i.startswith("Z") else i for i in aot.dims]
            aot_dz = xr.DataArray((delta(aot.values, axis=-3)/ds.drF.values[:,None,None]),
                                dims=aotdims, coords={i:ds.coords[i] for i in aotdims})
            intgrnd += aot_dz
        else:
            pass
        if bt is True:
            habt = self.get_HAbt(force, T=T)
            habtdims = [i[0] if i.startswith("X") else i for i in habt.dims]
            Hs = (getHFac(("Z", "Yp1", "X"), ds, option="avg")*ds.drF).sum("Z")
            habt_dxh = (xr.DataArray(delta(habt.values, axis=-1),
                                    dims=habtdims, coords={i:ds.coords[i] for i in habtdims})
                        / (ds.dxG*Hs))
            intgrnd -= habt_dxh
        else:
            pass
        intgrnd = intgrnd*getHFac(("Z", "Yp1", "X"), ds, option="avg")*ds.drF*ds.dxG
        return intgrnd
    def getdMOCz_dt(self, force, ot=True, bt=True, T=None):
        """
        Calculate the torque on the meridional overturning streamfunction in depth space.
        """
        intgrnd = self.getdMOC_dt_intgrnd(force, ot=ot, bt=bt, T=T)
        dmoczdims = ["Zp1" if i.startswith("Z") else i for i in intgrnd.dims if not i.startswith("X")]
        dmocz = (intgrnd.sum("X")
                .pipe(lambda x:
                    xr.DataArray(np.pad(x.values[...,::-1,:].cumsum(axis=-2)[...,::-1,:],
                                        [(0,1) if i.startswith("Z") else (0,0) for i in x.dims]),
                                    dims=dmoczdims, coords={i:self.ds.coords[i] for i in dmoczdims})))
        return dmocz
    def getdMOCt_dt(self, force, tbins, ot=True, bt=True, T=None):
        """
        Calculate the torque on the meridional overturning streamfunction in
        density space.
        """
        ds = self.ds
        intgrnd = self.getdMOC_dt_intgrnd(force, ot=ot, bt=bt, T=T)
        if T is None:
            tmin = ds.THETA.where(ds.HFacC).min(["Z","X"]).values
            tmax = ds.THETA.where(ds.HFacC).max(["Z","X"]).values
            trans = self.get_transform(intgrnd)
        else:
            tmin = ds.THETA.isel(T=T).where(ds.HFacC).min(["Z","X"]).values
            tmax = ds.THETA.isel(T=T).where(ds.HFacC).max(["Z","X"]).values
            trans = self.get_transform(intgrnd, T=T)
            pass
        tmin = bar(tmin, axis=-1, expand=True)
        tmax = bar(tmax, axis=-1, expand=True)
        intgrnd_t = trans(intgrnd.fillna(0), tbins)
        thetaF = intgrnd_t.attrs["thetaF"]
        mask = (np.logical_and(thetaF[:,None] < tmax[...,None,:], thetaF[:,None] > tmin[...,None,:]))
        dmoctdims = ["thetaF" if i=="thetaC" else i for i in intgrnd_t.dims if not i.startswith("X")]
        dmoct_dz = intgrnd_t.sum("X")
        dmoct = (dmoct_dz
                .pipe(lambda x: xr.DataArray(
                    np.pad(x.values.cumsum(axis=-2), [(0,1) if i=="thetaC" else (0,0) for i in x.dims]),
                    dims=dmoctdims,
                    coords={i:thetaF if i=="thetaF" else ds.coords[i] for i in dmoctdims}
                ))
                .where(mask))
        return dmoct
    def pipe(self, fun, *args, **kwargs):
        """
        Utility method equivalent to xarray DataArray pipe method.
        """
        return fun(self, *args, **kwargs)
