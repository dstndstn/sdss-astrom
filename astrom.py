from astrometry.libkd.spherematch import *
from astrometry.util.fits import *
import numpy as np
from astrometry.util.starutil_numpy import *
from astrometry.util.plotutils import *
from glob import glob
from collections import Counter
from legacyanalysis.gaiacat import *
from legacypipe.survey import radec_at_mjd

def radec_to_munu(ra, dec, node, incl):
    '''
    RA,Dec in degrees
    mu,nu (great circle coords) in degrees
    '''
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    node,incl = np.deg2rad(node), np.deg2rad(incl)
    mu = node + np.arctan2(np.sin(ra - node) * np.cos(dec) * np.cos(incl) +
                               np.sin(dec) * np.sin(incl),
                               np.cos(ra - node) * np.cos(dec))
    nu = np.arcsin(-np.sin(ra - node) * np.cos(dec) * np.sin(incl) +
                       np.sin(dec) * np.cos(incl))
    mu, nu = np.rad2deg(mu), np.rad2deg(nu)
    mu += (360. * (mu < 0))
    mu -= (360. * (mu > 360))
    return mu, nu

def nantozero(x):
    x = x.copy()
    x[np.logical_not(np.isfinite(x))] = 0.
    return x

def year_to_mjd(year):
    # year_to_mjd(2015.5) -> 57205.875
    from tractor.tractortime import TAITime
    return (year - 2000.) * TAITime.daysperyear + TAITime.mjd2k

def main():
    sdss_dir = '/project/projectdirs/cosmo/data/sdss/dr13/eboss'

    gaiacat = GaiaCatalog()
    
    W = fits_table(os.path.join(sdss_dir, 'resolve/2013-07-29/window_flist.fits'))
    
    # Resume previous run?
    if False:
        Results = fits_table('sdss-astrom.fits')
        IW = np.flatnonzero(Results.did_fit == False)
    else:
        Results = fits_table()
        Results.run = W.run
        Results.camcol = W.camcol
        Results.field = W.field
        Results.did_fit = np.zeros(len(W), bool)
        Results.ra_offset_median  = np.zeros(len(W), np.float32)
        Results.dec_offset_median = np.zeros(len(W), np.float32)
        Results.chisq_before_fit  = np.zeros(len(W), np.float32)
        Results.chisq_after_fit  = np.zeros(len(W), np.float32)
        Results.ra_offset  = np.zeros(len(W), np.float32)
        Results.dec_offset = np.zeros(len(W), np.float32)
        Results.dra_drow  = np.zeros(len(W), np.float32)
        Results.dra_dcol  = np.zeros(len(W), np.float32)
        Results.ddec_drow = np.zeros(len(W), np.float32)
        Results.ddec_dcol = np.zeros(len(W), np.float32)
        Results.n_sdss = np.zeros(len(W), np.int16)
        Results.n_gaia = np.zeros(len(W), np.int16)
        Results.n_sdss_good = np.zeros(len(W), np.int16)
        Results.n_gaia_good = np.zeros(len(W), np.int16)
        Results.n_matched = np.zeros(len(W), np.int16)
        IW = range(len(W))


    allmatches = []
    for iw_index,iw in enumerate(IW):
        w = W[iw]
        print()
        print(iw+1, 'of', len(W), '    R/C/F', w.run, w.camcol, w.field)
    
        fn = os.path.join(sdss_dir, 'photoObj/301/%i/%i/photoObj-%06i-%i-%04i.fits' % (w.run, w.camcol, w.run, w.camcol, w.field))
        print('Reading', fn)
        if not os.path.exists(fn):
            print('No such SDSS file:', fn)
            Results.n_sdss[iw] = 0
            continue
        SDSS = fits_table(fn)
        if SDSS is None:
            print('No such SDSS file')
            Results.n_sdss[iw] = 0
            continue

        Results.n_sdss[iw] = len(SDSS)
        print(len(SDSS), 'SDSS')
        ramin, ramax, decmin, decmax = SDSS.ra.min(), SDSS.ra.max(), SDSS.dec.min(), SDSS.dec.max()
    
        not_blended = np.logical_not((SDSS.objc_flags & 8 != 0) * ((SDSS.objc_flags & 64) == 0))
        not_bright  = (SDSS.objc_flags & 2 == 0)
        not_child   = (SDSS.objc_flags & 16 == 0)
        star = (SDSS.objc_type == 6)
        not_badflags = (SDSS.objc_flags & 4589572 == 0)
        not_badcenter = (SDSS.objc_flags2 & 6144 == 0)
        primary = ((SDSS.resolve_status & 1) != 0)
        not_edge = (SDSS.objc_rowc >= 64) * (SDSS.objc_rowc <= 1489-64)
        SDSS.cut(not_blended * not_bright *
                 #not_child *
                 star * not_badflags *
                 not_badcenter * primary * not_edge)
        Results.n_sdss_good[iw] = len(SDSS)
        print(len(SDSS), 'SDSS good')
        if len(SDSS) == 0:
            print('No good SDSS!')
            continue

        if (SDSS.ra.max() - SDSS.ra.min() > 180.):
            print('SDSS RA=0 wrap-around')
            mn = SDSS.ra[SDSS.ra > 180.].min()
            mx = SDSS.ra[SDSS.ra < 180.].max()
            g1 = gaiacat.get_catalog_radec_box(mn, 360., decmin, decmax)
            g2 = gaiacat.get_catalog_radec_box(0., mx, decmin, decmax)
            Gaia = merge_tables([g1,g2])
            print(len(Gaia), 'Gaia in box', len(g1), len(g2))
        else:
            Gaia = gaiacat.get_catalog_radec_box(ramin, ramax, decmin, decmax)
            print(len(Gaia), 'Gaia in box')

        mu_start = np.fmod(w.mu_start, 360.)
        mu_end = (w.mu_end - w.mu_start) + mu_start
        mu,nu = radec_to_munu(Gaia.ra, Gaia.dec, w.node, w.incl)
        if mu_end > 360.:
            # wrap
            mu += (mu < 180.) * 360.
        keep_gaia = ((mu >= mu_start) * (mu <= mu_end) * (nu >= w.nu_start) * (nu <= w.nu_end))
        #Counter(keep_gaia)
        Gaia.cut(keep_gaia)
        Results.n_gaia[iw] = len(Gaia)
        print(len(Gaia), 'Gaia in munu')
        if len(Gaia) == 0:
            print('No Gaia in munu')
            continue

        Gaia.pmra = nantozero(Gaia.pmra)
        Gaia.pmdec = nantozero(Gaia.pmdec)
        Gaia.parallax = nantozero(Gaia.parallax)

        sn = nantozero(Gaia.parallax / Gaia.parallax_error)
        Gaia.parallax[sn < 5] = 0.

        gra,gdec = radec_at_mjd(Gaia.ra, Gaia.dec, Gaia.ref_epoch,
                                Gaia.pmra, Gaia.pmdec, Gaia.parallax, w.mjd)

        keep = np.logical_or(Gaia.astrometric_excess_noise == 0,
                             Gaia.astrometric_excess_noise <= (0.3 * Gaia.phot_g_mean_mag - 5.3))

        # Keep good Gaia stars moved to SDSS MJD
        Gaia.ra = gra
        Gaia.dec = gdec
        Gaia.cut(keep)
        #ra = gra[keep]
        #dec = gdec[keep]
        #I,J,d = match_radec(ra, dec, SDSS.ra, SDSS.dec, 0.1/3600., nearest=True)
        I,J,d = match_radec(Gaia.ra, Gaia.dec, SDSS.ra, SDSS.dec, 0.2/3600.,
                            nearest=True)
        print(len(Gaia), 'Gaia good')
        print(len(I), 'matched')
        Results.n_gaia_good[iw] = len(Gaia)
        Results.n_matched[iw] = len(I)
        if len(I) == 0:
            print('No matches')
            continue

        Gaia.cut(I)
        SDSS.cut(J)

        dra  = (Gaia.ra  - SDSS.ra ) * 3600. * 1000. * np.cos(np.deg2rad(Gaia.dec))
        ddec = (Gaia.dec - SDSS.dec) * 3600. * 1000.

        A = np.zeros((len(I), 3))
        A[:,0] = 1.
        A[:,1] = (SDSS.objc_colc - 1024)
        A[:,2] = (SDSS.objc_rowc -  744)

        wra = 1./SDSS.raerr**2
        wra[np.logical_not(np.isfinite(wra))] = 0.
        R = np.linalg.lstsq(A * wra[:,np.newaxis], dra * wra)
        ra_terms = R[0]

        wdec = 1./SDSS.decerr**2
        wdec[np.logical_not(np.isfinite(wdec))] = 0.
        R = np.linalg.lstsq(A * wdec[:,np.newaxis], ddec * wdec)
        dec_terms = R[0]

        print('Offsets RA %.0f mas + %.1f mas/kcol + %.1f mas/krow,  Dec %.0f mas + %.3f mas/kcol + %.3f mas/krow' %
              (ra_terms[0], 1e3 * ra_terms[1], 1e3 * ra_terms[2],
               dec_terms[0], 1e3 * dec_terms[1], 1e3 * dec_terms[2]))

        fitdra = dra - (ra_terms[0] +
                        ra_terms[1] * (SDSS.objc_colc - 1024.) +
                        ra_terms[2] * (SDSS.objc_rowc -  744.))
        fitddec = ddec - (dec_terms[0] +
                          dec_terms[1] * (SDSS.objc_colc - 1024.) +
                          dec_terms[2] * (SDSS.objc_rowc -  744.))

        Results.ra_offset_median [iw] = np.median(dra)
        Results.dec_offset_median[iw] = np.median(ddec)
        Results.chisq_before_fit[iw] = np.sum((dra / (SDSS.raerr*1000.))**2 +
                                              (ddec / (SDSS.decerr*1000.))**2)
        Results.chisq_after_fit[iw] = np.sum((fitdra / (SDSS.raerr*1000.))**2 +
                                             (fitddec / (SDSS.decerr*1000.))**2)
        Results.ra_offset [iw] = ra_terms[0]
        Results.dec_offset[iw] = dec_terms[0]
        Results.dra_dcol  [iw] = ra_terms[1]
        Results.dra_drow  [iw] = ra_terms[2]
        Results.ddec_dcol [iw] = dec_terms[1]
        Results.ddec_drow [iw] = dec_terms[2]
        Results.did_fit[iw] = True

        if iw and (iw % 10000 == 0):
            print('Writing intermediate results')
            Results.writeto('sdss-astrom.fits')

        matches = fits_table()
        matches.gaia_source_id = Gaia.source_id
        matches.gaia_mjd = year_to_mjd(Gaia.ref_epoch).astype(np.float32)
        for c in ['astrometric_excess_noise', 'phot_g_mean_mag',
                  'parallax', 'parallax_error', 'pmra', 'pmra_error',
                  'pmdec', 'pmdec_error', 'phot_bp_mean_mag', 'phot_rp_mean_mag']:
            matches.set(c, Gaia.get(c))

        matches.ra = Gaia.ra
        matches.dec = Gaia.dec
        matches.dra = dra.astype(np.float32)
        matches.ddec = ddec.astype(np.float32)
        matches.fit_dra = fitdra.astype(np.float32)
        matches.fit_ddec = fitddec.astype(np.float32)
        matches.sdss_ra = SDSS.ra
        matches.sdss_dec = SDSS.dec
        sc = 3600.*1000.*np.cos(np.deg2rad(SDSS.dec))
        matches.sdss_fit_ra = (SDSS.ra +
                               (ra_terms[0] +
                                ra_terms[1] * (SDSS.objc_colc - 1024.) +
                                ra_terms[2] * (SDSS.objc_rowc -  744.)) / sc)
        sc = 3600.*1000.
        matches.sdss_fit_dec = (SDSS.dec +
                               (dec_terms[0] +
                                dec_terms[1] * (SDSS.objc_colc - 1024.) +
                                dec_terms[2] * (SDSS.objc_rowc -  744.)) / sc)
        matches.sdss_objid = SDSS.objid
        matches.sdss_mjd = SDSS.mjd
        for c in ['cmodelflux', 'objc_rowc', 'objc_colc', 'objc_type']:
            matches.set(c, SDSS.get(c))
        matches.airmass = SDSS.airmass[:,2]

        matches.run       = np.array([w.run]   *len(matches))
        matches.camcol    = np.array([w.camcol]*len(matches))
        matches.field     = np.array([w.field] *len(matches))
        matches.field_ra  = np.array([w.ra]    *len(matches))
        matches.field_dec = np.array([w.dec]   *len(matches))
        matches.window_flist_row = np.array([iw] * len(matches))

        allmatches.append(matches)

        write_matches = False
        if iw_index == len(IW)-1:
            # last one?
            write_matches = True
        else:
            write_matches = (w.run != W.run[IW[iw_index+1]])

        if write_matches:
            allmatches = merge_tables(allmatches)
            fn = '/global/cscratch1/sd/dstn/sdss-gaia/matches-run%04i.fits' % w.run
            allmatches.writeto(fn)
            print('Wrote', fn)
            allmatches = []

    #Results = Results[:20]

    Results.writeto('sdss-astrom.fits')



if __name__ == '__main__':
    main()
