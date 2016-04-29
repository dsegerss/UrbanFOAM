# -*- coding: us-ascii -*-
"""
Created on 18 nov 2009
"""
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from osgeo import osr
from osgeo import gdal
from osgeo import ogr
from osgeo.gdalconst import GDT_Float32, GA_ReadOnly

from UrbanFOAM.geometry import buffer_polygon, ring_to_line


DEFAULT_NCELLS = 100  # Number of cells used for homogenious terrain


def write_dataset(data, geotransform, srs_wkt, nodata, filename,
                  format='GTiff'):
    """Write numpy array to disk in gdal format."""

    mem_ds = gdal.GetDriverByName(b'MEM').Create(
        filename.encode('utf-8'),
        data.shape[1],
        data.shape[0],
        1,
        GDT_Float32
    )

    if mem_ds is None:
        raise IOError('Could not write raster %s' % filename)

    mem_ds.SetGeoTransform(geotransform)
    mem_ds.SetProjection(srs_wkt)

    out_band = mem_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)

    # Write data to disk
    out_band.WriteArray(data, 0, 0)
    out_band.FlushCache()
    output_driver = gdal.GetDriverByName(format.encode('utf-8'))
    output_driver.CreateCopy(filename, mem_ds, 0)


def create_geometry_mask(geom, shape, srs_wkt, geotransform, invert=False):
    """
    Create a boolean mask for geometry extent.
    @param geom: the geometry
    @param shape: size of mask (ny, nx)
    @param srs_wkt: spatial reference system in WKT format
    @param geotransform: geotransform of mask (xmin, dx, rot1, ymax, rot2, -dy)
    @param invert: the cells outside the geometry will be set to True
    """
    if invert:
        fill_value = 1
        burn_value = 0
    else:
        fill_value = 0
        burn_value = 1

    srs = osr.SpatialReference()
    srs.ImportFromWkt(srs_wkt.encode('ascii'))
    nx = shape[1]
    ny = shape[0]

    # Create a memory raster to rasterize into.
    mask = gdal.GetDriverByName(b'MEM').Create('', nx, ny, 1, gdal.GDT_Byte)
    mask.SetGeoTransform(geotransform)
    mask.SetProjection(srs_wkt.encode('ascii'))
    mask.GetRasterBand(1).Fill(fill_value)

    # Create a memory layer to rasterize from.
    tmp_ds = ogr.GetDriverByName(b'Memory').CreateDataSource(b'wrk')
    tmp_lyr = tmp_ds.CreateLayer(b'geom', srs=srs)
    feat = ogr.Feature(tmp_lyr.GetLayerDefn())
    feat.SetGeometry(geom)
    tmp_lyr.CreateFeature(feat)
    err = gdal.RasterizeLayer(mask, [1], tmp_lyr, burn_values=[burn_value])
    if err != 0:
        raise ValueError(
            'Error trying to add buffer to terrain' +
            '- could not find cells of terrain intersecting' +
            ' the domain boundaries'
        )

    return mask.GetRasterBand(1).ReadAsArray(0, 0, nx, ny).astype(np.bool_)


def simple_idw(x, y, z, xi, yi):
    """
    Make IDW interpolation
    x, y, z are observations
    xi, yi are points to be interpolated
    """
    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance_matrix
    weights = 1.0 / dist
    # Make weights sum to ones
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi


def distance_matrix(x0, y0, x1, y1):
    """
    Calculate pairwise distances between all obs and target points.
    x0, y0 are observation points
    x1, y1 are points to be interpolated
    """
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    return np.hypot(d0, d1)


class Terrain(object):

    """Class to represent a raster terrain."""

    def __init__(self, domain, srs=None, nodata=-9999.0,
                 cellsize=None, elevation=0):
        """
        Domain: the inner domain polygon (ogr.wkbPolygon object)
        srs: should be a osgeo.osr.SpatialReference object
        elevation: is used as default in case of flat ground
        """
        
        x1, x2, y1, y2 = domain.GetEnvelope()
        if cellsize is None:
            # add 1 to avoid risk of zero cellsize
            cellsize = \
                int(max(x2 - x1, y2 - y1) / DEFAULT_NCELLS) + 1
        # default rotation of raster is zero
        self.domain = domain
        self._geotransform = (x1, cellsize, 0, y2, 0, -1 * cellsize)
        self._invgeotransform = gdal.InvGeoTransform(self._geotransform)
        # Setup working spatial reference
        if srs is None:
            srs_wkt = 'LOCAL_CS["arbitrary"]'
            self.srs = osr.SpatialReference(srs_wkt.encode('ascii'))
        else:
            self.srs = srs

        self.nodata = nodata
        self.data = np.ones(
            (DEFAULT_NCELLS, DEFAULT_NCELLS), dtype=np.float32
        ) * elevation

    @property
    def nx(self):
        """Number of columns."""
        return self.data.shape[1]

    @property
    def ny(self):
        """Number of rows."""
        return self.data.shape[0]
        
    @property
    def xmax(self):
        """x-coordinate of upper right corner."""
        return self.xmin + self.nx * self.dx

    @property
    def ymax(self):
        """x-coordinate of upper right corner."""
        return self.geotransform[3]

    @ymax.setter
    def ymax(self, value):
        gt = list(self._geotransform)
        gt[3] = value
        self.geotransform = gt

    @property
    def xmin(self):
        """x-coordinate of lower left corner."""
        return self.geotransform[0]

    @xmin.setter
    def xmin(self, value):
        gt = list(self._geotransform)
        gt[0] = value
        self.geotransform = tuple(gt)

    @property
    def ymin(self):
        """y-coordinate of lower left corner."""
        return self.ymax - self.ny * self.dy

    @property
    def dx(self):
        return self.geotransform[1]

    @property
    def dy(self):
        return -1 * self.geotransform[5]

    def _limit_data(self):
        """
        Limit data to domain boundaries and
        set cells outside of domain to nodata.
        """

        # create mask for cells outside of domain
        outside_domain_mask = create_geometry_mask(
            self.domain, self.data.shape,
            self.srs.ExportToWkt(),
            self.geotransform,
            invert=True
        )

        not_on_domain_boundary_mask = create_geometry_mask(
            ring_to_line(self.domain.GetGeometryRef(0)),
            self.data.shape,
            self.srs.ExportToWkt(),
            self.geotransform,
            invert=True
        )

        # make sure all cells touching boundary is regarded as inside domain
        outside_domain_mask = (
            outside_domain_mask & not_on_domain_boundary_mask
        )
        # set cell values outside of domain to nodata
        self.data[outside_domain_mask] = self.nodata

        # find first and last indices of cells containing data
        first_row = 0
        last_row = self.ny - 1
        first_col = 0
        last_col = self.nx - 1
        for row in range(self.ny):
            if np.all(self.data[row, :] == self.nodata):
                first_row += 1
            else:
                break
        for row in range(self.ny - 1, 0, -1):
            if np.all(self.data[row, :] == self.nodata):
                last_row -= 1
            else:
                break
        for col in range(self.nx):
            if np.all(self.data[:, col] == self.nodata):
                first_col += 1
            else:
                break
        for col in range(self.nx - 1, 0, -1):
            if np.all(self.data[:, col] == self.nodata):
                last_col -= 1
            else:
                break

        # update geotransform
        self.xmin += first_col * self.dx
        self.ymax -= first_row * self.dy

        # limit data to area with data
        self.data = self.data[
            first_row: last_row + 1,
            first_col: last_col + 1
        ]

    def _get_boundary_height(self):
        """Return the average height around the domain boundary."""

        # Create mask to identify grid cells intersecting domain boundary
        boundary_mask = create_geometry_mask(
            ring_to_line(self.domain.GetGeometryRef(0)),
            self.data.shape,
            self.srs.ExportToWkt(),
            self.geotransform
        )
        return self.data[boundary_mask].mean()
        
    def buffer(self, distance, flat_fraction):
        """
        Smooth cells from raster elevation to average boundary height.
        @param Distance: the buffer width
        @param flat_fraction: the outer flat fraction of the buffer
        """
        self._limit_data()
        flat_boundary = buffer_polygon(
            self.domain, distance * (1 - flat_fraction)
        )

        # padding of raster to make room for buffer
        buffer_nx = np.ceil(distance / self.dx) + 1
        buffer_ny = np.ceil(distance / self.dy) + 1
        self.data = np.pad(
            self.data,
            ((buffer_ny, buffer_ny), (buffer_nx, buffer_nx)),
            mode=b'constant',
            constant_values=(
                (self.nodata, self.nodata),
                (self.nodata, self.nodata)
            )
        )

        # update geotransform for added padding
        self.xmin = self.xmin - buffer_nx * self.dx
        self.ymax = self.ymax + buffer_ny * self.dy

        # Create mask to identify grid cells in the flat part of buffer
        srs_wkt = self.srs.ExportToWkt()
        within_flat_buffer_mask = create_geometry_mask(
            flat_boundary,
            self.data.shape,
            srs_wkt,
            self.geotransform,
            invert=True
        )
        
        # set the height in the outer buffer to average domain boundary height
        boundary_height = self._get_boundary_height()
        self.data[within_flat_buffer_mask] = boundary_height

        # Create mask to identify grid cells on the flat buffer boundary
        on_flat_buffer_boundary_mask = create_geometry_mask(
            ring_to_line(flat_boundary.GetGeometryRef(0)),
            self.data.shape,
            srs_wkt,
            self.geotransform,
        )

        # Create mask to identify grid cells on the domain boundary
        on_domain_boundary_mask = create_geometry_mask(
            ring_to_line(self.domain.GetGeometryRef(0)),
            self.data.shape,
            srs_wkt,
            self.geotransform
        )

        # make sure cells on inner boundary of flat buffer
        # has same height as flat buffer
        self.data[on_flat_buffer_boundary_mask] = boundary_height

        obs_point_mask = on_flat_buffer_boundary_mask + on_domain_boundary_mask
        buffer_mask = (self.data == self.nodata)

        # get coordinates of all cells
        xi = np.linspace(
            self.xmin + 0.5 * self.dx,
            self.xmax - 0.5 * self.dx,
            self.nx
        )
        yi = np.linspace(
            self.ymin + 0.5 * self.dy,
            self.ymax - 0.5 * self.dy,
            self.ny
        )
        xi, yi = np.meshgrid(xi, yi)

        # extract cells to interpolate buffer heights from
        x0 = xi[obs_point_mask]
        y0 = yi[obs_point_mask]
        z0 = self.data[obs_point_mask]
        
        xi_buffer = xi[buffer_mask]
        yi_buffer = yi[buffer_mask]

        # Interpolate and apply result
        zi_buffer = simple_idw(x0, y0, z0, xi_buffer, yi_buffer)
        self.data[buffer_mask] = zi_buffer

    @property
    def geotransform(self):
        return self._geotransform

    @geotransform.setter
    def geotransform(self, value):
        self._geotransform = value
        self._invgeotransform = gdal.InvGeoTransform(value)

    @property
    def invgeotransform(self):
        return self._invgeotransform

    def read(self, filename):
        """Read raster data."""

        datasource = gdal.Open(filename.encode('utf-8'), GA_ReadOnly)
        if datasource is None:
            raise ValueError(
                'Could not open datasource %s' % datasource
            )
        band = datasource.GetRasterBand(1)
        self._nodata = band.GetNoDataValue()
        self.data = band.ReadAsArray().astype(np.float)
        self._srs = datasource.GetProjection()
        self.geotransform = datasource.GetGeoTransform()

        x1, x2, y1, y2 = self.domain.GetEnvelope()
        if self.xmin > x1 or self.ymin > y1 or \
           self.xmax < x2 or self.ymax < y2:
            raise ValueError('Terrain raster does not cover domain extent')

    def __getitem__(self, row, col):
        """Get data using operator [row, col]."""
        return self.data[row, col]

    def corner_points(self):
        ''' Return list of corner points.'''
        if self._geotransform is None or self.data is None:
            return None
                
        ext = []
        xarr = [0, self.data.shape[1]]
        yarr = [0, self.data.shape[0]]

        gt = self._geotransform
        for px in xarr:
            for py in yarr:
                x = gt[0] + (px * gt[1]) + (py * gt[2])
                y = gt[3] + (px * gt[4]) + (py * gt[5])
                ext.append([x, y])
            yarr.reverse()
        return ext

    def sample(self, x, y):
        """Get data at x, y."""
        if self.data is None or self._geotransform is None:
            return None

        ix, iy = gdal.ApplyGeoTransform(self._invgeotransform, x, y)
        col = int(ix) - 1
        row = int(iy) - 1
        try:
            return self.data[row, col]
        except IndexError:
            import pdb;pdb.set_trace()
            raise IndexError('Point (%g, %g) outside terrain' % (x, y))

    def write(self, filename, format='GTiff'):
        """Write terrain to raster file."""
        write_dataset(
            self.data, self.geotransform,
            self.srs.ExportToWkt(), self.nodata,
            filename,
            format=format
        )
