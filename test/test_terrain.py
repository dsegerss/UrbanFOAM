# coding=utf-8
"""Terrain tests.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'david.segersson@smhi.se'
__date__ = '2015-12-12'
__copyright__ = 'Copyright 2015, David Segersson'

import unittest

from osgeo import ogr
from UrbanFOAM.terrain import Terrain


DOMAIN_POLYGON_WKT = \
    "POLYGON((5 5, 48 5, 48 48, 5 48, 5 5))"

BUFFER = 100
FLAT_BUFFER_FRACTION = 0.2


class UrbanFOAMTerrainTests(unittest.TestCase):
    """Testing UrbanFOAM terrain class."""

    def setUp(self):
        """Runs before each test."""
        self.domain = ogr.CreateGeometryFromWkt(
            DOMAIN_POLYGON_WKT
        )

    def tearDown(self):
        """Runs after each test."""
        pass
    
    def test_read(self):
        """Test to read a DEM raster."""
        terrain = Terrain(self.domain)
        terrain.read('data/terrain.asc')
        self.assertEqual(terrain.data.max(), 99)
        self.assertEqual(terrain.data.shape, (100, 100))
        self.assertEqual(terrain.xmin, -100)
        self.assertEqual(terrain.xmax, 100)
        self.assertEqual(terrain.ymin, -100)
        self.assertEqual(terrain.ymax, 100)
        self.assertEqual(terrain.dx, 2)
        self.assertEqual(terrain.dy, 2)

    def test_limit(self):
        """Test to limit terrain to domain boundaries."""
        terrain = Terrain(self.domain)
        terrain.read('data/terrain.asc')
        nx, ny = terrain.data.shape
        xmin = terrain.xmin
        ymax = terrain.ymax

        terrain._limit_data()
        self.assertTrue(terrain.data.shape[0] < ny)
        self.assertTrue(terrain.data.shape[1] < nx)
        self.assertTrue(terrain.xmin > xmin)
        self.assertTrue(terrain.ymax < ymax)

    def test_buffer(self):
        """Test to buffer terrain."""
        terrain = Terrain(self.domain)
        terrain.read('data/terrain.asc')
        avg_hgt = terrain._get_boundary_height()
        terrain.buffer(BUFFER, FLAT_BUFFER_FRACTION)
        self.assertEqual(terrain.data[0, 0], avg_hgt)
        
        
