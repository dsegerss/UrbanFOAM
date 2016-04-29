# coding=utf-8
"""Tests for distmesh2d.

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
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from UrbanFOAM import geometry
from UrbanFOAM.test import points_to_patch
from UrbanFOAM.distmesh2d import distmesh2d

CONCAVE_POLYGON_WKT = \
    b"POLYGON((30 10,21 25,40 40,20 40,10 20,30 10))"

CONCAVE_POLYGON_WITH_HOLE_WKT = \
    b"POLYGON((35 10,45 45,30 36,15 40,10 20,35 10),(20 30,35 35,30 20,20 30))"

DOMAIN_POLYGON_WKT = \
    b"POLYGON((-20 -20,50 -20,50 50, 50 -20, -20 -20))"

MAX_SEGMENT_LENGTH = 2
PLOTTING = True


class UrbanFOAMDistmesh2dTests(unittest.TestCase):
    """Geometry tests."""

    def setUp(self):
        """Runs before each test."""
        self.concave_poly = ogr.CreateGeometryFromWkt(
            CONCAVE_POLYGON_WKT
        )
        self.concave_poly_with_hole = ogr.CreateGeometryFromWkt(
            CONCAVE_POLYGON_WITH_HOLE_WKT
        )

        self.domain = ogr.CreateGeometryFromWkt(
            DOMAIN_POLYGON_WKT
        )

    def tearDown(self):
        """Runs after each test."""
        pass

    def test_refine(self):
        """Test to triangulate polygon."""
        
        
