# coding=utf-8
"""Geometry tests.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'david.segersson@smhi.se'
__date__ = '2015-12-12'
__copyright__ = 'Copyright 2015, David Segersson'

import unittest
from UrbanFOAM import geometry
from UrbanFOAM.test import points_to_patch
from UrbanFOAM.triangulation import Tin, tri2vertices

from osgeo import ogr

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

CONCAVE_POLYGON_WKT = \
    b"POLYGON((30 10,21 25,40 40,20 40,10 20,30 10))"

CONCAVE_POLYGON_WITH_HOLE_WKT = \
    b"POLYGON((35 10,45 45,30 36,15 40,10 20,35 10),(20 30,35 35,30 20,20 30))"

DOMAIN_POLYGON_WKT = \
    b"POLYGON((-20 -20,50 -20,50 50, 50 -20, -20 -20))"

MAX_SEGMENT_LENGTH = 2

PLOTTING = True


class UrbanFOAMTriangulationTests(unittest.TestCase):
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

    def test_contrained_delaunay(self):
        vertices = []
        segments = []

        nrings = self.domain.GetGeometryCount()
        for i in range(nrings):
            ring = self.domain.GetGeometryRef(i)
            points = ring.GetPoints()
            vertices += points[:-1]
            for i in range(len(points)):
                
                

    # def test_refine(self):
    #     """Test to triangulate polygon."""
    #     poly = geometry.Polygon(self.concave_poly_with_hole.ExportToWkb())

    #     poly.remove_holes()

    #     poly.set_max_segment_length(MAX_SEGMENT_LENGTH)

    #     if PLOTTING:
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot(221)
    #         ax1.set_xlim(0, 50)
    #         ax1.set_ylim(0, 50)
    #         ax2 = fig.add_subplot(222)
    #         ax2.set_xlim(0, 50)
    #         ax2.set_ylim(0, 50)
    #         plt.ion()
    #         plt.show()

    #         patch = points_to_patch(
    #             poly._poly.GetGeometryRef(0).GetPoints(),
    #             lw=3,
    #             facecolor='none'
    #         )
    #         ax1.add_patch(patch)

    #     triangles = poly.triangulate()
        
    #     tin = Tin()
    #     for tri in triangles:
    #         tin.add_triangle(*tri2vertices(tri))

    #     if PLOTTING:
    #         for tri in tin.triangles.get_objects():
    #             patch = points_to_patch(
    #                 tri.asarray(),
    #                 alpha=0.2,
    #                 lw=3
    #             )
    #             ax1.add_patch(patch)
    #             plt.draw()

    #     tin_refined = tin.refine(MAX_SEGMENT_LENGTH)

    #     # if PLOTTING:
    #     #     fig = plt.figure()
    #     #     ax = fig.add_subplot(111)
    #     #     for tri in tin_refined.triangles.get_objects():
    #     #         tri.patch = points_to_patch([tri.v1, tri.v2, tri.v3])
    #     #         ax.add_patch(tri.patch)
    #     #         plt.draw()

    #     if PLOTTING:
    #         plt.show(block=True)

    #     self.assertEqual(
    #         len(tin_refined.triangles._index), 14,
    #         'Invalid triangles?, uncomment plot.show() to confirm'
    #     )
