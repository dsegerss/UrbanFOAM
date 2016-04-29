# coding=utf-8
"""Geometry tests.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

from __future__ import unicode_literals
from __future__ import division

__author__ = 'david.segersson@smhi.se'
__date__ = '2015-12-12'
__copyright__ = 'Copyright 2015, David Segersson'

import unittest
from UrbanFOAM import geometry
from UrbanFOAM.test import points_to_patch
from UrbanFOAM.db import (
    create_roads_table,
    drop_tables,
    connect,
    load_roads_from_edb
)

from utilities import get_qgis_app
from osgeo import ogr
import triangle
import triangle.plot

import matplotlib.pyplot as plt

QGIS_APP = get_qgis_app()

EPSG = 3006
CONCAVE_POLYGON_WKT = \
    "POLYGON((30 10,21 25,40 40,20 40,10 20,30 10))"

CONCAVE_POLYGON_WITH_HOLE_WKT = \
    "POLYGON((35 10,45 45,30 36,15 40,10 20,35 10),(20 30,35 35,30 20,20 30))"

PLOTTING = False


class UrbanFOAMGeometryTests(unittest.TestCase):
    """Geometry tests."""

    def setUp(self):
        """Runs before each test."""
        self.concave_poly = ogr.CreateGeometryFromWkt(
            CONCAVE_POLYGON_WKT
        )
        self.concave_poly_with_hole = ogr.CreateGeometryFromWkt(
            CONCAVE_POLYGON_WITH_HOLE_WKT
        )

    def tearDown(self):
        """Runs after each test."""
        pass

    def test_remove_holes(self):
        """Test to partition different polygons"""
        poly = geometry.Polygon(self.concave_poly_with_hole.ExportToWkb())
        nholes = poly._poly.GetGeometryCount() - 1
        no_ext_points = poly._poly.GetGeometryRef(0).GetPointCount()
        no_int_points = poly._poly.GetGeometryRef(1).GetPointCount()

        poly.remove_holes()

        if PLOTTING:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            patch = points_to_patch(
                poly._poly.GetGeometryRef(0).GetPoints(),
                facecolor='orange',
                alpha=0.2,
                lw=2
            )
            ax.add_patch(patch)
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)

        self.assertEqual(
            poly._poly.GetGeometryCount(), 1, 'Holes not removed'
        )
        
        # point of exterior is duplicated when incorporating inner ring
        self.assertEqual(
            poly._poly.GetGeometryRef(0).GetPointCount(),
            no_ext_points + no_int_points + nholes
        )

    def test_buffer_road(self):
        edb_con, edb_cur = connect('data/road_buffer_test.sqlite')
        con, cur = connect('data/db.sqlite')
        drop_tables(con, cur, table_names=['roads'])
        create_roads_table(con, cur, EPSG)
        load_roads_from_edb(con, cur, edb_con, EPSG)

    def test_triangulate(self):
        """Test to triangulate polygon."""
        poly = geometry.Polygon(self.concave_poly_with_hole.ExportToWkb())
        # poly.remove_holes()
        if PLOTTING:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)
            patch = points_to_patch(
                poly._poly.GetGeometryRef(0).GetPoints(),
                lw=3,
                alpha=0.2
            )
            ax.add_patch(patch)
            x, y, z = zip(*poly._poly.GetGeometryRef(0).GetPoints())
            ax.plot(x, y, 'o')

        tin = poly.triangulate()
        for tri in tin:
            if PLOTTING:
                patch = points_to_patch(
                    tri.GetGeometryRef(0).GetPoints(),
                    alpha=0.2,
                    lw=1
                )
                ax.add_patch(patch)
                patch = points_to_patch(
                    tri.GetGeometryRef(0).GetPoints(),
                    lw=1
                )
                ax.add_patch(patch)
                plt.draw()

        if PLOTTING:
            plt.show(block=True)

        self.assertTrue(
            len(tin) > 0,
            'No triangles generated'
        )

    def test_PSLG(self):
        """Test creating Planar Straight Line Graph for polygon."""
        poly = geometry.Polygon(self.concave_poly.ExportToWkb())
        pslg, elevations = poly.PSLG()

        self.assertTrue('vertices' in pslg)
        self.assertTrue('segments' in pslg)

    def test_constrained_delaunay(self):
        """Test to triangulate using Triangle package."""
        poly1 = geometry.Polygon(self.concave_poly.ExportToWkb())
        tin1 = poly1.constrained_delaunay(max_segment_length=3)

        poly2 = geometry.Polygon(self.concave_poly_with_hole.ExportToWkb())
        tin2 = poly2.constrained_delaunay(max_segment_length=3)
        if PLOTTING:
            plt.figure()
            ax = plt.subplot(221, aspect='equal')
            triangle.plot.plot(ax, **tin1)
            ax = plt.subplot(222, aspect='equal')
            triangle.plot.plot(ax, **tin2)
            plt.show()

    def test_set_max_segment_length(self):
        """Test to refine polygon around boundaries and triangulate."""
        poly = geometry.Polygon(self.concave_poly_with_hole.ExportToWkb())
        # poly.remove_holes()
        poly.set_max_segment_length(5)

        if PLOTTING:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)
            patch = points_to_patch(
                poly._poly.GetGeometryRef(0).GetPoints(),
                lw=3,
                alpha=0.2
            )
            ax.add_patch(patch)
            x, y, z = zip(*poly._poly.GetGeometryRef(0).GetPoints())
            plt.plot(x, y, 'o')

        tin = poly.triangulate()

        if PLOTTING:
            for tri in tin:
                patch = points_to_patch(
                    tri.GetGeometryRef(0).GetPoints(),
                    alpha=0.2,
                    lw=1
                )
                ax.add_patch(patch)
            plt.show()

    def test_make_domain(self):
        pass

if __name__ == "__main__":
    suite = unittest.makeSuite(UrbanFOAMGeometryTests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

