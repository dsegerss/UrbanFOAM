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

import sys
import unittest
from pyAirviro.test.edb import get_sqlite_test_edb
import UrbanFOAM.createcase.main as createcase
from UrbanFOAM.case import Case
from UrbanFOAM.db import (
    create_roads_table,
    drop_tables,
    connect,
    load_roads_from_edb
)

from utilities import get_qgis_app

EPSG = 3006


class UrbanFOAMToolTests(unittest.TestCase):
    """Test command line utilities of UrbanFOAM."""

    def setUp(self):
        """Runs before each test."""
        con, cur = connect('data/db.sqlite')
        self.case = Case()
        self.case.read(con)
        self.case.read_terrain('data/terrain_eskilstuna.asc')

    def tearDown(self):
        """Runs after each test."""
        pass

    # def test_calculate_road_vehicle_ts(self):
    #     edb_con, edb_cur = get_sqlite_test_edb()
    #     con, cur = connect(':memory')
    #     sys.argv = ['createcase', '--initdb', '
    #     createcase()


if __name__ == "__main__":
    suite = unittest.makeSuite(UrbanFOAMToolTests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

