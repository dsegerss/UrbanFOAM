# coding=utf-8
"""Tests for case management.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'david.segersson@smhi.se'
__date__ = '2015-12-12'
__copyright__ = 'Copyright 2015, David Segersson'

import unittest
import tempfile
from shutil import rmtree

from UrbanFOAM.db import connect
from UrbanFOAM.case import Case

EPSG = 3006


class UrbanFOAMCaseTests(unittest.TestCase):
    """Testing UrbanFOAM db module."""

    def setUp(self):
        """Runs before each test."""
        con, cur = connect('data/db.sqlite')
        self.case = Case()
        self.case.read(con)
        self.case.read_terrain('data/terrain_eskilstuna.asc')

    def tearDown(self):
        """Runs after each test."""
        pass
    
    def test_to_stl(self):
        """Test to triangulate, extrude and write stl."""
        try:
            temp_dir = tempfile.mkdtemp(prefix='tmp_')
            self.case.to_stl(temp_dir)
        finally:
            print temp_dir
            # rmtree(temp_dir)
            


        
