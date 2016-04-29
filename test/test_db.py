# coding=utf-8
"""Tests for db management.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'david.segersson@smhi.se'
__date__ = '2015-12-12'
__copyright__ = 'Copyright 2015, David Segersson'

import unittest

from UrbanFOAM.db import connect, initdb, TABLES, get_epsg

EPSG = 3006


class UrbanFOAMDbTests(unittest.TestCase):
    """Testing UrbanFOAM db module."""

    def setUp(self):
        """Runs before each test."""
        self.con, self.cur = connect(':memory:')
        initdb(self.con, self.cur, EPSG)

    def tearDown(self):
        """Runs after each test."""
        pass

    def test_get_epsg(self):
        """Test to get spatial reference of db."""
        self.assertEqual(get_epsg(self.con), EPSG)
        
    def test_table_in_db(self):
        """Test check if table exists in db."""
        for name in TABLES:
            self.assertTrue(self.cur, name)
            
