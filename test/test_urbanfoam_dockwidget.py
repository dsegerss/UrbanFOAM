# coding=utf-8
"""DockWidget test.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'david.segersson@smhi.se'
__date__ = '2015-12-12'
__copyright__ = 'Copyright 2015, David Segersson'

import unittest

from PyQt4.QtGui import QDockWidget

from urbanfoam_dockwidget import UrbanFOAMDockWidget

from utilities import get_qgis_app

QGIS_APP = get_qgis_app()


class UrbanFOAMDockWidgetTest(unittest.TestCase):
    """Test dockwidget works."""

    def setUp(self):
        """Runs before each test."""
        self.dockwidget = UrbanFOAMDockWidget(None)

    def tearDown(self):
        """Runs after each test."""
        self.dockwidget = None

    def test_dockwidget_ok(self):
        """Test we can click OK."""
        pass

if __name__ == "__main__":
    suite = unittest.makeSuite(UrbanFOAMDialogTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

