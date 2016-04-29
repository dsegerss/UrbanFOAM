# -*- coding: utf-8 -*-
"""
/***************************************************************************
 UrbanFOAM
                                 A QGIS plugin
 This plugin contains utilities for preparation and analysis of urban stresses using CFD
                             -------------------
        begin                : 2015-12-12
        copyright            : (C) 2015 by David Segersson
        email                : david.segersson@smhi.se
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load UrbanFOAM class from file UrbanFOAM.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .urbanfoam import UrbanFOAM
    return UrbanFOAM(iface)
