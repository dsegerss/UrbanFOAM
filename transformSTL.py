#!/usr/bin/env python
# -*- coding: us-ascii -*-
"""Modify ascii STL files."""

from __future__ import unicode_literals
from __future__ import division

from os import path
import sys
import argparse

import numpy as np
from osgeo import ogr

from UrbanFOAM.stl import MultiStl
from pyAirviro.other import logging
from pyAirviro.tools import utils

log = logging.getLogger('pyAirviro.' + __name__)


def to_shp(stl, filename):
    driver = ogr.GetDriverByName(b'ESRI Shapefile')
    if path.exists(filename):
        driver.DeleteDataSource(filename)
    ds = driver.CreateDataSource(filename)
    if ds is None:
        log.error("Could not open output shapefile %s" % filename)
        sys.exit(1)
    layer = ds.CreateLayer(filename, geom_type=ogr.wkbPolygon)
    feature_defn = layer.GetLayerDefn()
    for solid, tin in stl.solids.iteritems():
        for tri in tin:
            feature = ogr.Feature(feature_defn)
            feature.SetGeometry(tri)
            layer.CreateFeature(feature)
            feature.Destroy()
    ds.Destroy()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_standard_command_options(parser)

    parser.add_argument(
        '-i', '--infile',
        dest='infile', action='store',
        help='Input STL'
    )

    parser.add_argument(
        '-o', '--outfile',
        dest='outfile', action='store',
        help='Output STL'
    )

    parser.add_argument(
        '--from-epsg', dest='from_epsg', type=int,
        action='store',
        help='Current EPSG for STL'
    )

    parser.add_argument(
        '--to-epsg', dest='to_epsg', type=int,
        action='store',
        help='New EPSG for STL'
    )

    parser.add_argument(
        '--move', dest='move', metavar='C',
        type=float, nargs=3,
        action='store',
        help='Move STL by '
    )

    parser.add_argument(
        '--single', dest='single', action='store_true',
        help='Write each solid in a separate file'
    )

    parser.add_argument(
        '--decimals', dest='decimals', type=int,
        action='store',
        help='Number of decimals to use in STL'
    )

    parser.add_argument(
        '--format', dest='format', action='store', default='stl',
        help='Write to format (stl(default), shp)'
    )

    args = parser.parse_args()

    stl = MultiStl()
    stl.read(args.infile)

    if args.to_epsg is not None and args.from_epsg != args.to_epsg:
        if args.from_epsg is None:
            log.error('Must provide both --from-epsg and --to-epsg')
            sys.exit(1)
        stl.transform(args.from_epsg, args.to_epsg)
    if args.move is not None:
        stl.move(*args.move)

    if args.decimals is None:
        bbox = stl.bounding_box()
        dx = bbox[1][0] - bbox[0][0]
        dy = bbox[1][1] - bbox[0][1]
        dz = bbox[1][2] - bbox[0][2]
        decimals = abs(int(np.log(abs(min(dx, dy, dz) / 1.0e6))))
    else:
        decimals = args.decimals

    if args.format == 'stl':
        stl.write(args.outfile, decimals=decimals, single=args.single)
    elif args.format == 'shp':
        to_shp(stl, args.outfile)
    else:
        log.error('Unknown format specified')

if __name__ == '__main__':
    main()
