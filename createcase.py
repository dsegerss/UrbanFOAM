#!/usr/bin/env python
# -*- coding: us-ascii -*-
"""Resolve sources for emission calculations."""

from __future__ import unicode_literals
from __future__ import division

import os
import sys
import argparse
import datetime
from os import path

try:
    from pysqlite2 import dbapi2 as sqlite3
except ImportError:
    try:
        import sqlite3
    except:
        pass

from UrbanFOAM.case import (
    Case,
    DEFAULTS
)

from UrbanFOAM.db import (
    validate_roads_in_db,
    connect,
    load_roads_from_edb,
    initdb,
    get_epsg,
    drop_tables,
    create_roads_table,
    create_domain_table,
    create_structures_table,
    get_road_fractions_in_polys
)

from pyAirviro.other import logging
from pyAirviro.tools import utils
from pyAirviro.edb.sqlitecalc import calculate_road_vehicle_ts

log = logging.getLogger('pyAirviro.' + __name__)
ENCODING = 'utf-8'


def arg2datetime(string):
    return datetime.datetime.strptime(string, '%y%m%d%H')


def calculate_emission_group_fractions(emis_ts):
    """Sum emissions by emission group (light/heavy vechicles) and
    calculate the fraction contributed by each road.
    @param emis_ts: dataframe with levels (substance, isheavy, vehicle, road)
    """
    isheavy_sums = emis_ts.groupby(
        level=['isheavy'], axis=1
    ).sum()
    isheavy_accumulated = isheavy_sums.sum()
    road_accumulated = emis_ts.groupby(
        level=['isheavy', 'road'], axis=1
    ).sum().sum()
    
    road_accumulated[0] /= isheavy_accumulated[0]
    road_accumulated[1] /= isheavy_accumulated[1]
    return (isheavy_sums, road_accumulated)


def correct_by_fraction_in_poly(emis_ts, len_frac_in_poly):
    """Scale emission by the fraction of the road within the road polygon.
    @param emis_ts: dataframe
    @param len_frac_in_poly: dict with (roadid, fraction)

    Dataframe must have multiindex (substance, isheavy, vehicle, roadid)
    and be lexically sorted to it's full depth
    """
    substances = emis_ts.columns.levels[0].tolist()
    isheavy = emis_ts.columns.levels[1].tolist()
    vehicles = emis_ts.columns.levels[2].tolist()

    for road, frac in len_frac_in_poly.iteritems():
        emis_ts.loc[
            :,
            (
                substances,
                isheavy,
                vehicles,
                road
            )
        ] *= frac


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_standard_command_options(parser)

    parser.add_argument(
        '--info',
        dest='info', action='store_true',
        help='Print version of sqlite, spatialite, proj and geos'
    )

    parser.add_argument(
        '-e', metavar='edb',
        dest='edb',
        help='name of edb to load',
    )

    parser.add_argument(
        '-d',
        action='store', dest='db', metavar='FILENAME',
        help='Sqlite database file'
    )

    parser.add_argument(
        '--buffer',
        action='store', type=float, dest='buffer', default=DEFAULTS['buffer'],
        help='Domain buffer width (meters), default=%(default)s'
    )

    parser.add_argument(
        '--flat-buffer',
        action='store', dest='buffer',
        default=DEFAULTS['buffer_flat_fraction'],
        help='Fraction of buffer that should be completely' +
        ' flat, default=%(default)s'
    )

    parser.add_argument(
        '--max-segment-len',
        action='store', dest='max_segment_len',
        default=5,
        help='Maximum length of triangulation edgesx, default=%(default)s'
    )

    parser.add_argument(
        '--height',
        action='store', dest='height',
        default=DEFAULTS['domain_height'],
        help='Domain height above ground, default=%(default)s'
    )

    parser.add_argument(
        '-s',
        action='store', type=int, metavar='SUBST',
        dest='substances', nargs='+',
        help='Substance indices to create emission data for'
    )

    parser.add_argument(
        '--scenario', action='store', type=int,
        dest='scenario', default=1,
        help='Scenario index to create emission views for'
    )

    parser.add_argument(
        '-t', '--terrain',
        action='store', dest='terrain',
        help='Terrain raster file'
    )

    parser.add_argument(
        '-c', '--case',
        action='store', dest='case',
        help='Output case directory'
    )

    parser.add_argument(
        '--no-roads',
        action='store_false', dest='roads',
        help='Do not generate road STLs or adapt ground stl to roads'
    )
        
    parser.add_argument(
        '--load-roads',
        action='store_true', dest='load_roads',
        help='Load roads from edb and generate road polygons'
    )

    parser.add_argument(
        '--init-roads',
        action='store_true', dest='init_roads',
        help='Create empty roads table in case db' +
        ' before loading roads from edb'
    )

    parser.add_argument(
        '--init-domain',
        action='store_true', dest='init_domain',
        help='Create empty domain table in case db'
    )

    parser.add_argument(
        '--init-structures',
        action='store_true', dest='init_structures',
        help='Create empty structures table in case db'
    )

    parser.add_argument(
        '--init', dest='initdb',
        action='store_true',
        help='Create new or overwrite existing case db'
    )

    parser.add_argument(
        '--epsg', dest='epsg', type=int, default=3006,
        action='store',
        help='EPSG to use when creating new case db'
    )

    parser.add_argument(
        '--emis-ts', dest='emis_ts',
        action='store_true',
        help='Write emission timeseries to case'
    )

    parser.add_argument(
        '--begin', dest='begin', metavar='YYMMDDHH',
        action='store', type=arg2datetime,
        help='First hour of emission timeseries'
    )

    parser.add_argument(
        '--end', dest='end', metavar='YYMMDDHH',
        action='store', type=arg2datetime,
        help='First hour of emission timeseries'
    )

    parser.add_argument(
        '--translate', action='store', nargs=3, type=float,
        dest='translate',
        help='Translate STL (default is origo in centroid of domain)'
    )

    # parser.add_argument(
    #     '-s', '--sources',
    #     action='store_true', dest='sources',
    #     help='Generate emissions for sources'
    # )

    args = parser.parse_args()

    if args.info:
        con, cur = connect(":memory:")
        # testing library versions
        row = cur.execute(
            'SELECT sqlite_version(), spatialite_version()'
        ).next()
        msg = "> SQLite v%s Spatialite v%s" % (row[0], row[1])
        log.info(msg)
        sys.exit(0)

    if args.db is None:
        log.info('Connecting to in-memory db')
    else:
        log.info('Connecting to %s' % args.db)
    try:
        con, cur = connect(args.db or ':memory')
    except sqlite3.OperationalError as e:
        log.error(str(e))
        sys.exit(1)

    if args.initdb or not path.exists(args.db):
        log.info('Initializing case database %s...' % args.db)
        initdb(con, cur, args.epsg)
        sys.exit(0)

    epsg = get_epsg(con)

    if args.init_roads:
        drop_tables(con, cur, table_names=['roads'])
        create_roads_table(con, cur, epsg)

    if args.init_domain:
        drop_tables(con, cur, table_names=['domain'])
        create_domain_table(con, cur, epsg)

    if args.init_structures:
        drop_tables(con, cur, table_names=['structures'])
        create_structures_table(con, cur, epsg)

    if args.edb is not None:
        edb_con, edb_cur = connect(args.edb)
        edb_epsg = get_epsg(edb_con)

        if args.load_roads:
            load_roads_from_edb(con, cur, edb_con, edb_epsg)
            log.info('Updated roads from edb')
            sys.exit(0)
    
    case = Case(buffer=args.buffer)

    if args.case is not None:
        case.create_case(args.case)

    log.info('Reading geometry')
    case.read(con)

    if args.terrain is not None:
        log.info('Reading terrain')
        case.read_terrain(args.terrain)

    if args.edb is not None:
        roadids = []
        if args.emis_ts:
            log.info('Validating roads')
            validation_errors = validate_roads_in_db(con)
            if len(validation_errors) > 0:
                log.error(
                    '\nRoadid Validation error\n' +
                    '\n'.join(
                        ('%-6i %-s' % (err['id'], err['reason'])
                         for err in validation_errors)
                    )
                )
                sys.exit(1)
            else:
                log.info('Roads have valid geometries')

            if args.begin is None or args.end is None:
                log.error(
                    'Must specify time interval of emission time-series'
                )
                sys.exit(1)

            log.info(
                'Calculates fraction of road sources within road polygons'
            )
            len_frac_in_poly = get_road_fractions_in_polys(
                cur, edb_cur
            )

            for roadid, len_frac in len_frac_in_poly.iteritems():
                if len_frac == 0:
                    log.warning(
                        'Road source %i does not intersect the' % roadid +
                        ' corresponding road polygon %i'
                        )
                else:
                    roadids.append(roadid)
            
            log.info('Calculates road vehicle emission timeseries')
            emis_ts = calculate_road_vehicle_ts(
                edb_con, edb_cur, args.begin, args.end,
                args.substances, roadids=roadids
            )

            log.info(
                'Scale emission timeseries by fraction within road polygons'
            )
            correct_by_fraction_in_poly(emis_ts, len_frac_in_poly)

            log.info('Group road emissions')
            emission_group_ts, emis_group_fractions = \
                calculate_emission_group_fractions(
                    emis_ts
                )
            # convert to mg/s
            emission_group_ts *= 1000.0
            log.info('Writing constant/emissionTimeSeries_mg_per_s.csv')
            emission_group_ts.to_csv(
                path_or_buf=path.join(
                    args.case, 'constant', 'emissionTimeSeries_mg_per_s.csv'
                ),
                sep=b'\t',
                index_label='Time',
                date_format='%y%m%d%H',
                header=['LDV', 'HDV']
            )

    if args.case is not None:
        if args.emis_ts:
            log.info('Writing constant/trafficDict')
            case.write_traffic_dict(
                edb_con,
                emis_group_fractions,
                path.join(args.case, 'constant', 'trafficDict'),
                translate=args.translate
            )

        case.write_landuse_dict(
            path.join(args.case, 'constant', 'landuseDict')
        )

        log.info('Creating stl-files')
        if args.terrain is None:
            log.error('Must specify terrain file to generate STL')
            sys.exit(1)
        case.to_stl(
            path.join(args.case, 'constant', 'triSurface'),
            translate=args.translate or case.distance_to_origo()
        )

    
if __name__ == '__main__':
    main()
