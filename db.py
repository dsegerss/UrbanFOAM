# -*- coding: us-ascii -*-
"""Various utilities for pyAirviro."""

from __future__ import absolute_import
from __future__ import unicode_literals

try:
    from pysqlite2 import dbapi2 as sqlite3
except ImportError:
    try:
        import sqlite3
    except:
        pass

import numpy as np
from osgeo import ogr
from UrbanFOAM.geometry import buffer_road
from UrbanFOAM.defaults import DEFAULTS


TABLES = [
    'settings',
    'domain',
    'structures',
    'patches',
    'roads',
    'road_centrelines',
    'point_sources',  # point and area sources
    'area_sources',  # point and area sources
    'volume_sources',
    'landuse',
    'canopy_profiles'
]

GEOMETRY_TABLES_COLUMNS = [
    ('domain', 'geom'),
    ('structures', 'geom'),
    ('roads', 'geom')
    
]

GEOMETRY_COLUMN_NAME = 'geom'

SETTINGS = [
    ('ROAD_MIXING_HEIGHT', 4)
]


def connect(filename):
    """Connect to database."""
    con = sqlite3.connect(filename)
    con.enable_load_extension(True)
    try:
        con.execute("select load_extension('libspatialite')")
    except:
        con.execute("select load_extension('mod_spatialite')")
    con.enable_load_extension(False)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute('PRAGMA foreign_keys = ON')
    return con, cur


def validate_roads_in_db(con):
    """Validate road geometries in db."""

    # ST_AsText(ST_IsValidDetail(geom)) to get details on geometry
    return con.execute(
        """
        SELECT id, ST_IsValidReason(geom) as reason
        FROM roads
        WHERE ST_IsValid(geom) <> 1;
        """
    ).fetchall()


def execute_sql(cur, cmd, **kwargs):
    """
    Exectute cmd on cur, filling placeholders with kwargs.
    If cmd is a tuple it is assumed to contain (sql, value_dict),
    where sql uses named sqlite placeholders.

    """
    if isinstance(cmd, tuple):
        cmd, values = cmd
        cmd = cmd % (kwargs)
        # if values is not a dict, it is assumed to be a sequence of dicts
        if not isinstance(values, dict):
            cur.executemany(cmd, values)
        else:
            cur.execute(cmd, values)
    else:
        cur.execute(cmd % (kwargs))


def initdb(con, cur, epsg):
    if not table_in_db(cur, 'spatial_ref_sys'):
        cur.execute('SELECT InitSpatialMetaData()')
    drop_tables(con, cur)
    create_tables(con, cur, epsg)

    con.executemany(
        'INSERT INTO settings (key, value) VALUES (?, ?)',
        SETTINGS
    )


def get_epsg(con):
    try:
        epsg = con.execute(
            "SELECT srid FROM geometry_columns"
        ).next()[0]
    except sqlite3.OperationalError:
        epsg = None
    return epsg


def drop_tables(con, cur, table_names=None):
    """Drop tables.
    @param con: db connection
    @param cur: db cursor
    @param table_names: iterable with table names
    """
    rows = cur.execute('SELECT f_table_name FROM geometry_columns')
    geometry_columns = [row['f_table_name'] for row in rows.fetchall()]
    tables = table_names or TABLES
    for table in tables:
        if table.lower() in geometry_columns:
            cur.execute(
                "SELECT DiscardGeometryColumn('%s' , '%s')" % (
                    table.lower(),
                    GEOMETRY_COLUMN_NAME
                )
            )
        cur.execute('DROP TABLE IF EXISTS %s' % table)
    con.commit()


def create_roads_table(con, cur, epsg):
    """Create table for roads. """
    # height_ref: 0 ground, 1 average, 2 absolute
    # from_height: NULL means from ground level
    # to_height: NULL means to geom height
    # generated from edb road source data
    cur.execute(
        '''
        CREATE TABLE roads (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        patch VARCHAR(100) DEFAULT 'roads',
        source INTEGER,
        height_ref INTEGER DEFAULT 0,
        from_height REAL,
        to_height REAL,
        FOREIGN KEY (patch) REFERENCES patches (name) ON UPDATE CASCADE
        );
        '''
    )

    cur.execute(
        "SELECT AddGeometryColumn('roads', 'geom'," +
        "%i, 'POLYGON', 'XYZ', 1)" % epsg
    )


def create_structures_table(con, cur, epsg):
    # height_ref: 0 ground, 1 average, 2 absolute
    # from_height: NULL means from ground level
    # to_height: NULL means to geom height,
    cur.execute(
        '''
        CREATE TABLE structures (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        patch VARCHAR(100) DEFAULT 'buildings',
        height_ref INTEGER DEFAULT 1,
        from_height REAL,
        to_height REAL,
        FOREIGN KEY (patch) REFERENCES patches (name) ON UPDATE CASCADE
        );
        '''
    )

    cur.execute(
        "SELECT AddGeometryColumn('structures', 'geom'," +
        "%i, 'POLYGON', 'XYZ', 1)" % epsg
    )


def create_domain_table(con, cur, epsg):
    """Create table for domain. """
    cur.execute(
        '''
        CREATE TABLE domain (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name VARCHAR(100) NOT NULL
        );
        '''
    )
    cur.execute(
        "SELECT AddGeometryColumn('domain', 'geom'," +
        "%i, 'POLYGON', 'XY', 1)" % epsg
    )
    

def create_tables(con, cur, epsg):
    """Create edb tables (and drop old if any exist)."""

    cur.execute(
        """
        CREATE TABLE settings (
        key VARCHAR(100) PRIMARY KEY NOT NULL,
        value REAL NOT NULL
        );
        """
    )

    create_domain_table(con, cur, epsg)

    cur.execute(
        '''
        CREATE TABLE patches (
        name VARCHAR(100) PRIMARY KEY NOT NULL
        );
        '''
    )

    cur.executemany(
        '''
        INSERT INTO patches (name) VALUES (:name)
        ''',
        [
            {'name': 'buildings'},
            {'name': 'water'},
            {'name': 'terrace'},
            {'name': 'roads'},
            {'name': 'chimneys'}
        ]
    )

    create_structures_table(con, cur, epsg)
    create_roads_table(con, cur, epsg)
    
    # # from_height:  NULL means from ground level
    # # to_height: NULL means to geom height (0 - only mesh constraint)
    # cur.execute(
    #     '''
    #     CREATE TABLE road_centrelines (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    #     name VARCHAR(100) NOT NULL,
    #     speed INTEGER DEFAULT 0,
    #     from_height REAL,
    #     to_height REAL
    #     );
    #     '''
    # )

    # Generated from edb point source data
    # always given a unique patch name
    # cur.execute(
    #     '''
    #     CREATE TABLE chimney_outlets (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    #     name VARCHAR(100) NOT NULL,
    #     patch VARCHAR(100) NOT NULL',
    #     source INTEGER NOT NULL,
    #     FOREIGN KEY (patch) REFERENCES patches (name) ON UPDATE CASCADE
    #     );
    #     '''
    # )

    # # height_ref: 0 ground, 1 centroid, 2 absolute
    # # from_height: NULL means from ground level
    # # to_height: NULL means to geom height (0 - only mesh constraint)
    # cur.execute(
    #     '''
    #     CREATE TABLE area_sources (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    #     name VARCHAR(100) NOT NULL,
    #     patch VARCHAR(100) NOT NULL,
    #     source INTEGER NOT NULL,
    #     height_ref INTEGER DEFAULT 0,
    #     from_height REAL,
    #     to_height REAL,
    #     FOREIGN KEY (patch) REFERENCES patches (name) ON UPDATE CASCADE
    #     );
    #     '''
    # )

    # cur.execute(
    #     "SELECT AddGeometryColumn('area_sources', 'geom'," +
    #     "%i, 'POLYGON', 'XYZ', 1)" % epsg
    # )
    con.commit()


def value_in_column(cur, table_name, column_name, value):
    """True if given value exists in column."""
    if isinstance(value, basestring):
        value = "'%s'" % value
    sql = """
    SELECT 1 FROM views_geometry_columns WHERE
    {column_name}={value}
    """.format(
        column_name=column_name,
        table_name=table_name,
        value=value
    )
    try:
        if cur.execute(sql).next()[0]:
            return True
        else:
            return False
    except StopIteration:
        return False


def table_in_db(cur, table_name, table_type='table'):
    """True if table is found in db."""
    rows = cur.execute(
        """
        SELECT 1 from sqlite_master
        WHERE type='{table_type}' and name ='{table_name}'""".format(
            table_name=table_name,
            table_type=table_type
        )
    )

    try:
        if rows.next()[0]:
            return True
        else:
            return False
    except StopIteration:
        return False


def get_road_fractions_in_polys(cur, edb_cur):
    """Get dict for source id's and fraction of length within road polygon.
    Assumes road centreline is not crossing polygon borders multiple times
    """
    road_fracs = {}

    road_polygons = cur.execute(
        """
        SELECT source, AsText(geom) as wkt
        FROM roads
        """
    ).fetchall()

    for road in road_polygons:
        # get length fraction within road polygon
        centrelines = edb_cur.execute(
            """
            SELECT id,
                (GLength(ST_Intersection(geom, ST_GeomFromText('{wkt}'))) /
                GLength(geom)) as len_frac
            FROM roads
            WHERE id= {sourceid}
            """.format(wkt=road['wkt'], sourceid=road['source'])
        ).fetchall()

        if centrelines != []:
            rec = centrelines[0]
            road_fracs[rec['id']] = rec['len_frac'] or 0
        else:
            raise ValueError(
                'No road source with id %i found in edb' % (
                    road['source']
                )
            )
    return road_fracs


def load_roads_from_edb(con, cur, edb_con, epsg):
    """Generate and load road polyogons from edb.
    @param con: connection to case db,
    @param cur: cursor to case db,
    @param edb_con: connection to edb,
    @param espg: epsg of edb
    """

    domains = cur.execute(
        'SELECT ST_AsText(geom) as wkt FROM domain'
    ).fetchall()

    if len(domains) == 0:
            raise ValueError('No case domain found in db')

    # get parts of roads that intersect the domain
    road_recs = edb_con.execute(
        """
        SELECT id, name, width,
            ST_AsBinary(ST_Intersection(geom, ST_GeomFromText('{wkt}'))) as wkb
        FROM roads
        WHERE ST_Intersects(geom, ST_GeomFromText('{wkt}'))
        """.format(wkt=domains[0]['wkt'])
    )

    node_index = {}
    road_index = {}
    for rec in road_recs:
        # convert to ogr Geometry
        line = ogr.CreateGeometryFromWkb(bytes(rec['wkb']))
        # ogr.Geometry(bytrec['wkt'])
        
        # skip zero size roads
        if line.GetPointCount() <= 1:
            continue

        # snap all nodes by rounding coordinates
        points = np.array(line.GetPoints()).round(decimals=1)
        if points.shape[1] == 2:
            points = np.hstack(
                (points, np.zeros((points.shape[0], 1), dtype=np.float32))
            )
        
        key1 = tuple(points[0].tolist())
        key2 = tuple(points[-1].tolist())

        road_index[rec['id']] = (points, rec['width'])

        # index all lines by start point and endpoint
        # index point is always the first point of the line
        node_index.setdefault(key1, []).append(
            (rec['id'], rec['width'], points)
        )
        node_index.setdefault(key2, []).append(
            (rec['id'], rec['width'], np.flipud(points))
        )

    # insert parts of roads into case db
    for road_id in road_index.keys():
        road_points, width = road_index[road_id]
        links_before = [
            l for l in node_index[tuple(road_points[0].tolist())]
            if l[0] != road_id
        ]
        links_after = [
            l for l in node_index[tuple(road_points[-1].tolist())]
            if l[0] != road_id
        ]
        
        poly_wkt = buffer_road(
            road_points, width, links_before, links_after
        )

        sql = """ \
            INSERT INTO roads (
            id, patch, source, height_ref,
            from_height, to_height, geom) VALUES
            (NULL, NULL, {source}, 0, NULL, {height},
            ST_GeomFromText('{wkt}', {epsg}))
        """.format(
            wkt=poly_wkt,
            source=road_id,
            height=DEFAULTS['road_height'],
            epsg=epsg
        )
        
        cur.execute(sql)
        con.commit()
        
    con.commit()
