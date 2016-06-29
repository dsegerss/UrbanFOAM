# coding=utf-8

"""
Module for easy setup of OpenFOAM cases.
"""
from __future__ import absolute_import
from __future__ import unicode_literals

import os
from os import path
from itertools import combinations
from operator import attrgetter

from osgeo import osr
from osgeo import ogr

import numpy as np

from UrbanFOAM import db
from UrbanFOAM import ofdict
from UrbanFOAM.defaults import DEFAULTS
from UrbanFOAM.stl import MultiStl
from UrbanFOAM.terrain import Terrain
from UrbanFOAM.geometry import (
    Structure, Road, Polygon, buffer_polygon,
    set_tin_elevation,
    extrude_and_triangulate_patch,
    make_triangle,
    HEIGHT_REF
)

ROADS_PATCH_NAME = 'roads'
TOLERANCE = 0.001
EMISSION_GROUP_NAMES = ['LDV', 'HDV']


class Case(object):

    """An OpenFOAM case for urban wind and dispersion simulations."""

    def __init__(self, domain=None, terrain=None, structures=None,
                 roads=None, point_sources=None, area_sources=None, srs=None,
                 buffer=None, buffer_flat_fraction=None, domain_height=None):

        """Initialize Domain object.
        @param domain: Polygon object representing domain
        @param terrain: Terrain object
        @param structures: list of Structures
        @param roads: list of Roads
        @param point_sources: list of PointSource objects
        @param area_sources: list of AreaSource objects
        @param srs: Spatial reference system as WKT
        @param buffer: domain buffer width
        """
        self.srs = srs
        self.buffer_flat_fraction = buffer_flat_fraction or \
            DEFAULTS['buffer_flat_fraction']
        self.domain_height = domain_height or DEFAULTS['domain_height']
        self.buffer = buffer or DEFAULTS['buffer']
        self.domain = domain
        self.roads = roads or []
        self.structures = structures or []
        self.point_sources = point_sources or []
        self.area_sources = area_sources or []
        self.terrain = terrain

        for key, default_value in DEFAULTS.iteritems():
            if not hasattr(self, key):
                setattr(self, key, default_value)
            elif attrgetter(key)(self) is None:
                setattr(self, key, default_value)

    @property
    def srs(self):
        return self._srs

    @srs.setter
    def srs(self, value):
        wkt = value or 'LOCAL_CS["arbitrary"]'
        self._srs = osr.SpatialReference(wkt.encode('ascii'))

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        if value is None:
            self._domain = None
            self._flat_zone = None
            self._boundary = None
        else:
            self._domain = Polygon(value)
            # create domain's outer boundary by buffering the domain
            # store buffered domain in self.boundary
            self._boundary = buffer_polygon(self._domain.poly, self.buffer)
            self._flat_zone = buffer_polygon(
                self._domain.poly,
                self.buffer * (1 - self.buffer_flat_fraction)
            )
            
    def distance_to_origo(self):
        """Return distance from domain centroid to origo."""
        if self.domain is None:
            return (0, 0)
        dist = self.domain._poly.Centroid().GetPoint()[:2]
        dist = map(int, dist)
        return (-1 * dist[0], -1 * dist[1])
        
    def create_case(self, casepath):
        if not path.exists(casepath):
            try:
                os.mkdir(casepath)
            except:
                raise IOError('Could not create case directory %s' % casepath)

        constant_dir = path.join(casepath, 'constant')
        if not path.exists(constant_dir):
            try:
                os.mkdir(constant_dir)
            except:
                raise IOError(
                    'Could not create costant directory %s' % constant_dir
                )

        triSurface_dir = path.join(casepath, 'constant', 'triSurface')
        if not path.exists(triSurface_dir):
            try:
                os.mkdir(triSurface_dir)
            except:
                raise IOError(
                    'Could not create directory %s' % triSurface_dir
                )
        
    def read(self, con):
        """Read using sqlite db connection."""
        self.read_settings(con)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(db.get_epsg(con))
        self.srs = srs.ExportToWkt()
        self.read_domain(con)
        self.read_roads(con)
        self.read_structures(con)
        self.create_patch_from_roads(con)
        self.read_point_sources(con)
        self.read_area_sources(con)
    
    def remove_overlaps(self):
        """Remove overlaps between structures and roads."""
        # TODO: How can nested buildings be handled?
        # STL of inner buildings will have zero thickness walls...
        
        # sort with highest structure first
        self.structures.sort(key=lambda s: s.to_height(self.terrain))
        
        # check for overlaps between structures at same height
        new_structures = []
        for i, j in combinations(range(len(self.structures)), 2):
            s1 = self.structures[min(i, j)]
            s2 = self.structures[max(i, j)]
            if s1.intersects(s2):
                # subtract s1 from s2
                new_poly = s2.poly.difference(s1.poly)
                if new_poly.GetGeometryName() == 'POLYGON':
                    s2.poly = new_poly
                elif new_poly.GetGeometryName() == 'MULTIPOLYGON':
                    s2.poly = new_poly.GetGeometryRef(0)
                    for poly_ind in range(1, new_poly.GetGeometryCount()):
                        s2.poly = new_poly.GetGeometryRef(0)
                        new_structures.append(
                            Structure(
                                new_poly.GetGeometryRef(
                                    poly_ind).ExportToWkb(),
                                height_ref=s2.height_ref,
                                from_height=s2.from_height,
                                to_height=s2.to_height,
                                patch=s2.patch
                            )
                        )

    def to_stl(self, outdir, translate=None):
        """ Write stl files for domain, structures, roads and sources."""

        if self.terrain is None:
            raise ValueError('Must supply a Terrain before writing STL')

        domain = Polygon(self._boundary.ExportToWkb(), self.domain.id)
        domain.set_max_segment_length(5)

        # calculate domain upper boundary height
        domain_height = self.terrain.data.max() + self.domain_height
        
        domain_stl = MultiStl()

        # extrude interior triangles to form upper boundary
        internal_tin = domain.triangulate(upward=True)
        top_tin = []

        # set elevation of upper boundary
        for i in range(len(internal_tin)):
            tri = internal_tin[i]
            p1, p2, p3 = [
                list(p) for p in tri.GetGeometryRef(0).GetPoints()[:3]
            ]
            p1[2] = domain_height
            p2[2] = domain_height
            p3[2] = domain_height
            top_tin.append(make_triangle(p1, p2, p3))

        domain_stl.add_solid('top', top_tin)
        # extrude lateral boundaries
        # skip duplicate point
        points = domain.exterior_points()
        for i in range(1, len(points)):
            side_tin = []
            x1, y1 = points[i - 1][:2]
            x2, y2 = points[i][:2]

            p1 = (x1, y1, self.terrain.sample(x1, y1))
            p2 = (x2, y2, self.terrain.sample(x2, y2))

            p1_extruded = (x1, y1, domain_height)
            p2_extruded = (x2, y2, domain_height)
            side_tin.append(make_triangle(p2, p1, p1_extruded))
            side_tin.append(make_triangle(p2_extruded, p2, p1_extruded))
            
            # segment direction vector
            vec = np.array(p2[:2]) - np.array(p1[:2])
            x, y = vec
            vec_normal = np.array([-y, x])  # rotate 90 degrees clockwise

            # anti-clockwise angle from north
            vec_normal_angle = np.arctan2(
                vec_normal[1], vec_normal[0]) * 180 / np.pi - 90

            vec_normal_angle *= -1

            if vec_normal_angle < 0:
                vec_normal_angle += 360
            if vec_normal_angle >= 360:
                vec_normal_angle -= 360

            group_name = 'side%i' % int(vec_normal_angle)
            domain_stl.add_solid(group_name, side_tin)

        if translate is not None:
            domain_stl.move(*translate)
        domain_stl.write(path.join(outdir, 'domain.stl'))

        # dividing structures into patches (zero thickness) and buildings
        patches = [s for s in self.structures if s.is_patch_only()]
        buildings = [s for s in self.structures if not s.is_patch_only()]
        
        # subdivide boundaries of buildings and patches
        map(
            lambda b: b.set_max_segment_length(
                DEFAULTS['MAX_SEGMENT_LENGTH_BUILDINGS']
            ),
            buildings
        )
        map(
            lambda p: p.set_max_segment_length(
                DEFAULTS['MAX_SEGMENT_LENGTH_PATCHES']
            ),
            patches
        )

        map(
            lambda r: r.set_max_segment_length(
                DEFAULTS['MAX_SEGMENT_LENGTH_ROADS']
            ),
            self.roads
        )

        # ground is stored in a separate stl for further processing.
        ground_stl = MultiStl()
        
        # patches may not overlap
        # they are only subtracted from domain, an overlap will
        # cause duplicate or overlapping triangles in STL
        for patch in patches:
            if patch.patch is None:
                raise ValueError('Patches must have a name specified')

            # subtract patch polygon from ground polygon
            # might return MULTIPOLYGON
            domain._poly = domain._poly.Difference(patch._poly)

            if patch._to_height == 0 and (
                    patch._from_height is None or
                    patch.height_ref == HEIGHT_REF['ground']):
                # ground following non-elevated patch
                structure_tin = set_tin_elevation(
                    patch.triangulate(upward=False), terrain=self.terrain
                )
                ground_stl.add_solid(patch.patch, structure_tin)
            else:
                # elevated ground following
                # or above ground max elevation
                # or elevated to absolute height
                structure_tin = extrude_and_triangulate_patch(
                    patch, self.terrain
                )
                ground_stl.add_solid(patch.patch, structure_tin)

        # domain.remove_holes()
        ground_tin = domain.triangulate(upward=False)
        ground_tin = set_tin_elevation(ground_tin, terrain=self.terrain)
        ground_stl.add_solid('ground', ground_tin)
        if translate is not None:
            ground_stl.move(*translate)
        ground_stl.write(path.join(outdir, 'ground.stl'))
        
        structure_stl = MultiStl()
        for b in buildings:
            tin = b.extrude(self.terrain)
            structure_stl.add_solid(b.patch or 'structure_%i' % b.id, tin)

        if len(buildings) > 0:
            if translate is not None:
                structure_stl.move(*translate)
            structure_stl.write(path.join(outdir, 'structures.stl'))

        road_stl = MultiStl()
        for road in self.roads:
            road_tin = road.extrude(self.terrain)
            road_stl.add_solid(
                road.patch or 'road_%i' % road.sourceid,
                road_tin
            )

        if len(self.roads) > 0:
            if translate is not None:
                road_stl.move(*translate)
            road_stl.write(path.join(outdir, 'roads'), single=True)
            road_stl.write(path.join(outdir, 'roads.stl'), single=False)

    @property
    def terrain(self):
        return self._terrain

    @terrain.setter
    def terrain(self, terrain):
        """set terrain and update references to structures, roads etc."""
        self._terrain = terrain
        
        if terrain is not None:
            self._terrain.buffer(self.buffer, self.buffer_flat_fraction)

    def read_terrain(self, filename):
        """read terrain from file."""
        terrain = Terrain(self._domain._poly)
        terrain.read(filename)
        self.terrain = terrain

    def read_settings(self, con, **kwargs):
        recs = con.execute(
            """
            SELECT * from settings
            """
        )
        
        rec_dict = dict(
            ((rec['key'], rec['value']) for rec in recs)
        )
            
        # get general settings for domain
        for key, default_value in DEFAULTS.iteritems():
            if hasattr(self, key):
                value = attrgetter(key)(self)
                if value is None:
                    setattr(self, key, rec_dict.get(key, default_value))
            else:
                setattr(self, key, rec_dict.get(key, default_value))

    def read_domain(self, con):
        """Extract domain from db."""
        domains = con.execute(
            'SELECT ST_AsBinary(geom) as wkb FROM domain'
        ).fetchall()
        ndomains = len(domains)
        if ndomains == 0:
            raise ValueError('No case domain found in db')

        self.domain = bytes(domains[0]['wkb'])

        if not self.domain._poly.IsValid():
            raise ValueError(
                'Domain polygon is invalid, check for self-intersection, ' +
                'duplicate vertices and similar errors'
            )

    def read_structures(self, con):
        """Read structures from db."""

        # only retrieve features within domain boundaries
        structures = con.execute(
            """
            SELECT s.id, p.name as patch, s.height_ref,
                s.from_height, s.to_height,
            ST_AsBinary(
                ST_Intersection(s.geom, ST_GeomFromText('{wkt}'))
            ) as wkb
            FROM structures as s
            JOIN patches as p
            ON p.id=s.patch
            WHERE ST_Intersects(geom, ST_GeomFromText('{wkt}'))

            """.format(wkt=self.domain._poly.ExportToWkt())
            )

        for rec in structures:
            # polygons might be converted to multipolygons
            # when intersecting with domain
            # if so, they must be converted to singlepart polygons
            geom = ogr.CreateGeometryFromWkb(bytes(rec['wkb']))
            if geom.GetGeometryName() == 'MULTIPOLYGON':
                npolys = geom.GetGeometryCount()
                for i in range(npolys):
                    self.structures.append(
                        Structure(
                            geom.GetGeometryRef(i).ExportToWkb(),
                            id=rec['id'],
                            patch=rec['patch'],
                            height_ref=rec['height_ref'] or 1,
                            from_height=rec['from_height'],
                            to_height=rec['to_height']
                        )
                    )
            else:
                self.structures.append(
                    Structure(
                        geom.ExportToWkb(),
                        patch=rec['patch'],
                        height_ref=rec['height_ref'] or 1,
                        from_height=rec['from_height'],
                        to_height=rec['to_height']
                    )

                )

    def read_roads(self, con):
        """Read roads from db."""
        # only retrieve features within domain boundaries
        # clip roads that cross the domain boundary
        query = \
            """
            SELECT id, patch, source, height_ref, from_height, to_height,
                ST_AsText(
                  ST_Intersection(geom, ST_GeomFromText('{wkt}'))
                ) as wkt,
                ST_AsBinary(
                  ST_Intersection(geom, ST_GeomFromText('{wkt}'))
                ) as wkb
            FROM roads
            WHERE ST_Intersects(geom, ST_GeomFromText('{wkt}'))
            AND wkb IS NOT NULL
            """.format(wkt=self.domain._poly.ExportToWkt())

        recs = con.execute(query)
        for rec in recs:
            self.roads.append(
                Road(
                    bytes(rec['wkb']),
                    rec['source'],
                    id=rec['id'],
                    patch=rec['patch'],
                    height_ref=rec['height_ref'],
                    from_height=rec['from_height'],
                    to_height=rec['to_height']
                )
            )

    def create_patch_from_roads(self, con):
        """Create path for union of all roads.
        TODO: cannot handle crossing roads on different levels
        """

        union_query = \
            """
            SELECT ST_AsBinary(ST_Union(geom)) as wkb
            FROM roads
            """
        recs = con.execute(union_query).fetchall()
        geom = ogr.CreateGeometryFromWkb(bytes(recs[0]['wkb']))
        
        if geom.GetGeometryName() == 'MULTIPOLYGON':
            npolys = geom.GetGeometryCount()
            for i in range(npolys):
                self.structures.append(
                    Structure(
                        geom.GetGeometryRef(i).ExportToWkb(),
                        patch='roads',
                        height_ref=0,
                        from_height=None,
                        to_height=0
                    )
                )
        else:
            self.structures.append(
                Structure(
                    geom.ExportToWkb(),
                    patch='roads',
                    height_ref=0,
                    from_height=None,
                    to_height=0
                )

            )
            
    def read_point_sources(self, con):
        pass

    def read_area_sources(self, con):
        pass
    
    def write_landuse_dict(self, filename):
        x, y = self.distance_to_origo()
        header = ofdict.DICT_HEADER.format(filename='landuseDict')
        content = ofdict.LANDUSE_DICT_TEMPLATE.format(
            subtractedX=x,
            subtractedY=y
        )
        with open(filename, 'w') as outfile:
            outfile.write(
                header + content
            )

    def write_traffic_dict(self, edb_con, emis_grp_fractions, filename,
                           translate=None):
        road_names = []
        centrelines = []
        road_properties = []
        road_classification = []

        road_name_index = 0
        for road in self.roads:
            centreline = road.get_centreline(
                edb_con,
                translate=translate
            )

            road_names.append('road_%i' % road.sourceid)
            centrelines.append(centreline['points'])
            road_properties.append(
                '({speed} {nolanes} {vehicles})'.format(**centreline)
            )

            road_classification.append(
                '({name_index} 0 {LDV_weight})'.format(
                    name_index=road_name_index,
                    LDV_weight=emis_grp_fractions[0, road.sourceid]
                )
            )
            road_classification.append(
                '({name_index} 1 {HDV_weight})'.format(
                    name_index=road_name_index,
                    HDV_weight=emis_grp_fractions[1, road.sourceid]
                )
            )
            road_name_index +=1 
            
        header = ofdict.DICT_HEADER.format(filename='trafficDict')
        content = ofdict.TRAFFIC_DICT_TEMPLATE.format(
            road_names='\n'.join(road_names),
            centrelines='\n'.join(centrelines),
            road_properties='\n'.join(road_properties),
            emission_group_names='\n'.join(EMISSION_GROUP_NAMES),
            road_classification='\n'.join(road_classification)
        )

        with open(filename, 'w') as outfile:
            outfile.write(
                header + content
            )
