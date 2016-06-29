# coding=utf-8

"""
Module for creation of multi solid STLs.
"""

from os import path, mkdir
import numpy as np
from osgeo import ogr
from osgeo import osr


def tri2stl(tri, decimals=3):
    """Get ogr triangle polygon in stl format."""
    points = np.array(tri.GetGeometryRef(0).GetPoints()[:3])
    p1, p2, p3 = points.round(decimals=decimals)
    
    # calculate face normal
    n = np.cross(p2 - p1, p3 - p1)
    # tri_str = "  facet normal %.2e %.2e %.2e\n" % (n[0], n[1], n[2])
    # tri_str += "    outer loop\n"
    # tri_str += "      vertex %.2e %.2e %.2e\n" % (p1[0], p1[1], p1[2])
    # tri_str += "      vertex %.2e %.2e %.2e\n" % (p2[0], p2[1], p2[2])
    # tri_str += "      vertex %.2e %.2e %.2e\n" % (p3[0], p3[1], p3[2])
    # tri_str += "    endloop\n"
    # tri_str += "  endfacet\n"

    tri_str = """  facet normal {n[0]:.%(dec)ie} {n[1]:.%(dec)ie} {n[2]:.%(dec)ie}
    outer loop
      vertex {p1[0]:.%(dec)ie} {p1[1]:.%(dec)ie} {p1[2]:.%(dec)ie}
      vertex {p2[0]:.%(dec)ie} {p2[1]:.%(dec)ie} {p2[2]:.%(dec)ie}
      vertex {p3[0]:.%(dec)ie} {p3[1]:.%(dec)ie} {p3[2]:.%(dec)ie}
    endloop
  endfacet
""" % ({'dec': decimals})

    return tri_str.format(p1=p1, p2=p2, p3=p3, n=n)


def createtri(p1, p2, p3):
    """Create polygon triangle from points."""
    return ogr.CreateGeometryFromWkt(
        "POLYGON Z (({x1} {y1} {z1},{x2} {y2} {z2},{x3} {y3} {z3},{x1} {y1} {z1}))".format(
            x1=p1[0], y1=p1[1], z1=p1[2],
            x2=p2[0], y2=p2[1], z2=p2[2],
            x3=p3[0], y3=p3[1], z3=p3[2]
        )
    )


def tri2array(tri):
    return np.array(tri.GetGeometryRef(0).GetPoints()[:3])
                

class MultiStl(object):
    """Container class for multiple STL solids."""

    def __init__(self, solids=None):
        """
        Initialize MultiSolid
        @param solids: a dictionary {name: [tri1, tri2, tri3, ..., triN]}
        """

        self.solids = solids or {}

    def bounding_box(self):
        minx = miny = minz = 9e8
        maxx = maxy = maxz = -9e8
        
        for solid, tin in self.solids.iteritems():
            for tri in tin:
                points = tri2array(tri)
                minx = min(points[:, 0].min(), minx)
                miny = min(points[:, 1].min(), miny)
                minz = min(points[:, 2].min(), minz)
                maxx = max(points[:, 0].max(), maxx)
                maxy = max(points[:, 1].max(), maxy)
                maxz = max(points[:, 2].max(), maxz)

        return ((minx, miny, minz), (maxx, maxy, maxz))

    def move(self, x, y, z=0):
        new_solids = {}
        for solid, tin in self.solids.iteritems():
            new_tris = []
            for tri in tin:
                points = tri2array(tri)
                points[:, 0] += x
                points[:, 1] += y
                if z != 0:
                    points[:, 2] += z
                new_tris.append(createtri(*points))
            new_solids[solid] = new_tris
        self.solids = new_solids

    def transform(self, from_epsg, to_epsg):
        source = osr.SpatialReference()
        source.ImportFromEPSG(from_epsg)
        target = osr.SpatialReference()
        target.ImportFromEPSG(to_epsg)
        transform = osr.CoordinateTransformation(source, target)
        for solid, tin in self.solids.iteritems():
            for tri in tin:
                tri.Transform(transform)

    def add_solid(self, name, tin):
        """
        Add solid to group name
        @param name: add solid to group 'name'
        @param tin: triangles representing solid
        """
        if name in self.solids:
            self.solids[name] += tin
        else:
            self.solids[name] = tin

    def read(self, filename):
        """Read STL in ascii format."""

        with open(filename, 'r') as stl:
            lines = iter(stl)
            lineno = 0
            solid_incomplete = False
            tin = []
            while 1:
                try:
                    line = lines.next()
                except StopIteration:
                    if solid_incomplete:
                        raise ValueError('STL ends in incomplete solid')
                    else:
                        break
                lineno += 1
                if 'outer loop' in line:
                    try:
                        v1 = lines.next().strip().split()[1:]
                        lineno += 1
                        v2 = lines.next().strip().split()[1:]
                        lineno += 1
                        v3 = lines.next().strip().split()[1:]
                        lineno += 1
                        tin.append(createtri(v1, v2, v3))
                    except StopIteration:
                        raise ValueError(
                            'Incomplete triangle on line ' +
                            '%i of %s' % (lineno, filename)
                        )
                elif line.startswith('solid'):
                    solid_name = line.split()[1]
                    solid_incomplete = True
                    if len(tin) > 0:
                        raise ValueError(
                            'Invalid ending of solid at line %i' % (lineno - 1)
                        )
                elif line.startswith('endsolid'):
                    self.add_solid(solid_name, tin)
                    solid_incomplete = False
                    tin = []

    def write(self, filename, decimals=3, single=False):
        """Write STL to file in ascii-format."""
        if not single:
            with open(filename, 'w') as stl:
                for name, tin in self.solids.iteritems():
                    stl.write('solid %s\n' % name)
                    for tri in tin:
                        stl.write(tri2stl(tri, decimals=decimals))
                    stl.write('endsolid %s\n' % name)
        else:
            if not path.exists(filename):
                mkdir(filename)
            for name, tin in self.solids.iteritems():
                with open(path.join(filename, name + '.stl'), 'w') as stl:
                    stl.write('solid %s\n' % name)
                    for tri in tin:
                        stl.write(tri2stl(tri, decimals=decimals))
                    stl.write('endsolid %s\n' % name)
                    
