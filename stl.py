# coding=utf-8

"""
Module for creation of multi solid STLs.
"""

import numpy as np


def tri2stl(tri):
    """Get ogr triangle polygon in stl format."""
    p1, p2, p3 = np.array(tri.GetGeometryRef(0).GetPoints()[:3])
    
    # calculate face normal
    n = np.cross(p2 - p1, p3 - p1)
    tri_str = "  facet normal %.6e %.6e %.6e\n" % (n[0], n[1], n[2])
    tri_str += "    outer loop\n"
    tri_str += "      vertex %.6e %.6e %.6e\n" % (p1[0], p1[1], p1[2])
    tri_str += "      vertex %.6e %.6e %.6e\n" % (p2[0], p2[1], p2[2])
    tri_str += "      vertex %.6e %.6e %.6e\n" % (p3[0], p3[1], p3[2])
    tri_str += "    endloop\n"
    tri_str += "  endfacet\n"
    return tri_str


class MultiStl(object):
    """Container class for multiple STL solids."""

    def __init__(self, solids=None):
        """
        Initialize MultiSolid
        @param solids: a dictionary {name: [tri1, tri2, tri3, ..., triN]}
        """

        self.solids = solids or {}

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

    def write(self, filename):
        """Write STL to file in ascii-format."""
        with open(filename, 'w') as stl:
            for name, tin in self.solids.iteritems():
                stl.write('solid %s\n' % name)
                for tri in tin:
                    stl.write(tri2stl(tri))
                stl.write('endsolid %s' % name)
