# coding=utf-8

"""
Module to manipulate triangulations.
"""

from operator import methodcaller
from itertools import ifilterfalse, ifilter
from copy import deepcopy, copy

import numpy as np
from osgeo import ogr

from UrbanFOAM.geometry import is_convex

DEFAULT_PRECISION = 3
DEFAULT_SPATIAL_INDEX_BIN_SIZE = 1.0
DEBUG_PLOTTING = True


if DEBUG_PLOTTING:
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as patches

    def init_plot(verts, subplot=1):
        fig = plt.figure(2)
        ax = fig.add_subplot(220 + subplot)
        xmin = min(verts, key=lambda p: p[0])[0]
        xmax = max(verts, key=lambda p: p[0])[0]
        margin = (xmax - xmin) * 0.05
        ymin = min(verts, key=lambda p: p[1])[1]
        ymax = max(verts, key=lambda p: p[1])[1]
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        plt.ion()
        plt.show()
        return ax

    def points_to_path(verts, **kwargs):
        """Debug path plotting."""
        points2D = [(v.c[0], v.c[1]) for v in verts]
        codes = [Path.MOVETO]
        for p in verts[1:]:
            codes.append(Path.LINETO)
        path = Path(points2D, codes)
        patch = patches.PathPatch(
            path,
            **kwargs
        )
        return patch

    def points_to_patch(verts, **kwargs):
        """Debug patch plotting."""
        points2D = [(v.c[0], v.c[1]) for v in verts]
        codes = [Path.MOVETO]
        for v in verts[1:-1]:
            codes.append(Path.LINETO)
        if verts[0] != verts[-1]:
            codes.append(Path.LINETO)
        points2D.append(points2D[0])
        codes.append(Path.CLOSEPOLY)
        path = Path(points2D, codes)
        patch = patches.PathPatch(
            path,
            **kwargs
        )
        return patch


def tri2vertices(tri, precision=DEFAULT_PRECISION):
    """
    Return vertices representing raw triangle.
    @param tri: triangle as ogr.Polygon
    @returns: tuple of 3 Vertex objects

    """
    points = tri.GetGeometryRef(0).GetPoints()
            
    return (
        TinVertex(points[0][0], points[0][1], points[0][2]),
        TinVertex(points[1][0], points[1][1], points[1][2]),
        TinVertex(points[2][0], points[2][1], points[2][2])
    )


class SpatialIndex(object):

    def __init__(self, cellsize=DEFAULT_SPATIAL_INDEX_BIN_SIZE):

        self._index = {}
        self._xll = 0
        self._yll = 0
        self._zll = 0
        self.cellsize = cellsize

    def add(self, obj):
        """Add object to index and return object in index.
        @param obj: object to add to index

        If object already exist in index, the existing one is returned
        otherwise the new object is added to the index and then returned
        """
        objects = self._index.setdefault(self.index_key(obj), [])
        object_keys = [o.key for o in objects]
        try:
            ind = object_keys.index(obj.key)
            return objects[ind]
        except ValueError:
            objects.append(obj)
            return obj

    def index_key(self, obj):
        """Get index key for pos."""
        # bin index is calculated from (xll, yll, zll)
        x, y, z = obj.centroid()
        return (
            int((x - self._xll) / self.cellsize),
            int((y - self._yll) / self.cellsize),
            int((z - self._zll) / self.cellsize)
        )

    def __contains__(self, obj):
        """Check if object exist in index."""
        try:
            objects = self._index[self.index_key(obj)]
            return obj in objects
        except KeyError:
            return False

    def get_objects(self, bbox=None):
        """Generator function for all objects in bounding box.
        @param v1: lowest corner of bounding box
        @param v2: highest corner of bounding box
        """
        if bbox is not None:
            v1, v2 = bbox
            xi1, yi1, zi1 = self.index_key(v1)
            xi2, yi2, zi2 = self.index_key(v2)
            for xi in range(xi1, xi2 + 1):
                    for yi in range(yi1, yi2 + 1):
                        for zi in range(zi1, zi2):
                            for obj in self._index.get((xi, yi, zi), []):
                                yield obj
        else:
            for key, obj_list in self._index.iteritems():
                for obj in obj_list:
                    yield obj

    def remove(self, obj):
        key = self.index_key(obj)
        objects = self._index[key]
        objects.remove(obj)
        if len(objects) == 0:
            del(self._index[key])


class TinVertex(object):

    def __init__(self, x, y, z=0, precision=DEFAULT_PRECISION):
        self.c = np.array([x, y, z]).round(precision)
    
    @property
    def x(self):
        return self.c[0]

    @property
    def y(self):
        return self.c[1]

    @property
    def z(self):
        return self.c[2]

    @property
    def key(self):
        return (self.x, self.y, self.z)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.all(self.c == other.c)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, i):
        return self.c[i]

    def __setitem__(self, i, val):
        self.c[i] = val

    def centroid(self):
        return (self.x, self.y, self.z)


class TinEdge(object):

    def __init__(self, v1, v2):
        
        self.vertices = [v1, v2]
        self.triangles = dict()

    @property
    def v1(self):
        return self.vertices[0]

    @property
    def v2(self):
        return self.vertices[1]

    @v1.setter
    def v1(self, v):
        self.vertices[0] = v

    @v2.setter
    def v2(self, v):
        self.vertices[1] = v

    @property
    def key(self):
        """key representing edge vertices"""
        # lower left vertex should be first in key (sorting by x, y and then z)
        s = np.vstack((self.v1.c, self.v2.c))
        sorted_array = s[
            np.lexsort((s[:, 2], s[:, 1], s[:, 0]))
        ]
        return tuple(map(tuple, sorted_array.tolist()))

    def is_boundary(self):
        return len(self.triangles) == 1

    def add_triangle_ref(self, tri):
        self.triangles[tri.key] = tri
    
    def remove_triangle_ref(self, tri):
        del(self.triangles[tri.key])

    def __eq__(self, other):
        """Edge comparison (direction does not effect equality)"""
        if isinstance(other, self.__class__):
            return (
                (
                    np.all(self.v1.c == other.v1.c) and
                    np.all(self.v2.c == other.v2.c)
                ) or
                (
                    np.all(self.v1.c == other.v2.c) and
                    np.all(self.v2.c == other.v1.c)
                )
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # def __eq__(self, e):
    #     return self.v1 == e.v1 and self.v2 == e.v2

    # def __ne__(self, e):
    #     return not (self.v1 == e.v1 and self.v2 == e.v2)
    
    def centroid(self):
        p = self.v1.c + 0.5 * (self.v2.c - self.v1.c)
        return (p[0], p[1], p[2])

    def length(self):
        vec = self.v2.c - self.v1.c
        return np.sqrt(vec.dot(vec))


class TinTriangle(object):

    def __init__(self, e1, e2, e3):
        
        self.edges = [e1, e2, e3]

    @property
    def e1(self):
        return self.edges[0]

    @property
    def e2(self):
        return self.edges[1]

    @property
    def e3(self):
        return self.edges[2]

    @property
    def v1(self):
        return self.e1.v1

    @property
    def v2(self):
        return self.e1.v2

    @property
    def v3(self):
        # if two triangles references the same edge,
        # the edge will be flipped for one of them
        if self.e2.v2 != self.v2 and self.e2.v2 != self.v1:
            return self.e2.v2
        else:
            return self.e2.v1

    @e1.setter
    def e1(self, e):
        self.edges[0] = e

    @e2.setter
    def e2(self, e):
        self.edges[1] = e

    @e3.setter
    def e3(self, e):
        self.edges[2] = e

    @v1.setter
    def v1(self, v):
        self.e1.v1 = v
        self.e3.v2 = v

    @v2.setter
    def v2(self, v):
        self.e1.v2 = v
        self.e2.v1 = v

    @v3.setter
    def v3(self, v):
        self.e3.v1 = v
        self.e2.v2 = v

    @property
    def vertices(self):
        return [self.v1, self.v2, self.v3]

    @property
    def key(self):
        # lower left vertex should be first in key (sorting by x, y and then z)
        s = np.vstack((self.v1.c, self.v2.c, self.v3.c))
        sorted_array = s[
            np.lexsort((s[:, 2], s[:, 1], s[:, 0]))
        ]
        return tuple(map(tuple, sorted_array.tolist()))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key == other.key
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def asarray(self):
        """Return vertices as a numpy array."""
        return np.array((self.v1.c, self.v2.c, self.v3.c))

    def normal(self):
        """Return triangle normal."""
        # vectors that defines the plane
        p1, p2, p3 = self.asarray()
        a = p2 - p1
        b = p3 - p1
        return np.cross(a, b)
    
    def centroid(self):
        """Return centroid of triangle."""
        return tuple(self.asarray().mean(axis=0))

    def neighbours(self):
        for e in self.edges:
            for tri in e.triangles:
                if tri != self:
                    yield tri

    def remove_references(self):
        """Remove references to triangle from edges."""
        for e in self.edges:
            del(e.triangles[self.key])


class Tin(object):

    def __init__(self):
        self.vertices = SpatialIndex()
        self.edges = SpatialIndex()
        self.triangles = SpatialIndex()

    def __copy__(self):
        tin = Tin()
        for tri in self.triangles.get_objects():
            tin.add_triangle(tri.v1, tri.v2, tri.v3)
        return tin

    def __clean__(self):
        """Remove references to debug plotting object to allow deepcopy
        Matplotlib throws an error otherwise...
        """
        for e in self.edges.get_objects():
            if hasattr(e, 'patch'):
                del(e.patch)
        for t in self.triangles.get_objects():
            if hasattr(t, 'patch'):
                del(t.patch)

    def add_triangle(self, v1, v2, v3):
        """Add triangle to Tin.
        @param v1, v2, v3: vertices of triangle
        
        Make sure vertices are unique
        Ensure anti-clockwise winding order of triangle
        """

        # add vertices to spatial index
        v1 = self.vertices.add(v1)
        v2 = self.vertices.add(v2)
        v3 = self.vertices.add(v3)

        # add edges to spatial index
        if not is_convex(v1.c, v2.c, v3.c):
            e1 = self.edges.add(TinEdge(v1, v2))
            e2 = self.edges.add(TinEdge(v2, v3))
            e3 = self.edges.add(TinEdge(v3, v1))
        else:
            e1 = self.edges.add(TinEdge(v1, v3))
            e2 = self.edges.add(TinEdge(v3, v2))
            e3 = self.edges.add(TinEdge(v2, v1))

        # add triangle to spatial index
        new_tri = TinTriangle(e1, e2, e3)
        tri = self.triangles.add(TinTriangle(e1, e2, e3))
        
        e1.add_triangle_ref(tri)
        e2.add_triangle_ref(tri)
        e3.add_triangle_ref(tri)
        return tri

    def delete_triangle(self, t):
        """Remove all references to triangle.
        returns: list of left-over edges
        """
        self.triangles.remove(t)  # remove from spatial index
        t.remove_references()  # remove references from edges
        if hasattr(t, 'patch'):
            t.patch.set_visible(False)

        return [e for e in t.edges if len(e.triangles.values()) == 0]

    def delete_edge(self, edge):
        """Remove all references to edge.
        @param edge: edge to remove
        @returns: segments of hole in tin.
        """
        # remove from spatial index
        self.edges.remove(edge)

        # check if edge has reference to plot object
        # if so, hide
        if hasattr(edge, 'patch'):
            edge.patch.set_visible(False)

        # delete triangles using edge and find exposed edges
        exposed_edges = []
        tris_to_remove = []
        for key, tri in edge.triangles.iteritems():
            exposed_edges += [e for e in tri.edges if e != edge]
            tris_to_remove.append(tri)
        while len(tris_to_remove) > 0:
            self.delete_triangle(tris_to_remove.pop())
        return exposed_edges

    def refine(self, max_edge_length):
        tin1 = self
        new_tin = copy(tin1)
        added_triangles = 1
        refinement_level = 1
        while added_triangles > 0:
            if DEBUG_PLOTTING:
                tin1.__clean__()
                verts = map(lambda v: v.c, list(self.vertices.get_objects()))
                ax = init_plot(verts, subplot=refinement_level)
                for tri in tin1.triangles.get_objects():
                    try:
                        tri.patch = points_to_patch([tri.v1, tri.v2, tri.v3])
                    except AssertionError:
                        import pdb;pdb.set_trace()
                    ax.add_patch(tri.patch)
                    plt.draw()

            added_triangles = 0
            for tri in self.triangles.get_objects():
                if tri not in new_tin.triangles:
                    continue
                edges = sorted(
                    tri.edges, key=methodcaller('length'), reverse=True
                )
                internal_edges = [e for e in edges if not e.is_boundary()]
                long_internal_edges = ifilter(
                    lambda e: e.length() > max_edge_length, internal_edges
                )

                # longest edge will always be deleted
                try:
                    e1 = long_internal_edges.next()
                except StopIteration:
                    continue

                if DEBUG_PLOTTING:
                    e1.patch = points_to_path(
                        [e1.v1, e1.v2], color='r', linewidth=3
                    )
                    ax.add_patch(e1.patch)
                    plt.draw()

                # next edge to be deleted must not
                # create convex corners in hole
                e2 = None
                # for e2 in long_internal_edges:
                #     # ensure that
                #     if is_convex(e1.v1.c, e1.v2.c, e2.v2.c):
                #         break

                hole_segments = new_tin.delete_edge(e1)

                if DEBUG_PLOTTING:
                    verts = []
                    for e in hole_segments:
                        if e.v1 not in verts:
                            verts.append(e.v1)
                        elif e.v2 not in verts:
                            verts.append(e.v2)
                        
                    try:
                        patch_hole = points_to_patch(
                            verts, color='g', linewidth=3, alpha=0.2
                        )
                    except AssertionError:
                        import pdb;pdb.set_trace()
                    except IndexError:
                        import pdb;pdb.set_trace()

                    ax.add_patch(patch_hole)
                    plt.draw()

                if e2 is not None:
                    if DEBUG_PLOTTING:
                        patch = points_to_path(
                            [e2.v1, e2.v2], color='r', linewidth=3
                        )
                        ax.add_patch(patch)
                        plt.draw()

                    e2_ind = hole_segments.index(e2)
                    hole2_segments = new_tin.delete_edge(e2)
                    hole_segments = hole_segments[:e2_ind] + hole2_segments + \
                        hole_segments[e2_ind + 1:]

                hole_vertices = [hole_segments[0].v1.c, hole_segments[0].v2.c]
                for e in hole_segments[1:]:
                    if np.all(e.v2.c != hole_vertices[-1]):
                        hole_vertices.append(e.v2.c)
                    else:
                        hole_vertices.append(e.v1.c)
                # centroid = np.vstack(hole_vertices).mean(axis=0)

                hole = ogr.Geometry(ogr.wkbPolygon)
                ring = ogr.Geometry(ogr.wkbLinearRing)
                for v in hole_vertices:
                    ring.AddPoint(*v)
                ring.CloseRings()
                hole.AddGeometry(ring)
                new_vertex = TinVertex(*hole.Centroid().GetPoint())
#                new_vertex = TinVertex(*centroid)
                if DEBUG_PLOTTING:
                    ax.plot(
                        new_vertex.c[0], new_vertex.c[1], '*'
                    )
                    plt.draw()

                for e in hole_segments:
                    added_triangles += 1
                    new_tri = new_tin.add_triangle(
                        e.v1, e.v2, new_vertex
                    )
                    patch = points_to_patch(
                        [new_tri.v1, new_tri.v2, new_tri.v3],
                        linewidth=1,
                        facecolor='none'
                    )
                    ax.add_patch(patch)
                e1.patch.set_visible(False)
                patch_hole.set_visible(False)
                plt.draw()
            
            tin1 = copy(new_tin)
            refinement_level += 1
        
        if DEBUG_PLOTTING:
            ax = init_plot(verts, subplot=refinement_level)
            for tri in tin1.triangles.get_objects():
                tri.patch = points_to_patch([tri.v1, tri.v2, tri.v3])
                ax.add_patch(tri.patch)
                plt.draw()
        return new_tin

