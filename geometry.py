# -*- coding: us-ascii -*-
"""Created on 18 nov 2009."""
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from scipy.interpolate import SmoothBivariateSpline
from osgeo import ogr

try:
    from pysqlite2 import dbapi2 as sqlite3
except ImportError:
    try:
        import sqlite3
    except:
        pass

import triangle

from UrbanFOAM.exceptions import OutsideExtentError

ANGLE_TOLERANCE = 1e-6
IN_TRIANGLE_TOLERANCE = 1e-8

HEIGHT_REF = {
    'ground': 0,
    'max ground elevation': 1,
    'absolute': 2
}

DEBUG_PLOTTING = False


# debug plot functions
if DEBUG_PLOTTING:
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as patches
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    plt.axes().set_aspect('equal', 'datalim')
    plt.ion()
    plt.show()

    def points_to_patch(points, **kwargs):
        """Debug plotting."""
        points2D = [(p[0], p[1]) for p in points]
        codes = [Path.MOVETO]
        for p in points[1:-1]:
            codes.append(Path.LINETO)
        if points[0] != points[-1]:
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)
        path = Path(points2D, codes)
        patch = patches.PathPatch(
            path,
            **kwargs
        )
        return patch

    def plot_point(point, *args, **kwargs):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(min(xmin, point[0] - 10), max(xmax, point[0] + 10))
        ax.set_ylim(min(ymin, point[1]) - 10, max(ymax, point[1] + 10))
        p = ax.plot(point[0], point[1], *args, **kwargs)
        plt.draw()
        return p

    def plot_points(points, *args, **kwargs):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(
            min(xmin, points[:, 0].min() - 10),
            max(xmax, points[:, 0].max() + 10)
        )
        ax.set_ylim(
            min(ymin, points[:, 1].min()) - 10,
            max(ymax, points[:, 1].max()) + 10
        )
        p = ax.plot(points[:, 0], points[:, 1], *args, **kwargs)
        plt.draw()
        return p

    def plot_text(point, text, *args, **kwargs):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(min(xmin, point[0] - 10), max(xmax, point[0] + 10))
        ax.set_ylim(min(ymin, point[1]) - 10, max(ymax, point[1] + 10))
        t = ax.text(point[0], point[1], text, *args, **kwargs)
        plt.draw()
        return t


def get_point_in_ring(ring):
    """Get any point inside LinearRing."""
    poly = ogr.ForceToPolygon(ring)
    centroid = poly.Centroid()
    if centroid.Within(poly):
        return centroid.GetPoint()[:2]

    p1 = np.array(ring.GetPoint(0))
    p2 = np.array(ring.GetPoint(1))
    
    segment = np.array([p2[0] - p1[0], p2[1] - p1[1], 0])

    normal = np.cross(
        segment,
        np.array([0, 0, 1])
    )

    midpoint = p1 + 0.5 * segment
    normal /= np.sqrt(normal.dot(normal))
    p = midpoint + 0.1 * normal
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(p[0], p[1])
    if point.Within(poly):
        return (p[0], p[1])
        
    p = midpoint - 0.1 * normal
    return (p[0], p[1])
        

def get_internal_angles(points):
    """Get internal angles of points in polygon"""
    segments = points - np.roll(points, 1, axis=0)
    v1 = segments
    v2 = np.roll(segments, -1, axis=0)
    
    # anti-clockwise winding gives positive angles for convex corners
    angles = np.pi - np.arctan2(
        v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0],
        v1[:, 0]*v2[:, 0] + v1[:, 1]*v2[:, 1]
    )
    return angles


def get_angle_between_segments2(p1, p2, p3):
    """Get angle at corner of segments (p1, p2) and (p2, p3).
    @param p1: point array
    @param p2: point array
    @param p3: point array
    """
    v1 = p2 - p1
    v2 = p3 - p2
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    #    angles = get_internal_angles(np.vstack((p3, p2, p1)))
    return angle


def is_clockwise(points):
    """Check winding direction for ring.
    return True if clockwise.
    """
    sum = 0.0
    if points[0] != points[-1]:
        closed_poly_points = points + [points[0]]
    else:
        closed_poly_points = points
    for i in range(len(closed_poly_points) - 1):
        x1, y1 = closed_poly_points[i][:2]
        x2, y2 = closed_poly_points[i + 1][:2]
        sum += (x2 - x1) * (y2 + y1)
    return sum > 0.0


def extrude_and_triangulate_patch(structure, terrain):
    """
    Extrude and triangulate patch structure.
    
    Can be used to describe depressions in the ground,
    typically useful to describe water with a step shoreline
    
    """
    # max ground elevation along building walls
    zmax = structure._get_max_ground_elevation(terrain)

    # set elevation of internal faces
    # depends on height_ref of structures
    tin = structure.triangulate()
    if structure.height_ref == HEIGHT_REF['ground']:
        # set elevation relative to ground
        tin = set_tin_elevation(
            tin,
            terrain=terrain,
            height=structure._to_height
        )
    elif structure.height_ref == HEIGHT_REF['max ground elevation']:
        # set elevation relative to max elevation
        # of ground within structure bounds
        zmax = structure._get_max_ground_elevation(terrain)
        tin = set_tin_elevation(
            tin,
            height=zmax + (structure._to_height or 0)
        )
    elif structure.height_ref == HEIGHT_REF['absolute']:
        # set elevation to absolute height
        # for patches, from_height and to_height is equal
        tin = set_tin_elevation(
            tin,
            height=structure._to_height or 0
        )

    # add walls
    for ring_index in range(structure._poly.GetGeometryCount()):
        points = structure._poly.GetGeometryRef(ring_index).GetPoints()
        for point_index in range(1, len(points)):
            x1, y1 = points[point_index - 1][:2]
            x2, y2 = points[point_index][:2]
            p1 = (x1, y1, terrain.sample(x1, y1))
            p2 = (x2, y2, terrain.sample(x2, y2))
            
            if structure.height_ref == HEIGHT_REF['ground']:
                p1_extruded = (x1, y1, p1[2] + structure._to_height)
                p2_extruded = (x2, y2, p2[2] + structure._to_height)
            elif structure.height_ref == HEIGHT_REF['max ground elevation']:
                p1_extruded = (x1, y1, zmax + structure._to_height)
                p2_extruded = (x2, y2, zmax + structure._to_height)
            elif structure.height_ref == HEIGHT_REF['absolute']:
                p1_extruded = (x1, y1, structure._to_height)
                p2_extruded = (x2, y2, structure._to_height)

            tin.append(make_triangle(p1, p2, p1_extruded))
            tin.append(make_triangle(p1_extruded, p2, p2_extruded))
    return tin


def set_tin_elevation(tin, terrain=None, height=None):
    """Set z-coords of tin nodes to elevation.
    @param terrain: a Terrain object,
    @param height: a fixed absolute height
    """
    if terrain is None and height is None:
        raise ValueError('Must specify either terrain and/or height')

    out_tin = []
    for i in range(len(tin)):
        tri = tin[i]
        p1, p2, p3 = tri.GetGeometryRef(0).GetPoints()[:3]
        z1 = z2 = z3 = 0
        if terrain is not None:
            z1 += terrain.sample(p1[0], p1[1])
            z2 += terrain.sample(p2[0], p2[1])
            z3 += terrain.sample(p3[0], p3[1])
        if height is not None:
            z1 += height
            z2 += height
            z3 += height
        p1 = (p1[0], p1[1], z1)
        p2 = (p2[0], p2[1], z2)
        p3 = (p3[0], p3[1], z3)
        out_tin.append(make_triangle(p1, p2, p3))
    return out_tin


def get_tin_envelope(tin):
    """Return xmin, xmax, ymin, ymax, zmin, zmax."""
    xmin = ymin = zmin = 999999999
    xmax = ymax = zmax = -99999999
    for tri in tin:
        x1, x2, y1, y2, z1, z2 = tri.GetEnvelope3D()
        xmin = min(xmin, x1)
        xmax = max(xmax, x2)
        ymin = min(ymin, y1)
        ymax = max(ymax, y2)
        zmin = min(zmin, z1)
        zmax = max(zmax, z2)
    return (xmin, xmax, ymin, ymax, zmin, zmax)


def buffer_polygon(poly, distance):
    """
    Buffer convex polygon outwards from centroid.
    poly: a ogr.wkbPolygon
    distance: buffer distance
    """
    centroid = geom_to_array(poly.Centroid())
    exterior = poly.GetGeometryRef(0)
    boundary = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    points = exterior.GetPoints()
    for i, xyz in enumerate(points):
        p = np.array(xyz)
        dir_vec = p[:2] - centroid[:2]
        dir_vec /= np.sqrt(dir_vec.dot(dir_vec))
        new_p = p + distance * dir_vec
        ring.AddPoint(*new_p)
    ring.CloseRings()
    boundary.AddGeometry(ring)
    return boundary


def translate_centreline(points, width, width_before=None, width_after=None):
    """Translate road points to (one) road-side.
    @param points: list of points along road,
    @param width: width of road, translates half this distance
    @param width_before: if connected at start, set to width of connected link
    @param width_after: if connected at end, set to width of connected link
    @returns: list of translated points
    """
    new_points = []
    p1 = points[0]
    p2 = points[1]

    # if no link connected before, make box at first node
    if width_before is None:
        # first point along road
        dir_vec = get_segment_dir(p1, p2)
        normal = get_segment_normal(dir_vec)
        new_points.append(p1)
        new_points.append(p1 + 0.5 * width * normal)
    else:
        # intersection point between centreline and link after
        new_points.append(points[1])

    npoints = len(points)
    for i in range(1, npoints - 1):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[i + 1]

        # get direction of segments before and after p2
        dir_vec1 = get_segment_dir(p1, p2)
        dir_vec2 = get_segment_dir(p2, p3)

        # get normal direction
        normal1 = get_segment_normal(dir_vec1)
        normal2 = get_segment_normal(dir_vec2)

        phi = get_angle_between_segments2(p1, p2, p3)
        # if road turns less than 5 degrees
        # use average width
        if abs(phi) < (5 * np.pi / 180):
            # take average normal direction for segment before and after
            new_points.append(p2 + 0.5 * width * 0.5 * (normal1 + normal2))
            continue

        # translated extended lines
        line1 = ogr.Geometry(ogr.wkbLineString)
        line2 = ogr.Geometry(ogr.wkbLineString)

        if i > 1 or width_before is None:
            line1p1 = p1 + normal1 * 0.5 * width - dir_vec1 * 20
            line1p2 = p2 + normal1 * 0.5 * width + dir_vec1 * 20
            line1.AddPoint(*line1p1.round(decimals=1))
            line1.AddPoint(*line1p2.round(decimals=1))
        else:
            line1p1 = p1 + normal1 * 0.5 * width_before - dir_vec1 * 20
            line1p2 = p2 + normal1 * 0.5 * width_before + dir_vec1 * 20
            line1.AddPoint(*line1p1.round(decimals=1))
            line1.AddPoint(*line1p2.round(decimals=1))

        if i < npoints - 2 or width_after is None:
            line2p1 = p2 + normal2 * 0.5 * width - dir_vec2 * 20
            line2p2 = p3 + normal2 * 0.5 * width + dir_vec2 * 20
            line2.AddPoint(*line2p1.round(decimals=1))
            line2.AddPoint(*line2p2.round(decimals=1))
        else:
            line2p1 = p2 + normal2 * 0.5 * width_after - dir_vec2 * 20
            line2p2 = p3 + normal2 * 0.5 * width_after + dir_vec2 * 20
            line2.AddPoint(*line2p1.round(decimals=1))
            line2.AddPoint(*line2p2.round(decimals=1))

        line1_points = np.vstack((line1p1, line1p2))
        line2_points = np.vstack((line2p1, line2p2))
        if DEBUG_PLOTTING:
            plot_points(line1_points, color='green')
            plot_points(line2_points, color='green')

        # find intersection of translated lines
        intersection = line1.Intersection(line2)
        intersection_point = np.array(
            intersection.GetPoint(0)
        ).round(decimals=1)
        if DEBUG_PLOTTING:
            plot_point(intersection_point, '+', markersize=20)
        new_points.append(intersection_point)

    # if no link connected after, make box at last node
    if width_after is None:
        # last point along road
        p1 = points[-2]
        p2 = points[-1]
        dir_vec = get_segment_dir(p1, p2)
        normal = get_segment_normal(dir_vec)
        new_points.append(p2 + 0.5 * width * normal)
        new_points.append(p2)
    else:
        # intersection point between centreline and link after
        new_points.append(points[-2])

    return np.array(new_points)


def buffer_road(points, width, links_before=None, links_after=None):
    """
    Buffer road at a right angle from centreline.
    @param points: points along a road centreline
    @param width: road width
    @param links_before: (id, width, points) to connected links before
    @param links_after: (id, width, points) to connected links after
    """
    road_buffer = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)

    if DEBUG_PLOTTING:
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        plot_points(points, '-*', color='black')

    if links_before is not None:
        # sort links before by angle to first line segment
        links_before.sort(
            key=lambda l: get_angle_between_segments2(
                l[2][1], l[2][0], points[1]
            )
        )

    if links_after is not None:
        # sort links after by angle to last line segment
        links_after.sort(
            key=lambda l: get_angle_between_segments2(
                points[-2],
                points[-1],
                l[2][1]
            )
        )

    if DEBUG_PLOTTING:
        texts = []
        for l in links_before:
            p = l[2][1]
            texts.append(
                plot_text(
                    p,
                    'b%3.0f' % (
                        get_angle_between_segments2(p, points[0], points[1])
                        / np.pi * 180
                    )
                )
            )

        for l in links_after:
            p = l[2][1]
            texts.append(
                plot_text(
                    p,
                    'a%3.0f' % (
                        get_angle_between_segments2(points[-2], points[-1], p)
                        / np.pi * 180
                    )
                )
            )

    # first side of road (going forward along road)
    # prepend points from segment of link before
    if links_before is not None and len(links_before) > 0:
        forward_points = np.vstack((links_before[0][2][1], points))
        width_before = links_before[0][1]
        if DEBUG_PLOTTING:
            plot_point(links_before[0][2][1], '*', markersize=20, color='blue')
    else:
        forward_points = points[:]
        width_before = None

    # append points from segment of link after
    if links_after is not None and len(links_after) > 0:
        forward_points = np.vstack((forward_points, links_after[0][2][1]))
        width_after = links_after[0][1]
        if DEBUG_PLOTTING:
            plot_point(links_after[0][2][1], '+', markersize=20, color='blue')
    else:
        width_after = None

    if DEBUG_PLOTTING:
        for t in texts:
            t.set_visible(False)

    trans_points1 = translate_centreline(
        forward_points, width, width_before, width_after
    )

    if DEBUG_PLOTTING:
        plot_points(trans_points1, '-*', color='red')

    # second side (going backwards along road)
    reversed_points = np.flipud(points)

    # prepend points from segment of link before
    if links_after is not None and len(links_after) > 0:
        reversed_points = np.vstack((links_after[-1][2][1], reversed_points))
        width_after = links_after[-1][1]
    else:
        width_after = None

    if links_before is not None and len(links_before) > 0:
        reversed_points = np.vstack((reversed_points, links_before[-1][2][1]))
        width_before = links_before[-1][1]
    else:
        width_before = None

    if DEBUG_PLOTTING:
        plot_points(trans_points1, 'o', color='black')

    trans_points2 = translate_centreline(
        reversed_points, width, width_after, width_before
    )
    
    if DEBUG_PLOTTING:
        plot_points(trans_points2, '-*', color='red')
    buffer_points = np.vstack((trans_points1, trans_points2))

    for p in buffer_points:
        ring.AddPoint(*p)
    ring.CloseRings()
    road_buffer.AddGeometry(ring)
    return road_buffer.ExportToIsoWkt()


def ring_to_line(ring):
    """Create a linestring from points in linearring."""
    points = ring.GetPoints()
    geom = ogr.Geometry(ogr.wkbLineString)
    for p in points:
        geom.AddPoint(p[0], p[1])
    return geom
        

def is_convex(a, b, c):
    """Check if 2D polygon node b is convex when traversing clockwise."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    crossp = (b[0] - a[0]) * (c[1] - a[1]) - \
             (b[1] - a[1]) * (c[0] - a[0])
    if crossp >= 0:
        return True
    return False


def get_xmost_vertex_index(points):
    """get index of x-most vertex."""
    npoints = len(points)
    max_point_index = 0
    xmax = points[0][0]
    for i in range(1, npoints):
        x = points[i][0]
        if x > xmax:
            max_point_index = i
            xmax = x
    return max_point_index


def get_mutually_visible_points(xpoints, ipoints):
    """
    Return ind of 2 mutually visible nodes in exterior and interior ring.
    @param xpoints: points of polygon exterior
    @param ipoints: points of interior ring
    """
    pM_ind = get_xmost_vertex_index(ipoints)
    pM = make_point(ipoints[pM_ind])
    
    # create ray in x-direction
    exterior_ring_xmax = xpoints[get_xmost_vertex_index(xpoints)][0]
    ray = ogr.Geometry(ogr.wkbLineString)
    ray.AddPoint(pM.GetX(), pM.GetY())
    ray.AddPoint(exterior_ring_xmax + 1, pM.GetY())

    # test ray intersection with all segments of exterior ring
    # find the intersection neareast to pM (the hole)
    min_dist = 99999999
    pI_seg_ind = None
    nxpoints = len(xpoints)
    for i in range(1, nxpoints):
        p1 = xpoints[i-1]
        p2 = xpoints[i]

        # check if segment is to the right of pM
        if max(p1[0], p2[0]) < pM.GetX():
            continue
        
        # check if ray intersects segment
        segment = ogr.Geometry(ogr.wkbLineString)
        segment.AddPoint(*p1)
        segment.AddPoint(*p2)
        
        pI_tmp = ray.Intersection(segment)
        if pI_tmp.GetGeometryName() == 'GEOMETRYCOLLECTION' and \
           pI_tmp.GetGeometryCount() == 0:
            continue

        dist = pM.Distance(pI_tmp)
        if dist < min_dist:
            min_dist = dist
            pI = pI_tmp
            pI_seg_ind = i  # segment index (xpoints[i-1] to xpoints[i])

    # if intersection is at vertex of exterior ring
    # pM and the point of the exterior ring are mutually visible
    p1 = xpoints[pI_seg_ind - 1]
    p2 = xpoints[pI_seg_ind]
    if pI.Equals(make_point(p1)):
        return (pI_seg_ind - 1, pM_ind)
    elif pI.Equals(make_point(p2)):
        return (pI_seg_ind, pM_ind)

    if p1[0] > p2[0]:
        pP = make_point(p1)
        pP_ind = pI_seg_ind - 1
    else:
        pP = make_point(p2)
        pP_ind = pI_seg_ind

    tri = make_triangle(pM, pI, pP)
    
    # Find any reflex points pA of the exterior ring within triangle pM, pI, pP
    # If such exist, use the one with smallest angle
    # phi (between line pM-pI and pM-pA)
    min_phi = 999
    pA_ind = None
    for i in range(nxpoints):
        p1 = xpoints[(i-1) % nxpoints]
        p2 = xpoints[i % nxpoints]
        p3 = xpoints[(i+1) % nxpoints]
        if i == pP_ind:
            continue
        p = ogr.Geometry(ogr.wkbPoint)
        p.AddPoint(*p2)

        if not is_convex(p1, p2, p3):
            if p.Within(tri):
                # vectors for lines pM-pI and pM-pA
                v_pM_pI = np.array(pI) - np.arrray(pM)
                v_pM_pA = np.arrray(p2) - np.array(pM)
                # normalize
                v_pM_pI = v_pM_pI / np.sqrt(v_pM_pI.dot(v_pM_pI))
                v_pM_pA = v_pM_pA / np.sqrt(v_pM_pA.dot(v_pM_pA))

                phi = np.arccos(v_pM_pI.dot(v_pM_pA))
                if phi < min_phi:
                    min_phi = phi
                    pA_ind = i
    if pA_ind is not None:
        return (pA_ind, pM_ind)
    else:
        return (pP_ind, pM_ind)


def to_segments(geom):
    """Return all segments of polygon."""
    segments = []
    if geom.GetGeometryName() == 'LINEARRING':
        points = geom.GetPoints()
        for i in range(1, len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            segment = ogr.Geometry(ogr.wkbLineString)
            segment.AddPoint(p1.GetX(), p1.GetY(), p1.GetZ())
            segment.AddPoint(p2.GetX(), p2.GetY(), p2.GetZ())
            segments.append(segment)
    return segments
    

def make_triangle(p1, p2, p3):
    """Create LinearRing for triangle."""
    tri = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in [p1, p2, p3]:
        if isinstance(p, ogr.Geometry):
            ring.AddPoint(p.GetX(), p.GetY(), p.GetZ())
        elif hasattr(p, '__iter__'):
            ring.AddPoint(*p[:3])
    ring.CloseRings()
    tri.AddGeometry(ring)
    return tri


def make_point(coords):
    """Create point geometry."""
    p = ogr.Geometry(ogr.wkbPoint)
    p.AddPoint(*coords)
    return p


def clip_ear2(points):
    npoints = len(points)
    if npoints < 3:
        return []
    if npoints == 3:
        p1 = points[0]
        p2 = points[1]
        p3 = points[2]
        tri = make_triangle(p1, p2, p3)
        
        if DEBUG_PLOTTING:
            patch = points_to_patch([p1, p2, p3, p1], alpha=0.5)
            ax.add_patch(patch)
            plt.draw()
        del points[:]
        return [tri]
    parray = np.array(points)[:, :2]
    angles = get_internal_angles(parray)
    
    if DEBUG_PLOTTING:
        for i, a in enumerate(angles):
            ax.text(parray[i, 0], parray[i, 1], '%.0f' % (a / np.pi * 180))
        plt.draw()
    # concave ears have angles <0, by setting to NaN,
    # these are sorted to the end
    sorted_ind = np.argsort(
        np.where(angles <= ANGLE_TOLERANCE, np.NaN, angles)
    )[: (angles > ANGLE_TOLERANCE).sum()]
    for i in sorted_ind:
        if angles[i] <= 0:
            break
        p1 = points[(i - 1) % npoints]
        p2 = points[i % npoints]
        p3 = points[(i + 1) % npoints]
        tri = make_triangle(p1, p2, p3)
        
        if DEBUG_PLOTTING:
            tmp_patch = points_to_patch([p1, p2, p3, p1], fill=False)
            ax.add_patch(tmp_patch)
            plt.draw()

        contains_point = points_in_triangle(parray, i).any()
        if not contains_point:
            if DEBUG_PLOTTING:
                tmp_patch.set_visible(False)
                patch = points_to_patch([p1, p2, p3, p1], alpha=0.5)
                ax.add_patch(patch)
                plt.draw()
            del points[i]
            return [tri]
        else:
            if DEBUG_PLOTTING:
                tmp_patch.set_visible(False)


def clip_ear(points):
    """Clip ear from polygon represented by list of points."""
    npoints = len(points)
    if npoints < 3:
        return []
    if npoints == 3:
        p1 = points[0]
        p2 = points[1]
        p3 = points[2]
        tri = make_triangle(p1, p2, p3)
        
        if DEBUG_PLOTTING:
            patch = points_to_patch([p1, p2, p3, p1], alpha=0.5)
            ax.add_patch(patch)
#            ax.text(p1[0], p1[1], 'P1')
#            ax.text(p2[0], p2[1], 'P2')
#            ax.text(p3[0], p3[1], 'P3')
            plt.draw()

        del points[:]
        return [tri]
    
    txt = []
    tris = {}
    for i in range(npoints):
        tritest = False
        p1 = points[(i-1) % npoints]
        p2 = points[i % npoints]
        p3 = points[(i+1) % npoints]
        
        tri = make_triangle(p1, p2, p3)
        tri.interior_angle = get_angle_between_segments(
            (p1, p2), (p2, p3)
        )
        
        for t in txt:
            t.set_visible(False)
        txt = []

        if tri.interior_angle > 0.0001 and is_convex(p1, p2, p3):
            # plot triangle candidate
            if DEBUG_PLOTTING:
                patch = points_to_patch([p1, p2, p3, p1], alpha=0.5)
                ax.add_patch(patch)
                txt.append(ax.text(p1[0], p1[1], 'P1'))
                txt.append(ax.text(p2[0], p2[1], 'P2'))
                txt.append(ax.text(p3[0], p3[1], 'P3'))
                plt.draw()

            for j in range(npoints):
                # if DEBUG_PLOTTING:
                #    ax.text(points[j][0], points[j][1], 't')
                #    plt.draw()

                if j in ((i-1) % npoints, i % npoints, (i+1) % npoints):
                    continue
                p = make_point(points[j])

                # check if any point within or touches ear candidate
                if p.Within(tri) or \
                   (p.GetPoint(0) not in (p1, p2, p3) and p.Touches(tri)):
                    tritest = True

                    # if DEBUG_PLOTTING:
                    #    patch.set_visible(False)
                    #    plt.draw()

                    break
            # no points intersecting ear candidate
            if tritest is False:
                if DEBUG_PLOTTING:
                    ax.text(
                        points[i % npoints][0],
                        points[i % npoints][1], '%.0f' % (
                            tri.interior_angle / 3.14 * 180
                        ),
                        size=14
                    )
                    plt.draw()
                    # patch.set_visible(False)

                tris[i % npoints] = tri
            else:
                if DEBUG_PLOTTING:
                    patch.set_visible(False)
                    plt.draw()
    
    min_angle_point_index = None
    min_angle = 99999
    for i, tri in tris.iteritems():
        if tri.interior_angle < min_angle:
            min_angle_point_index = i
            min_angle = tri.interior_angle

    del points[min_angle_point_index % npoints]
    return [tris[min_angle_point_index]]


def get_segment_coeffs(p1, p2):
    """Return (A, B, C) in Ax+By=C from segment between p1 and p2"""
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = A * p1[0] + B * p1[1]
    return (A, B, C)


def get_segment_from_coeffs(A, B, C):
    """Defines segment from koefficients by setting two points"""
    line = ogr.Geometry(ogr.wkbLineString())
    if A == 0:
        # line is parallell to x-axis
        line.AddPoint(0, C / B, 0)
        line.AddPoint(10, C / B, 0)

    elif B == 0:
        # line is parallell to y-axis
        line.AddPoint(C / A, 0, 0)
        line.AddPoint(C / A, 10, 0)
    else:
        line.AddPoint(C / A, 0, 0)
        line.AddPoint(0, C / B, 0)
    return line


def get_segment_normal(dirvec):
    """Return normal of 2D vector."""
    if dirvec.size == 3:
        return np.array([dirvec[1], -1 * dirvec[0], 0])
    else:
        return np.array([dirvec[1], -1 * dirvec[0]])


def get_segment_dir(p1, p2):
    """Normalized direction vector of segment."""
    if isinstance(p1, ogr.Geometry):
        direction = geom_to_array(p2) - geom_to_array(p1)
    else:
        direction = np.array(p2) - np.array(p1)
    return direction / np.sqrt(direction.dot(direction))


def get_angle_between_segments(s1, s2):
    """
    Calculate horizontal angle between segments s1, s2
    @param s1: segment given by tuple of coordinates ((x1, y1), (x2, y2))
    @param s2: segment given by tuple of coordinates ((x1, y1), (x2, y2))
    """
    p1, p2 = s1
    p3, p4 = s2
    # only use first 2 dimensions to horizontal evaluation
    v1 = np.array(p2[:2]) - np.array(p1[:2])
    v2 = np.array(p4[:2]) - np.array(p3[:2])
    phi = np.arccos(v1.dot(v2) / (np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2))))
    if phi == 0:
        phi = 180
    return phi


def points_in_triangle(points, i):
    """Return boolean array of points in triangle described by points:
    points[i - 1], points[i], points[i + 1]
    @param points: array of points
    @param i: index of triangle (ear)
    """
    barymetric_points = cartesian2barymetric(points, i)
    return barymetric_points.min(axis=1) > IN_TRIANGLE_TOLERANCE


def cartesian2barymetric(points, i):
    """Return barymetric coordinates of point in triangle described by points:
    points[i - 1], points[i], points[i + 1]
    @param points: array of points
    @param i: index of triangle (ear)
    """

    npoints = len(points)
    # triangle defined by points p0, p1, p2
    p0x, p0y = points[(i - 1) % npoints][:2]
    p1x, p1y = points[i][:2]
    p2x, p2y = points[(i + 1) % npoints][:2]
    
    area = 1/2 * (
        -1 * p1y * p2x +
        p0y * (-1 * p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y
    )

    s = 1 / (2 * area) * (
        p0y * p2x - p0x * p2y + (p2y - p0y) * points[:, 0] +
        (p0x - p2x) * points[:, 1]
    )
    t = 1 / (2 * area) * (
        p0x * p1y - p0y * p1x + (p0y - p1y) * points[:, 0] +
        (p1x - p0x) * points[:, 1]
    )
    u = 1 - s - t
    return np.array([s, t, u]).T


def get_closest_segment(line, point):
    """Get segment of line closest to point."""
    npoints = line.GetPointCount()
    segment = None
    min_dist = 99999999
    closest_segment = None
    for i in range(1, npoints):
        p1 = line.GetPoint(i - 1)
        p2 = line.GetPoint(i)
        segment = ogr.Geometry(ogr.wkbLineString)
        segment.AddPoint(p1.GetX(), p1.GetY(), p1.GetZ())
        segment.AddPoint(p2.GetX(), p2.GetY(), p2.GetZ())
        dist = line.Distance(segment)
        if dist < min_dist:
            min_dist = dist
            closest_segment = segment
    return closest_segment


def multipoly_to_polys(multipoly):
    """Convert multipolygno to list of polygons."""
    polys = []
    geom = multipoly.GetGeometryRef()
    if geom.GetGeometryName() == 'MULTIPOLYGON':
        for geom_part in geom:
            polys.append(geom_part)
    return polys


def geom_to_array(geom):
    """Return numpy array of point coordinates."""
    geom_type = geom.GetGeometryName()
    if geom_type == 'POINT':
        return np.array([geom.GetX(), geom.GetY(), geom.GetZ()])
    elif geom_type == 'LINESTRING':
        points = geom.GetPoints()
        return np.array(
            [[p.GetX(), p.GetY(), p.GetZ()] for p in points]
        )
    else:
        raise TypeError(
            'No array conversion implemented for ' +
            'geometry type %i' % geom.GeometryType()
        )


class Polygon(object):

    """Class adding some geometric functions to OGR wkbPolygon."""

    def __init__(self, polygon_wkb, id=None):
        self._poly = ogr.CreateGeometryFromWkb(polygon_wkb)
        if self._poly is None or not self._poly.IsValid():
            raise ValueError('Invalid geometry')
        geom_type = self._poly.GetGeometryName()
        if geom_type != 'POLYGON':
            raise TypeError(
                'Wrong geometry type %s, should be POLYGON' % geom_type
            )

        self.id = id
        if len(self.exterior_points()) == 0:
            raise ValueError(
                'Empty geometry not allowed in Polygon class'
            )

    @property
    def poly(self):
        return self._poly

    @poly.setter
    def poly(self, value):
        self._poly = value

    def exterior_ring(self):
        return self._poly.GetGeometryRef(0)

    def inner_rings(self):
        return [
            self._poly.GetGeometryRef(i)
            for i in range(1, self._poly.GetGeometryCount())
        ]

    def exterior_points(self):
        return self._poly.GetGeometryRef(0).GetPoints()

    # In order for STL to follow ground, more nodes are needed.
    # also for interior of polygon to follow ground, more nodes are needed
    # Check triangulation methods with better quality than ear clipping
    # Alternatives:
    #    - More advanced meshing/remeshing algorithms
    #    - The open Matlab mesher?
    #    - Tuning of ear-clipping algorith and adding internal points

    def get_max_segment_length(self):
        """Calculate maximum segment length of polygon."""

        max_segment_length = 0
        nrings = self._poly.GetGeometryCount()
        for ring in (self._poly.GetGeometryRef(i) for i in range(nrings)):
            points = ring.GetPoints()
            for j in range(1, len(points)):
                p1 = np.array(points[j - 1])
                p2 = np.array(points[j])
                diff = p2 - p1
                dist = np.sqrt(diff.dot(diff))
                max_segment_length = max(max_segment_length, dist)
        return max_segment_length
                
    def set_max_segment_length(self, length):
        """Set a minimum segment length by sub-dividing segments.
        @param length: minimum segment length
        """

        nrings = self._poly.GetGeometryCount()
        
        poly = ogr.Geometry(ogr.wkbPolygon)
        for i in range(nrings):
            new_ring = ogr.Geometry(ogr.wkbLinearRing)
            ring = self._poly.GetGeometryRef(i)
            points = ring.GetPoints()
            npoints = len(points)
            for j in range(1, npoints):
                p1 = np.array(points[j - 1])
                p2 = np.array(points[j])
                diff = p2 - p1
                dist = np.sqrt(diff.dot(diff))
                n_interp_points = int(np.ceil(dist / length) + 1)
                x = np.linspace(p1[0], p2[0], n_interp_points)
                y = np.linspace(p1[1], p2[1], n_interp_points)
                if len(p1) == 3:
                    z = np.linspace(p1[2], p2[2], n_interp_points)
                else:
                    z = np.zeros((n_interp_points,))
                for k in range(n_interp_points - 1):
                    new_ring.AddPoint(x[k], y[k], z[k])
                        
            new_ring.CloseRings()
            poly.AddGeometry(new_ring)

        self._poly = poly

    def self_intersects(self):
        segments = to_segments(self._poly)
        for i, s1 in enumerate(segments):
            for j, s2 in enumerate(segments):
                if i == j:
                    continue
                if s1.Crosses(s2):
                    return True
        return False

    def non_simple_nodes(self, precision=2):
        """Return nodes shared by more than two edges."""
        nrings = self._poly.GetGeometryCount()
        points = set()
        non_simple_nodes = []
        for i in range(nrings):
            ring = self._poly.GetGeometryRef(i)

            # last and first point are the same
            npoints = ring.GetPointCount() - 1

            for i in range(npoints):
                p = ring.GetPoint(i)
                # Scaling to represent as integers for exact comparison
                scale_factor = 10**precision
                node = (
                    int(p.GetX() * scale_factor),
                    int(p.GetY() * scale_factor)
                )
                if node in points:
                    non_simple_nodes.append(p)
                points.add(node)
        return non_simple_nodes

    def has_holes(self):
        """True if polygon has inner rings."""
        return self._poly.GetGeometryCount() > 1

    def is_simple(self):
        """
        Polygon is simple if:
        - it has no holes
        - it does not self-intersect
        - no vertex is shared by more than two edges
        """
        if self.has_holes():
            return False
        # when holes are cut open, nodes are duplicated...
        # if self.non_simple_nodes() != []:
        #    return False
        if self.self_intersects():
            return False
        return True

    def remove_holes(self):
        """Cut hole open and add to exterior ring."""
        # todo: handle mutipolygon domains

        exterior_points = self._poly.GetGeometryRef(0).GetPoints()
        nholes = self._poly.GetGeometryCount() - 1
        holes = [self._poly.GetGeometryRef(i) for i in range(1, nholes + 1)]

        if len(holes) == 0:
            return
        
        # sort holes accoring to xmax of hole envelope
        holes.sort(key=lambda r: r.GetEnvelope()[1])
        holes.reverse()

        for hole in holes:
            hole_points = hole.GetPoints()
            ext_p_ind, hole_p_ind = get_mutually_visible_points(
                exterior_points,
                hole_points
            )
            new_exterior = exterior_points[:ext_p_ind + 1]
            new_exterior += hole_points[hole_p_ind:-1]
            new_exterior += hole_points[:hole_p_ind + 1]
            new_exterior += exterior_points[ext_p_ind:]
            exterior_points = new_exterior
        
        # create ring from exterior points
        exterior_ring = ogr.Geometry(ogr.wkbLinearRing)
        for p in exterior_points:
            exterior_ring.AddPoint(*p)

        # update polygon using new exterior ring
        self._poly = ogr.Geometry(ogr.wkbPolygon)
        self._poly.AddGeometry(exterior_ring)

    def triangulate_ear_clipping(self):
        """Triangulate polygon, return list of triangles."""
        tin = []
        if self._poly.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(self._poly.GetGeometryCount()):
                p = self._poly.GetGeometryRef(i)
                poly = Polygon(p.ExportToWkb(), self.id)
                tin += poly.triangulate()
        else:
            ring = self._poly.GetGeometryRef(0)
            points = ring.GetPoints()[:-1]
            if is_clockwise(points):
                points.reverse()

            if DEBUG_PLOTTING:
                ax.clear()
                patch = points_to_patch(
                    points + [points[0]],
                    lw=3,
                    facecolor='none'
                )
                ax.add_patch(patch)

                xmin = min(points, key=lambda p: p[0])[0]
                xmax = max(points, key=lambda p: p[0])[0]
                margin = (xmax - xmin) * 0.05
                ymin = min(points, key=lambda p: p[1])[1]
                ymax = max(points, key=lambda p: p[1])[1]
                ax.set_xlim(xmin - margin, xmax + margin)
                ax.set_ylim(ymin - margin, ymax + margin)
                x, y, z = zip(*points)
                ax.plot(x, y, 'ro')
#                for i, p in enumerate(points):
#                    pass
#                    ax.text(p[0] + 1, p[1] + 1, '%i' % i)
    
                plt.draw()
            while len(points) > 0:
                tin += clip_ear2(points)
        return tin

    def triangulate(self):
        """Triangulate polygon, return list of triangles."""

        tin = []
        if self._poly.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(self._poly.GetGeometryCount()):
                p = self._poly.GetGeometryRef(i)
                poly = Polygon(p.ExportToWkb(), self.id)
                tin += poly.triangulate()
        else:
            triangle_tin = self.constrained_delaunay()
            for tri in triangle_tin['triangles']:
                v1, v2, v3 = triangle_tin['vertices'][tri]
                z1, z2, z3 = triangle_tin['elevations'][tri]
                tin.append(
                    make_triangle(
                        (v1[0], v1[1], z1),
                        (v2[0], v2[1], z2),
                        (v3[0], v3[1], z3)
                    )
                )
        return tin

    def PSLG(self, precision=2, max_segment_length=None, holes=True):
        pslg = {}

        if holes and self.has_holes():
            points_in_hole = [
                get_point_in_ring(ring)
                for ring in self.inner_rings()
            ]
            pslg['holes'] = np.array(points_in_hole)

        # subdivide edges of poly
        if max_segment_length is not None:
            self.set_max_segment_length(max_segment_length)
        else:
            max_segment_length = self.get_max_segment_length()

        # collect all points in a list
        vertices = []
        segments = []
        nrings = self._poly.GetGeometryCount()
        for i in range(nrings):
            ring = self._poly.GetGeometryRef(i)
            points = ring.GetPoints()
            if points[-1] == points[0]:
                vertices += points[:-1]
            else:
                vertices += points

        # array of polygon vertices
        vertices = np.vstack(vertices).round(decimals=precision)

        # duplicate vertices are removed
        # vertex equality is tested with given precision
        # for np.unique to work for multiple columns:
        # view rows as a single element
        dtype = vertices.dtype
        ncols = vertices.shape[1]
        vertices_rows = vertices.view(
            np.dtype((np.void, dtype.itemsize * ncols))
        )
        vertices = np.unique(vertices_rows)

        # find vertex indices for all segments
        for i in range(nrings):
            ring = self._poly.GetGeometryRef(i)
            points = ring.GetPoints()
            for i in range(1, len(points)):
                p1 = np.array(
                    points[i - 1]).round(
                        decimals=precision
                    ).view(dtype=vertices_rows.dtype)
                p2 = np.array(
                    points[i]).round(
                        decimals=precision
                    ).view(dtype=vertices_rows.dtype)
                i1 = np.where(vertices_rows == p1)
                i2 = np.where(vertices_rows == p2)

                segment = (
                    i1[0] if len(i1) == 1 else i1[0][0],
                    i2[0] if len(i2) == 1 else i2[0][0]
                )
                if segment not in segments:
                    segments.append(segment)

        # restore vertices from rows to individual coordinates
        vertices = vertices_rows.view(dtype).reshape(
            (vertices_rows.shape[0], ncols)
        )

        pslg['segments'] = np.vstack(segments)

        # to include internal points in tin
        # points are first laid out in a matrix covering the
        # whole extent of the polygon
        # all points outside of the polygon or very close to the
        # border are then removed
        borders = ogr.ForceToMultiLineString(self._poly)
        x1, x2, y1, y2 = self._poly.GetEnvelope()
        x_verts = np.linspace(
            x1 + max_segment_length,
            x2 - max_segment_length,
            (x2 - x1) / max_segment_length - 1
        )
        y_verts = np.linspace(
            y1 + max_segment_length,
            y2 - max_segment_length,
            (y2 - y1) / max_segment_length - 1
        )
        x_grid, y_grid = np.meshgrid(x_verts, y_verts)
        x_internal = x_grid.ravel()
        y_internal = y_grid.ravel()

        points_within_poly = []
        for i in range(x_internal.shape[0]):
            p = ogr.Geometry(ogr.wkbPoint)
            p.AddPoint(x_internal[i], y_internal[i])
            if p.Within(self._poly) and \
               borders.Distance(p) >= 0.5 * max_segment_length:
                points_within_poly.append(i)

        internal_points = np.vstack(
            (
                x_internal[points_within_poly],
                y_internal[points_within_poly]
            )
        )

        # elevation from polygon edges are interpolated to the
        # interior points

        if vertices.shape[1] == 2:
            vertices = np.hstack(
                (
                    vertices,
                    np.zeros((vertices.shape[0], 1), dtype=np.float32)
                )
            )
            
        try:
            interp_func = SmoothBivariateSpline(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2]
            )
        except:
            try:
                interp_func = SmoothBivariateSpline(
                    vertices[:, 0],
                    vertices[:, 1],
                    vertices[:, 2],
                    kx=1, ky=1
                )
            except:
                # too few vertices to create interp_func
                interp_func = None

        if interp_func is not None:
            internal_elevations = interp_func.ev(
                internal_points[0, :],
                internal_points[1, :]
            )
        else:
            internal_elevations = vertices[:, 2].mean()
        
        # Planar Straight Line Graph must be 2d
        # therefore elevations are returned separately
        pslg['vertices'] = np.vstack((vertices[:, 0:2], internal_points.T))
        return (
            pslg,
            np.hstack((vertices[:, 2], internal_elevations.flatten()))
        )

    def constrained_delaunay(self, precision=2, max_segment_length=None):
        pslg, elevations = self.PSLG(
            precision=precision, max_segment_length=max_segment_length
        )

        tin = triangle.triangulate(pslg, 'p')
        tin['elevations'] = elevations
        return tin


class Structure(Polygon):

    """Class representing structures as polygons with height attributes."""

    def __init__(self, polygon_wkb, id=None, height_ref=None, from_height=None,
                 to_height=None, patch=None):

        self.patch = patch
        self.height_ref = height_ref
        self._from_height = from_height
        self._to_height = to_height
        super(Structure, self).__init__(polygon_wkb, id=id)
        
    def _get_max_ground_elevation(self, terrain):
        """Get max ground elevation at building exterior points."""
        exterior_points = self._poly.GetGeometryRef(0).GetPoints()
        zmax = -9999999999
        for p in exterior_points:
            try:
                elevation = terrain.sample(p[0], p[1])
            except IndexError:
                # outside raster
                raise OutsideExtentError(
                    'Trying to get cell at index ' +
                    '(%i, %i) from terrain with dimensions (%i, %i)' % (
                        p[0], p[1], terrain.ny, terrain.nx)
                )
            zmax = max(zmax, elevation)
        return zmax

    def _get_avg_geom_elevation(self):
        """Calculate average elevation of exterior points. """
        exterior_points = self._poly.GetGeometryRef(0).GetPoints()
        heights = [p[2] for p in exterior_points]
        return sum(heights) / len(heights)

    def intersects(self, terrain, s2):
        """Check if structures intersects both vertically and horizontally."""

        if s2.poly.intersects(self.poly) and \
           s2.from_height(terrain) < self.to_height(terrain) and \
           s2.to_height(terrain) > self.from_height(terrain):
            return True
        else:
            return False

    def vertical_extent(self):
        """Return (zmin, zmax) of vertical elevation."""
        return (self.from_height(), self.to_height())

    def to_height(self, terrain, point):
        """Get to_height of building at TIN node.
        @param point: tuple or list of coordinates (x, y, z) of TIN node

        Only valid for nodes of TIN created from building contour.
        """
        if self._to_height is None:
            # using original height from node
            if len(point) != 3:
                raise ValueError(
                    'Structure %i does not ' % self.id +
                    'have z-coordinates and no to_height is specified'
                )
            return point[2]
        elif self.height_ref == HEIGHT_REF['ground']:
            return terrain.sample(
                point[0], point[1]) + self._to_height
        elif self.height_ref == HEIGHT_REF['max ground elevation']:
            return self._get_max_ground_elevation(
                terrain) + self._to_height
        elif self.height_ref == HEIGHT_REF['absolute']:
            return self._to_height

    def from_height(self, terrain, point):
        """Get from_height of building at TIN node.
        @param point: tuple or list of coordinates (x, y, z) of TIN node

        Only valid for nodes of TIN created from building contour.
        """
        if self._from_height is None:
            return terrain.sample(
                point[0], point[1])
        elif self.height_ref == HEIGHT_REF['ground']:
            return terrain.sample(
                point[0], point[1]) + self._from_height
        elif self.height_ref == HEIGHT_REF['max ground elevation']:
            return self._get_max_ground_elevation(
                terrain) + self._from_height
        elif self.height_ref == HEIGHT_REF['absolute']:
            return self._from_height

    def max_to_height(self, terrain):
        """Get avg elevation of structure roof.
        @param terrain: Terrain object

        If hight_ref is ground or height from geometry i used,
        the height is average for all vertices in polygon
        """
        if self._to_height is None:
            return self._get_max_geom_elevation()
        elif self.height_ref in (0, 1):
            return self._get_max_ground_elevation(terrain) + self._to_height
        elif self.height_ref == 2:
            return self._to_height
        else:
            raise ValueError('Height ref must be in 0, 1 or 2')

    def min_from_height(self, terrain):
        """Get avg elevation of structure floor.
        @param terrain: Terrain object

        If hight_ref is ground or no from_height is given,
        the height is average for all vertices in polygon
        """

        if self._from_height is None:
            return self._get_min_ground_elevation(terrain)
        elif self.height_ref in (0, 1):
            return self._get_min_ground_elevation(terrain) + self._from_height
        elif self.height_ref == 2:
            return self._from_height
        else:
            raise ValueError('Height ref must be in 0, 1 or 2')

    def in_vertical_extent(self, zmin, zmax):
        """Check if structure in vertical extent."""
        z1, z2 = self.vertical_extent()
        if z1 > zmax:
            return False
        elif z2 < zmin:
            return False

    def is_patch_only(self):
        """Return True if structure has 0 thickness,
        i.e. only contrains mesh at boundary.
        """
        if self._to_height == 0 and (
                self._from_height is None or
                self.height_ref == HEIGHT_REF['ground'] or
                self.height_ref == HEIGHT_REF['max ground elevation']):
            # elevation is set to zero relative to the ground
            return True
        elif (self._from_height is not None and
              self._from_height == self._to_height):
            # equal values are given to from_height and to_height
            return True
        else:
            return False

    def extrude(self, terrain):
        """Triangulate and extrude to 3D.
        @param terrain: a Terrain object
        """
        if terrain is None:
            raise ValueError('Must supply a Terrain before writing STL')

        if self._to_height == 0 and self._from_height:
            raise ValueError('Structure is patch-only and cannot be extruded')

        # only necessary to remove holes for ear clipping
        # self.remove_holes()

        # set elevation for interior faces
        internal_tin = self.triangulate()

        tin = []
        for tri in internal_tin:
            from_points = [
                list(p) for p in tri.GetGeometryRef(0).GetPoints()[:3]
            ]
            to_points = [
                list(p) for p in tri.GetGeometryRef(0).GetPoints()[:3]
            ]
            for p in from_points:
                p[2] = self.from_height(terrain, p)
            for p in to_points:
                p[2] = self.to_height(terrain, p)
            tin.append(make_triangle(*from_points))
            tin.append(make_triangle(*to_points))
        nrings = self._poly.GetGeometryCount()
        for i in range(nrings):
            ring = self._poly.GetGeometryRef(i)
            points = ring.GetPoints()
            for i in range(1, len(points)):
                p1, p2 = points[i - 1: i + 1]
                p1 = (p1[0], p1[1], self.from_height(terrain, p1))
                p2 = (p2[0], p2[1], self.from_height(terrain, p2))
                p1_extruded = (p1[0], p1[1], self.to_height(terrain, p1))
                p2_extruded = (p2[0], p2[1], self.to_height(terrain, p2))
                tin.append(make_triangle(p1, p2, p1_extruded))
                tin.append(make_triangle(p1_extruded, p2, p2_extruded))
        return tin


class Road(Structure):

    """A road surface and it's centreline."""
    
    def __init__(self, polygon_wkb, sourceid, id=None,
                 height_ref=None, from_height=None,
                 to_height=None, patch=None, **kwargs):
        """
        Create instance of Road
        
        @param polygon_wkb, road polygon as WKB,
        @param source id: id of road source,
        @param id: road polygon id,
        @param height_ref: height reference,
        @param from_height: road volume start height relative to height_ref
        @param to_height: road volume top relative to height_ref
        @param patch: name of patch in which to include road_id
        """
        self.sourceid = sourceid
        try:
            super(Road, self).__init__(
                polygon_wkb,
                id,
                height_ref=height_ref, from_height=from_height,
                to_height=to_height, patch=patch, **kwargs
            )
        except ValueError:
            raise ValueError(
                'Could not create road object from road polygon' +
                ' with id: %i: invalid geometry' % id
            )

    def get_centreline(self, edb_con):
        recs = edb_con.execute(
            """
            SELECT id, name, speed, ST_AsText(geom) as wkt, nolanes, vehicles
            FROM roads
            WHERE id = {source_id}
            """.format(source_id=self.sourceid)
        ).fetchall()
        
        if len(recs) == 0:
            raise ValueError('No road with id %i found i edb' % self.sourceid)
        points = ogr.CreateGeometryFromWkt(recs[0]['wkt']).GetPoints()

        centreline = {
            'id': self.sourceid,
            'name': recs[0]['name'],
            'speed': recs[0]['speed'],
            'nolanes': recs[0]['nolanes'],
            'vehicles': recs[0]['vehicles'],
            'points': '(' + ' '.join(
                ['(%f %f 0)' % (p[0], p[1]) for p in points]
            ) + ')'
        }

        return centreline


class Triangle(object):

    def __init__(self, *args, **kwargs):

        if len(args) == 3:
            p1, p2, p3 = args
            self._tri = ogr.Geometry(ogr.wkbLinearRing)
            for p in args:
                self._tri.AddPoint(p)
            self._tri.closeRings()

        elif len(args) == 1:
            self._tri = args[0]
        else:
            raise ValueError(
                'Must suppy either ogr.wkbLinearRing or 3 ogr.wkbPoint'
            )
    
    @property
    def p1(self):
        return self._tri.GetPoint(0)

    @property
    def p2(self):
        return self._tri.GetPoint(1)

    @property
    def p3(self):
        return self._tri.GetPoint(2)

    def toarray(self):
        """Return points as a numpy array."""
        return np.array(
            [
                [self.p1.GetX(), self.p1.GetY(), self.p1.GetZ()],
                [self.p2.GetX(), self.p2.GetY(), self.p2.GetZ()],
                [self.p3.GetX(), self.p3.GetY(), self.p3.GetZ()]
            ]
        )

    def set_precision(self, precision):
        points = self._tri.GetPoints()
        for p in points:
            p.SetX(round(p.GetX(), precision))
            p.SetY(round(p.GetY(), precision))
            p.SetoZ(round(p.GetZ(), precision))
        self._tri.setPoint(points)

    def normal(self):
        # vectors that defines the plane
        p1, p2, p3 = self.toarray()
        a = p2 - p1
        b = p3 - p1
        return np.cross(a, b)

    def circum_circle(self):
        """3 points give 2 bisectrices (right angle with the line between the points)
        if a line is given by Ax+By=C, then -Bx+Ay=D defines the bisectrice
        D found from mid-point of the 2 points of the line (x y independently)
        The intersection betwen the two bisectrices is found using
        the line.intersection method
        """
        p1, p2, p3 = self.toarray()
        
        midx = 0.5 * (p1[0] + p2[0])
        midy = 0.5 * (p1[1] + p2[1])
        A1, B1, C1 = get_segment_coeffs(p1, p2)
        # equation of bisectrice expressed in coefficients of l1
        D1 = -1 * B1 * midx + A1 * midy
        
        b1 = get_segment_from_coeffs(-1 * B1, A1, D1)
            
        midx = 0.5 * (p2[0] + p3[0])
        midy = 0.5 * (p2[1] + p3[1])
        A2, B2, C2 = get_segment_coeffs(p2, p3)
        D2 = -1 * B2 * midx + A2 * midy
    
        b2 = get_segment_from_coeffs(-1 * B2, A2, D2)
        
        centre = b1.Intersection(b2)
        p_centre = geom_to_array(centre)
        radius_vector = p_centre - p1
        radius = radius_vector.dot(radius_vector)
        return Circle(centre, radius)
        
    def circum_circle_radius(self):
        p1, p2, p3 = self.toarray()
        a = (p2 - p1).dot(p2 - p1)
        b = (p3 - p2).dot(p3 - p2)
        c = (p3 - p1).dot(p3 - p1)
        radius = a * b * c / np.sqrt(
            (a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)
        )
        return radius

    def to_stl(self):
        n = self.normal()
        p1, p2, p3 = self.toarray()
        t1_str = "  facet normal %.6e %.6e %.6e\n" % (n[0], n[1], n[2])
        t1_str += "    outer loop\n"
        t1_str += "      vertex %.6e %.6e %.6e\n" % (p1[0], p1[1], p1[2])
        t1_str += "      vertex %.6e %.6e %.6e\n" % (p2[0], p2[1], p2[2])
        t1_str += "      vertex %.6e %.6e %.6e\n" % (p3[0], p3[1], p3[2])
        t1_str += "    endloop\n"
        t1_str += "  endfacet\n"
        return t1_str


# class Circle(object):

#     def __init__(self, centre, radius):
#         self.centre = centre
#         self.radius = radius
        
#     def inCircle(self, p):
#         """Returns True if inside, None if on the border, False if outside"""
#         if (self.centre - p).mag() < self.radius:
#             return True
#         elif (self.centre - p).mag() == self.radius:
#             return None
#         else:
#             return False
        

# class RoadCentreLine(object):

#     def __init__(self, road_id, centreline_wkb, speed, nlanes, vehicles):
#         self.id = road_id
#         self.speed = speed
#         self.line = line
#         self.nlanes = nlanes
#         self.vehicles = vehicles

#     def __str__(self):
#         return str(self.points)
    
#     def __eq__(self, l):
#         if self.points == l.points:
#             return True
#         else:
#             return False

            
        

    # def get_speed_vector(self, point):
    #     """Get direction of line segment closest to point."""
    #     segment = get_closest_segment(self.line, point)
    #     if segment is None:
    #         raise ValueError(
    #             'Could not find segment in RoadCentreLine %i' % self.id +
    #             'closest to point (%f, %f)' % (point.GetX(), point.GetY())
    #         )
    #     return get_segment_dir(segment)

    # def createIndex(objects):
    #     newObjects=unique(objects)
    #     indices=[]
    #     for o in objects:
    #         indices.append(newObjects.index(o))
    #     return (newObjects,indices)
        

