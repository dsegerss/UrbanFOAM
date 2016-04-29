# import qgis libs so that ve set the correct sip api version
import qgis   # pylint: disable=W0611  # NOQA

from matplotlib.path import Path
import matplotlib.patches as patches


def points_to_patch(points, **kwargs):
    """ """
    points2D = [(p[0], p[1]) for p in points]
    codes = [Path.MOVETO]
    for p in points[1:-1]:
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(points2D, codes)
    patch = patches.PathPatch(
        path,
        **kwargs
    )
    return patch

