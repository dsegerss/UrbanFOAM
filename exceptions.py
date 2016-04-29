# -*- coding: us-ascii -*-
"""Exceptions for the pyAirviro package."""

from __future__ import unicode_literals


class OutsideExtentError(Exception):

    """Error trying to get a value outside of raster dimensions."""

    def __init__(self, filename, message):
        super(OutsideExtentError, self).__init__(message)

