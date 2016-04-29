# -*- coding: us-ascii -*-
"""Exceptions for the UrbanFOAM package."""

from __future__ import unicode_literals

class SQLIntegrityError(Exception):

    """Violation of database integrity."""

    def __init__(self, err, record_id, message, *args, **kwargs):
        message = '{0} {1}, {2}'.format(
            err, message, record_id
        )
        super(SQLIntegrityError, self).__init__(
            message.format(*args, **kwargs)
        )


class SQLOperationalError(Exception):

    """Error in database operation."""

    def __init__(self, err, record_id, message, *args, **kwargs):

        message = '{0} {1}, {2}'.format(
            err, message, record_id
        )

        super(SQLOperationalError, self).__init__(
            message.format(*args, **kwargs)
        )


class CommandError(Exception):

    """Error running a subprocess."""

    def __init__(self, command, stdout, stderr, message, *args, **kwargs):
        msg = '{0}\nCommand: {1}'
        if len(stdout) > 0:
            msg += '\nStdout: {2}'
        if len(stderr) > 0:
            msg += '\nStderr: {3}'
        super(CommandError, self).__init__(
            msg.format(message, command, stdout, stderr, *args, **kwargs)
        )
        self.command = command
        self.stdout = stdout
        self.stderr = stderr


class OutOfBoundsError(Exception):

    """Out of range error."""


class ResourceError(Exception):

    """Error reading parameter from controlfile."""


class ValidationError(Exception):

    """Invalid parameters assigned to object."""

    def __init__(self, obj, fieldname, message, *args, **kwargs):
        message = "Value %s  for field %s of %s %s %s" % (
            obj._fields[fieldname].to_unicode(obj._fieldvalues[fieldname]),
            fieldname,
            obj.__class__.__name__,
            obj.primary_key,
            message)
        super(ValidationError, self).__init__(message, *args, **kwargs)

