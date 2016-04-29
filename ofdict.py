#!/usr/bin/env python
# -*- coding: us-ascii -*-
"""Resolve sources for emission calculations."""

from __future__ import unicode_literals
from __future__ import division


DICT_HEADER = \
"""
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.3                                   |
|   \\  /    A nd           | Web:      http://www.OpenFOAM.org               |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.3;
    format      ascii;
    class       dictionary;
    object      {filename};
}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

TRAFFIC_DICT_TEMPLATE = """
traffic yes;


//Average of the sides of the car bounding box
Acar 6.0;

//Cfcar according to wind tunnel experiments by Mochida 2008
Cfcar 0.64;

//Cecar according to wind tunnel experiments by Mochida 2008
Cecar 2.47;

L 10;

roadNames (
{road_names}
);

centreLines (
{centrelines}
);


//roadproperties are speed, nolanes, ADT
roadProperties (
{road_properties}

);

emissionGroupNames (
{emission_group_names}
);

roadClassification(
{road_classification}
);
"""
