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

roadClassification (
{road_classification}
);
"""

LANDUSE_DICT_TEMPLATE = \
"""
//Switch to include canopy effects for simulations (yes/no)
canopy          yes;

//Switch to read from raster
readFromRaster yes;

//Raster file name (if reading from raster)
rasterFileName "constant/landuse/LandUse.asc";

//Offset in east-west direction
subtractedX {subtractedX};

//offset in north-south direction
subtractedY {subtractedY};

//list of landuse codes, the properties are given in order as follows
//(landusecode Cd LAD-max fraction z0 Height 0 0 0)
//the paranthesis should always contain 9 values (even if only 6 are used)
//when maxLAD=-1, maxLAD is calculated from LAI and height distribution
landuseList
(  //Code Cd   LAI  Frac   z0   Height LAD-max
    (-1   0.2   0    0     0.1    0      0     0 0) // internal field val.
    ( 0   0.2   0    0     0.1    0     -1     0 0) // default
    ( 1   0.2   1    1     0.1    5     -1     0 0) //urban, low build.
    ( 2   0.2   3    1     0.05   7     -1     0 0) //low dense veg
    ( 3   0.2   1    1     0.05   5     -1     0 0) // low sparse veg
    ( 4   0.2   3    1     0.05   12     1.2   0 0) // high dense veg
    ( 5   0.2   0    0     0.0005 0      0     0 0) // open
    ( 6	  0.2	0    0	   0.1	  0	 0     0 0) // open asphalt
    ( 7	  0.2	0    0	   0.1	  0	 0     0 0) // city field
);

//Patches to set landuse for
sourcePatches
(
    wall.ground
);

// if no raster is given as an argument, the following code indices are used
// for the corresponding patch in the sourcePatches list
// Correspondance implied by position in the lists
patchLanduse
(
    0
    0
    0
    0
);

//11 points defining the leaf area density as a fraction of
//its maximum (which depends on LAI).
//The distance between the points is 0.1*height
heightDistribution
(	
    0.0	 //0
    0.1  //0.1
    0.2  //0.2
    0.5  //0.3
    0.8  //0.4
    1.0  //0.5
    0.9  //0.6
    0.7  //0.7
    0.4  //0.8
    0.2  //0.9
    0.0  //1.0
);				

// ************************************************************************* //
"""
