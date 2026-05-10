"""
================================================================================
Traverse Bearing Adjustment
================================================================================

This module provides functionality to close misclosed traverses by adjusting
only the bearings while keeping the given distances constant. It is intended
for use when preparing adjusted traverse data for plotting in AutoCAD.

The adjustment is performed via matrix operations (varying rows and columns)
using NumPy and related numerical libraries. The specific adjustment mathematics
will be implemented on top of this foundation.

CAPABILITIES (planned):
----------------------
- Accept raw traverse data (bearings, distances)
- Form and solve the adjustment system via matrix operations
- Output adjusted bearings with original distances unchanged
- Support traverses of arbitrary size (variable matrix dimensions)

Author: SurvyAI Team
License: MIT
================================================================================
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Any, Sequence

import numpy as np
from numpy.typing import NDArray

# Optional: scipy for advanced linear algebra (e.g. least-squares, matrix solve)
try:
    import scipy.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# -----------------------------------------------------------------------------
# Traverse data from AI agent (misclosed traverse)
# -----------------------------------------------------------------------------
# Start coordinate: Easting = E, Northing = N
# Leg i: distance dist_i, bearing α_i
# Departure (change in Easting): D_i = dist_i * sin(α_i)
# Latitude (change in Northing):  L_i = dist_i * cos(α_i)
# Bearings are in decimal degrees (0–360 or -180–180); converted to radians internally.
# -----------------------------------------------------------------------------


@dataclass
class TraverseFromAgent:
    """
    Traverse data as supplied by the AI agent for a misclosed traverse.
    Supports a variable number of legs n; bearings α1..αn and distances d1..dn.
    Departures D_i and latitudes L_i are computed from D_i = dist_i*sin(α_i), L_i = dist_i*cos(α_i).
    """

    E: float  # Start Easting
    N: float  # Start Northing
    bearings: NDArray[np.floating]  # α1, α2, ..., αn (decimal degrees)
    distances: NDArray[np.floating]  # dist1, dist2, ..., distn
    departures: NDArray[np.floating]  # D1, D2, ..., Dn  (D_i = d_i * sin(α_i))
    latitudes: NDArray[np.floating]   # L1, L2, ..., Ln  (L_i = d_i * cos(α_i))
    n_legs: int  # number of traverse legs

    @classmethod
    def from_agent(
        cls,
        start_easting: float,
        start_northing: float,
        bearings: Union[Sequence[float], NDArray[np.floating]],
        distances: Union[Sequence[float], NDArray[np.floating]],
        bearings_in_degrees: bool = True,
    ) -> "TraverseFromAgent":
        """
        Build traverse data from parameters provided by the AI agent.

        Parameters
        ----------
        start_easting : float
            Start coordinate Easting (E).
        start_northing : float
            Start coordinate Northing (N).
        bearings : sequence of float
            Bearings α1, α2, ..., αn for legs 1..n (one per leg).
        distances : sequence of float
            Distances dist1, dist2, ..., distn for legs 1..n (one per leg).
        bearings_in_degrees : bool, default True
            If True, bearings are in decimal degrees; converted to radians for D, L.

        Returns
        -------
        TraverseFromAgent
            Instance with E, N, bearings, distances, and computed departures (D)
            and latitudes (L): D_i = dist_i*sin(α_i), L_i = dist_i*cos(α_i).

        Raises
        ------
        ValueError
            If lengths of bearings and distances differ.
        """
        b = np.asarray(bearings, dtype=float)
        dist = np.asarray(distances, dtype=float)
        if b.size != dist.size:
            raise ValueError(
                f"Bearings and distances must have the same length; got {b.size} and {dist.size}"
            )
        n_legs = len(b)
        if n_legs == 0:
            raise ValueError("At least one traverse leg is required.")

        if bearings_in_degrees:
            alpha_rad = np.deg2rad(b)
        else:
            alpha_rad = b.copy()

        # D_i = dist_i * sin(α_i),  L_i = dist_i * cos(α_i)
        departures = dist * np.sin(alpha_rad)
        latitudes = dist * np.cos(alpha_rad)

        return cls(
            E=float(start_easting),
            N=float(start_northing),
            bearings=b,
            distances=dist,
            departures=departures,
            latitudes=latitudes,
            n_legs=n_legs,
        )


def traverse_from_agent(
    start_easting: float,
    start_northing: float,
    bearings: Union[List[float], Tuple[float, ...], NDArray[np.floating]],
    distances: Union[List[float], Tuple[float, ...], NDArray[np.floating]],
    bearings_in_degrees: bool = True,
) -> TraverseFromAgent:
    """
    Convenience function: build traverse from AI agent parameters.

    Parameters
    ----------
    start_easting : float
        E (start Easting).
    start_northing : float
        N (start Northing).
    bearings : list, tuple or array
        α1, α2, ..., αn (one per leg).
    distances : list, tuple or array
        dist1, dist2, ..., distn (one per leg).
    bearings_in_degrees : bool, default True
        If True, bearings are in decimal degrees.

    Returns
    -------
    TraverseFromAgent
        Traverse with computed departures D and latitudes L.
    """
    return TraverseFromAgent.from_agent(
        start_easting=start_easting,
        start_northing=start_northing,
        bearings=bearings,
        distances=distances,
        bearings_in_degrees=bearings_in_degrees,
    )

def form_matrices(traverse: TraverseFromAgent) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Form the matrices for the adjustment system.
    """
    n_legs = traverse.n_legs
    b_deg = traverse.bearings
    dist = traverse.distances
    D = traverse.departures
    L = traverse.latitudes

    # Ensure we do trig in radians (bearings are supplied in decimal degrees)
    b = np.deg2rad(b_deg)
    # Sum of Departures
    sumD = float(np.sum(D))
    sumL = float(np.sum(L))
    # Formation of the A matrix (2 x n):
    # dD/db = dist * cos(bearing), dL/db = -dist * sin(bearing)
    dDb = (dist * np.cos(b)).reshape(1, n_legs)
    dLb = (-(dist * np.sin(b))).reshape(1, n_legs)
    A = np.vstack([dDb, dLb])  # shape: (2, n_legs)

    # Solve least-squares correction for bearings:
    # A * db + w = 0  ->  A * db = -w, where w is misclosure vector [sumD, sumL]^T
    # Using minimum-norm solution: db = A^T (A A^T)^-1 (-w)
    AT = A.T  # (n x 2)
    AAT = A @ AT  # (2 x 2)
    IAAT = np.linalg.inv(AAT)
    IA = AT @ IAAT  # (n x 2)

    w = np.array([[sumD], [sumL]], dtype=float)  # (2 x 1)
    db = -(IA @ w)  # (n x 1) delta bearing in radians
    # Formation of the adjusted bearings
    ''' Recall that
        dDb = dist * cos(bearing)
        dLb = -dist * sin(bearing)
        dD = dist * cos(bearing) * db
        dL = -(dist * sin(bearing) * db)
        and Da =  D + dD; La = L + dL
        Da = D + (dist * cos(bearing) * db)
        La = L + (-dist * sin(bearing) * db)

        Da = dist * sin(ba) where ba is the adjusted bearing
        La = dist * cos(ba) where ba is the adjusted bearing
    '''
    # Adjusted bearings (radians). For small corrections, ba ≈ b + db.
    ba = b + db.reshape(n_legs)

    #print("ba_rad=", ba)
    #print("db_rad=", db.reshape(n_legs))
    return ba
    # ba = np.array([[np.sin(b[i]) + (np.cos(b[i]) * db[0, i]) for i in range(n_legs)]])
    # adjusted_bearings = np.dot(A_T_A_inv_A_T, D)

    # ba = np.array([[np.sin(b[i]) + (np.cos(b[i]) * db[0, i]) for i in range(n_legs)]])
    # adjusted_bearings = np.dot(A_T_A_inv_A_T, D)