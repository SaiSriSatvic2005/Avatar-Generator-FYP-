#!/usr/bin/env python
# coding: utf-8
"""
Dual Hand Classifier for HamNoSys Two-Handed Sign Structure.
Classifies two-handed signs into:
  - hamsymmlr:   Both hands mirror each other symmetrically
  - hamplus:     Both hands active but doing different things (asymmetric)
  - hamnonipsi:  One hand stationary as reference/base, other moves
  - none:        Only one hand active (one-handed sign)
"""

import numpy as np
from collections import Counter


def _to_vec(p):
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def _wrist_trajectory(frames, hand_key):
    """Extract wrist (landmark 0) trajectory for a specific hand."""
    traj = []
    for f in frames:
        hand = f.get(hand_key)
        if hand and hasattr(hand, "landmark"):
            traj.append(_to_vec(hand.landmark[0]))
        else:
            traj.append(None)
    return traj


def _interpolate_gaps(traj):
    """Fill None gaps with linear interpolation."""
    result = list(traj)
    valid_indices = [i for i, v in enumerate(result) if v is not None]
    
    if len(valid_indices) < 2:
        return result
    
    for i in range(len(result)):
        if result[i] is None:
            # Find nearest valid before and after
            before = max([j for j in valid_indices if j < i], default=valid_indices[0])
            after = min([j for j in valid_indices if j > i], default=valid_indices[-1])
            
            if before == after:
                result[i] = result[before]
            else:
                t = (i - before) / (after - before)
                result[i] = result[before] * (1 - t) + result[after] * t
    
    return result


def _trajectory_variance(traj):
    """Compute total variance of a trajectory (measures how much movement)."""
    valid = [v for v in traj if v is not None]
    if len(valid) < 3:
        return 0.0
    arr = np.array(valid)
    return np.var(arr, axis=0).sum()


def _displacement_correlation(left_traj, right_traj):
    """
    Compute correlation between left and right hand displacement vectors.
    High positive correlation → symmetric movement (hands move same direction)
    High negative correlation → mirror movement (hands move opposite in X, same in Y)
    Low correlation → independent movement
    """
    # Get valid paired frames
    valid_pairs = []
    for l, r in zip(left_traj, right_traj):
        if l is not None and r is not None:
            valid_pairs.append((l, r))
    
    if len(valid_pairs) < 5:
        return 0.0, 0.0  # y_corr, x_corr
    
    l_arr = np.array([p[0] for p in valid_pairs])
    r_arr = np.array([p[1] for p in valid_pairs])
    
    # Compute frame-to-frame displacement
    l_disp = np.diff(l_arr, axis=0)
    r_disp = np.diff(r_arr, axis=0)
    
    if len(l_disp) < 3:
        return 0.0, 0.0
    
    # Y-axis correlation (vertical): same direction = symmetric
    y_corr = np.corrcoef(l_disp[:, 1], r_disp[:, 1])[0, 1] if np.std(l_disp[:, 1]) > 1e-6 else 0.0
    
    # X-axis correlation (horizontal): opposite direction = mirror-symmetric
    x_corr = np.corrcoef(l_disp[:, 0], r_disp[:, 0])[0, 1] if np.std(l_disp[:, 0]) > 1e-6 else 0.0
    
    # Handle NaN
    y_corr = 0.0 if np.isnan(y_corr) else y_corr
    x_corr = 0.0 if np.isnan(x_corr) else x_corr
    
    return y_corr, x_corr


def _hand_is_stationary(traj, threshold=0.001):
    """Check if a hand's trajectory has very low variance (acting as a base/anchor)."""
    return _trajectory_variance(traj) < threshold


def classify_two_handed_structure(frames, right_handshapes=None, left_handshapes=None):
    """
    Classify the two-handed structure of a sign.
    
    Args:
        frames: List of frame_info dicts from shared_landmarks
        right_handshapes: Optional list of right hand shape labels per frame
        left_handshapes: Optional list of left hand shape labels per frame
    
    Returns:
        dict with:
            - "label": one of "hamsymmlr", "hamplus", "hamnonipsi", "none"
            - "confidence": float 0-1
            - "details": dict with analysis details
    """
    right_traj = _wrist_trajectory(frames, "right_hand")
    left_traj = _wrist_trajectory(frames, "left_hand")
    
    # Count valid detections
    right_valid = sum(1 for v in right_traj if v is not None)
    left_valid = sum(1 for v in left_traj if v is not None)
    total = max(len(frames), 1)
    
    right_ratio = right_valid / total
    left_ratio = left_valid / total
    
    details = {
        "right_detection_ratio": round(right_ratio, 2),
        "left_detection_ratio": round(left_ratio, 2),
    }
    
    # ONE-HANDED: one hand barely detected
    if right_ratio < 0.15 or left_ratio < 0.15:
        return {"label": "none", "confidence": 0.9, "details": details}
    
    # Interpolate gaps for trajectory analysis
    right_filled = _interpolate_gaps(right_traj)
    left_filled = _interpolate_gaps(left_traj)
    
    right_var = _trajectory_variance(right_filled)
    left_var = _trajectory_variance(left_filled)
    
    details["right_variance"] = round(right_var, 6)
    details["left_variance"] = round(left_var, 6)
    
    # NON-DOMINANT IPSILATERAL: one hand is stationary (base hand)
    stationary_threshold = max(right_var, left_var) * 0.1  # relative threshold
    stationary_threshold = max(stationary_threshold, 0.0005)  # minimum absolute
    
    if right_var < stationary_threshold and left_var > stationary_threshold * 5:
        details["stationary_hand"] = "right"
        return {"label": "hamnonipsi", "confidence": 0.8, "details": details}
    
    if left_var < stationary_threshold and right_var > stationary_threshold * 5:
        details["stationary_hand"] = "left"
        return {"label": "hamnonipsi", "confidence": 0.8, "details": details}
    
    # Both hands are active — check symmetry
    y_corr, x_corr = _displacement_correlation(left_filled, right_filled)
    details["y_correlation"] = round(y_corr, 3)
    details["x_correlation"] = round(x_corr, 3)
    
    # SYMMETRIC (hamsymmlr): vertical movement correlated, horizontal may be mirrored
    # In mirror-symmetric signs, Y moves the same direction, X moves opposite
    is_symmetric_motion = (y_corr > 0.4) or (y_corr > 0.2 and x_corr < -0.2)
    
    # Also check handshape similarity
    handshapes_match = True
    if right_handshapes and left_handshapes:
        right_hs_valid = [h for h in right_handshapes if h is not None]
        left_hs_valid = [h for h in left_handshapes if h is not None]
        
        if right_hs_valid and left_hs_valid:
            right_dominant = Counter(right_hs_valid).most_common(1)[0][0]
            left_dominant = Counter(left_hs_valid).most_common(1)[0][0]
            handshapes_match = (right_dominant == left_dominant)
            details["right_handshape"] = right_dominant
            details["left_handshape"] = left_dominant
    
    if is_symmetric_motion and handshapes_match:
        confidence = min(0.9, 0.5 + abs(y_corr) * 0.3 + (0.2 if handshapes_match else 0))
        return {"label": "hamsymmlr", "confidence": round(confidence, 2), "details": details}
    
    # ASYMMETRIC (hamplus): both active but different patterns
    confidence = min(0.85, 0.5 + (1 - abs(y_corr)) * 0.2 + (0.15 if not handshapes_match else 0))
    return {"label": "hamplus", "confidence": round(confidence, 2), "details": details}
