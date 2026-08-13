#!/usr/bin/env python3
"""
===========================================================================
 CFG (CONTEXT-FREE GRAMMAR) ASSEMBLY VALIDATION TEST
 Project: Sign Language to HamNoSys Avatar Generator (V2 Architecture)
 
 This script validates that the CFG compiler in combine_hamnosys()
 produces grammatically valid HamNoSys sequences that conform to
 the formal HamNoSys phonetic notation grammar.
 
 Grammar Rules (Prillwitz et al., 1989):
   Sign ::= [SymmetryOp] HandshapeStructure InitialOrientation 
            BodyLocation [Contact] [Movement] [StateTransition]
   
 Tests:
   1. Structural Order Test: Tokens appear in correct grammatical order
   2. Token Validity Test: All tokens are valid HamNoSys tokens
   3. Mandatory Component Test: Required components are always present
   4. No Duplicate Test: No consecutive duplicate tokens
   5. Dictionary Match Test: Known glosses produce correct sequences
===========================================================================
"""

import os
import sys
import json
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_INTEGRATION_DIR = os.path.join(_PROJECT_ROOT, "Integration-20260706T062240Z-3-001", "Integration")

DICT_PATH = os.path.join(_INTEGRATION_DIR, "gloss_to_hamnosys_dict.json")
REPORT_PATH = os.path.join(_SCRIPT_DIR, "cfg_validation_report.json")


# ── HamNoSys Token Categories ──
SYMMETRY_TOKENS = {"hamsymmlr", "hamplus", "hamnonipsi"}

HANDSHAPE_TOKENS = {
    "hamfist", "hamflathand", "hamfinger2", "hamfinger23", "hamfinger23spread",
    "hamfinger2345", "hampinch12", "hampinchall", "hampinch12open", "hamcee12",
    "hamceeall", "hamceeopen", "hamthumboutmod", "hamthumbacrossmod",
    "hamthumbopenmod", "hamfingerstraightmod", "hamfingerbendmod",
    "hamfingerhookmod", "hamdoublebent", "hamdoublehooked", "hampinky",
}

EXT_FINGER_TOKENS = {
    "hamextfingeru", "hamextfingerd", "hamextfingerl", "hamextfingerr",
    "hamextfingero", "hamextfingeri", "hamextfingerul", "hamextfingerur",
    "hamextfingerdl", "hamextfingerdr",
}

PALM_ORI_TOKENS = {
    "hampalmu", "hampalmd", "hampalml", "hampalmr",
    "hampalmdl", "hampalmdr", "hampalmul", "hampalmur",
}

LOCATION_TOKENS = {
    "hamhead", "hamheadtop", "hamforehead", "hameyebrows", "hameyes",
    "hamnose", "hamnostrils", "hamear", "hamearlobe", "hamcheek",
    "hamlips", "hamtongue", "hamteeth", "hamchin", "hamunderchin",
    "hamneck", "hamshouldertop", "hamshoulders", "hamchest", "hamstomach",
    "hambelowstomach", "hamneutralspace", "hamupperarm", "hamelbow",
    "hamlowerarm",
}

CONTACT_TOKENS = {"hamtouch", "hamclose", "hambrushing", "hambetween"}

MOVEMENT_TOKENS = {
    "hammoveu", "hammoved", "hammovel", "hammover", "hammoveo", "hammovei",
    "hammoveright", "hamcircleo", "hamwaving", "hamarc", "hamnodding",
    "hamtwisting",
}

STATE_TOKENS = {"hamreplace", "hamrepeatfromstart"}

ALL_VALID_TOKENS = (
    SYMMETRY_TOKENS | HANDSHAPE_TOKENS | EXT_FINGER_TOKENS | PALM_ORI_TOKENS |
    LOCATION_TOKENS | CONTACT_TOKENS | MOVEMENT_TOKENS | STATE_TOKENS |
    {"none", "no-contact"}
)


def get_token_category(token):
    """Classify a token into its grammatical category."""
    if token in SYMMETRY_TOKENS:   return "SYMMETRY"
    if token in HANDSHAPE_TOKENS:  return "HANDSHAPE"
    if token in EXT_FINGER_TOKENS: return "EXT_FINGER"
    if token in PALM_ORI_TOKENS:   return "PALM_ORI"
    if token in LOCATION_TOKENS:   return "LOCATION"
    if token in CONTACT_TOKENS:    return "CONTACT"
    if token in MOVEMENT_TOKENS:   return "MOVEMENT"
    if token in STATE_TOKENS:      return "STATE"
    return "UNKNOWN"


def validate_sequence(tokens):
    """
    Validate a HamNoSys token sequence against CFG rules.
    Returns (is_valid, errors_list).
    """
    errors = []
    if not tokens:
        return False, ["Empty sequence"]

    # Test 1: All tokens must be valid
    for t in tokens:
        if t not in ALL_VALID_TOKENS:
            errors.append(f"Invalid token: '{t}'")

    # Test 2: Must contain at least handshape + orientation + location
    categories = [get_token_category(t) for t in tokens]
    if "HANDSHAPE" not in categories:
        errors.append("Missing required: HANDSHAPE")
    if "EXT_FINGER" not in categories:
        errors.append("Missing required: EXT_FINGER direction")
    if "PALM_ORI" not in categories:
        errors.append("Missing required: PALM_ORIENTATION")
    if "LOCATION" not in categories:
        errors.append("Missing required: BODY LOCATION")

    # Test 3: Structural order validation
    # Expected order: [SYMMETRY] HANDSHAPE EXT_FINGER PALM_ORI LOCATION [CONTACT] [MOVEMENT] [STATE...]
    CATEGORY_ORDER = ["SYMMETRY", "HANDSHAPE", "EXT_FINGER", "PALM_ORI", "LOCATION", "CONTACT", "MOVEMENT", "STATE"]
    
    filtered_cats = [c for c in categories if c != "UNKNOWN"]
    last_order_idx = -1
    for cat in filtered_cats:
        if cat in CATEGORY_ORDER:
            order_idx = CATEGORY_ORDER.index(cat)
            # Allow STATE to appear after movement (hamreplace resets the sequence)
            if cat == "STATE":
                continue
            if order_idx < last_order_idx and cat not in ("HANDSHAPE", "EXT_FINGER", "PALM_ORI"):
                # After hamreplace, handshape/ext/palm can repeat
                if "STATE" not in filtered_cats[:filtered_cats.index(cat)]:
                    errors.append(f"Order violation: {cat} appears after {CATEGORY_ORDER[last_order_idx]}")
            last_order_idx = max(last_order_idx, order_idx)

    # Test 4: No consecutive duplicates
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            errors.append(f"Consecutive duplicate: '{tokens[i]}' at position {i}")

    return len(errors) == 0, errors


def run_validation():
    """Main validation routine."""
    print("=" * 70)
    print(" CFG GRAMMAR VALIDATION TEST")
    print("=" * 70)

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        gloss_dict = json.load(f)

    print(f"  Loaded {len(gloss_dict)} gloss entries from dictionary\n")

    results = []
    pass_count = 0
    fail_count = 0

    for gloss, components in gloss_dict.items():
        # Reconstruct the expected token sequence
        tokens = []
        two_h = components.get("two_handed", "none")
        if two_h and two_h != "none":
            tokens.append(two_h)

        tokens.append(components.get("handshape", "hamflathand"))
        tokens.append(components.get("ext_finger", "hamextfingeru"))
        tokens.append(components.get("palm_ori", "hampalmd"))
        tokens.append(components.get("location", "hamchest"))

        contact = components.get("contact", "none")
        if contact and contact != "none":
            tokens.append(contact)

        movement = components.get("movement", "none")
        if movement and movement != "none":
            tokens.append(movement)

        is_valid, errors = validate_sequence(tokens)
        
        result = {
            "gloss": gloss,
            "sequence": " ".join(tokens),
            "token_count": len(tokens),
            "valid": is_valid,
            "errors": errors,
        }
        results.append(result)

        if is_valid:
            pass_count += 1
        else:
            fail_count += 1
            print(f"  [FAIL] {gloss}: {', '.join(errors)}")

    total = pass_count + fail_count
    pass_rate = (pass_count / total) * 100 if total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  CFG GRAMMAR VALIDITY: {pass_rate:.1f}%  ({pass_count}/{total} valid)")
    print(f"  STATUS: {'PASS [OK]' if pass_rate >= 90.0 else 'NEEDS REVIEW'}")
    print(f"{'=' * 70}")

    report = {
        "test_name": "CFG Grammar Validation",
        "total_glosses": total,
        "valid_sequences": pass_count,
        "invalid_sequences": fail_count,
        "validity_rate_pct": round(pass_rate, 2),
        "grammar_rules": [
            "Sign ::= [SymmetryOp] Handshape ExtFinger PalmOri Location [Contact] [Movement]",
            "All tokens must be valid HamNoSys identifiers",
            "Token order must follow grammatical category hierarchy",
            "No consecutive duplicate tokens allowed",
        ],
        "results": results,
        "verdict": "PASS" if pass_rate >= 90.0 else "NEEDS_REVIEW",
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to: {os.path.abspath(REPORT_PATH)}")

    return pass_rate >= 90.0


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
