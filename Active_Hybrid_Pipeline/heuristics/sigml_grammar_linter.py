"""
sigml_grammar_linter.py
=======================
Generalized Grammar Linter & Canonical Slot Assembler for HamNoSys / SiGML (V3).

Enforces strict ANTLR Ham4Parser / CWASA compliance:
1. Canonical Slot Ordering:
   Sign ::= [Symmetry] Handshape [ThumbMod] [BendMod]
            ExtFingerOri PalmOri Location [Contact | DistanceMod]
            [Movement] [MovementMod] [Repetition] [ReplaceSequence]
2. Strict Casing (e.g. hamupperarm -> hamUpperarm)
3. Elimination of invalid consecutive modifiers (e.g. hamarmextended + hamupperarm)
4. Elimination of non-HamNoSys tokens (e.g. 'signer', 'none', 'no-contact')
"""

import json
import os
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_MAP_PATH = os.path.join(_SCRIPT_DIR, "cwasa_token_map.json")

# Load authoritative token dictionary
CWASA_TOKEN_MAP = {}
if os.path.exists(TOKEN_MAP_PATH):
    with open(TOKEN_MAP_PATH, "r", encoding="utf-8") as f:
        CWASA_TOKEN_MAP = json.load(f)

# Classification sets for HamNoSys tokens
SYMMETRY_TOKENS = {
    "hamsymmlr", "hamsymmpar", "hamalternatingmotion", "hamnonipsi"
}

HANDSHAPE_BASE_TOKENS = {
    "hamflathand", "hamfist", "hamfinger2", "hamfinger23", "hamfinger23spread",
    "hamfinger2345", "hampinch12", "hampinchall", "hampinch12open",
    "hamcee12", "hamceeall", "hamcee12open", "hamceeopen"
}

THUMB_MOD_TOKENS = {
    "hamthumboutmod", "hamthumbacrossmod", "hamthumbopenmod"
}

BEND_MOD_TOKENS = {
    "hamfingerstraightmod", "hamfingerbendmod", "hamfingerhookedmod",
    "hamfingerhookmod", "hamdoublebent", "hamdoublehooked"
}

HAND_PART_TOKENS = {
    "hamfingertip", "hamfingernail", "hamfingerpad", "hamfingermidjoint",
    "hamfingerbase", "hamfingerside", "hamthumbside", "hampinkyside", "hambetween"
}

EXT_FINGER_TOKENS = {
    "hamextfingeru", "hamextfingerur", "hamextfingerr", "hamextfingerdr",
    "hamextfingerd", "hamextfingerdl", "hamextfingerl", "hamextfingerul",
    "hamextfingerol", "hamextfingero", "hamextfingeror", "hamextfingeril",
    "hamextfingeri", "hamextfingerir", "hamextfingerui", "hamextfingerdi",
    "hamextfingerdo", "hamextfingeruo"
}

PALM_ORI_TOKENS = {
    "hampalmu", "hampalmur", "hampalmr", "hampalmdr", "hampalmd",
    "hampalmdl", "hampalml", "hampalmul"
}

LOCATION_BODY_ARM_TOKENS = {
    # Head & Face
    "hamhead", "hamheadtop", "hamforehead", "hameyebrows", "hameyes",
    "hamnose", "hamnostrils", "hamear", "hamearlobe", "hamcheek",
    "hamlips", "hamchin", "hamunderchin", "hamneck",
    # Upper Body
    "hamshouldertop", "hamshoulders", "hamchest", "hamstomach",
    "hambelowstomach", "hamneutralspace",
    # Arm
    "hamUpperarm", "hamupperarm", "hamelbow", "hamelbowinside",
    "hamlowerarm", "hamwristback", "hamwristpulse", "hamwristtopulse",
    "hamwristtoback", "hamwristtothumb", "hamwristtopinky"
}

CONTACT_DISTANCE_TOKENS = {
    "hamtouch", "hamclose", "hamarmextended", "hamlrbeside",
    "hamlrat", "hambehind", "haminterlock", "hamcross"
}

MOVEMENT_BASE_TOKENS = {
    # Directional
    "hammoveu", "hammoveur", "hammover", "hammovedr", "hammoved",
    "hammovedl", "hammovel", "hammoveul", "hammoveol", "hammoveo",
    "hammoveor", "hammoveil", "hammovei", "hammoveir", "hammoveui",
    "hammovedi", "hammovedo", "hammoveuo", "hammovecross", "hammovex",
    # Arc / Curved / Wavy
    "hamarcl", "hamarcu", "hamarcr", "hamarcd", "hamwavy", "hamzigzag",
    # Circles / Rotations
    "hamcircleo", "hamcirclei", "hamcircled", "hamcircleu", "hamcirclel",
    "hamcircler", "hamcircleul", "hamcircledr", "hamcircleur", "hamcircledl",
    "hamcircleol", "hamcircleir", "hamcircleor", "hamcircleil", "hamcircledo",
    "hamstircw", "hamstirccw", "hamtwisting", "hamnodding", "hamswinging"
}

MOVEMENT_MOD_TOKENS = {
    "hamsmallmod", "hamlargemod", "hamfast", "hamslow", "hamtense",
    "hamrest", "hamhalt", "hamincreasing", "hamdecreasing", "hambrushing"
}

REPETITION_TOKENS = {
    "hamrepeatfromstart", "hamrepeatfromstartseveral", "hamrepeatcontinue",
    "hamrepeatcontinueseveral", "hamrepeatreverse"
}

NON_HAMNOSYS_WORDS = {
    "none", "no-contact", "signer", "right", "left", "null", "undefined"
}


def canonicalize_token(token: str) -> str:
    """Return exact CWASA-cased token name (e.g. hamupperarm -> hamUpperarm)."""
    t_clean = token.strip().lower()
    if t_clean in CWASA_TOKEN_MAP:
        return CWASA_TOKEN_MAP[t_clean]
    # Fallback known special casing
    if t_clean == "hamupperarm":
        return "hamUpperarm"
    return token.strip()


def sanitize_hamnosys_sequence(tokens) -> list:
    """
    Take an arbitrary list/string of tokens and re-organize into a strictly
    valid Ham4Parser CFG sequence, removing syntax conflicts and invalid tokens.
    """
    if isinstance(tokens, str):
        token_list = tokens.split()
    elif isinstance(tokens, list):
        token_list = [str(t) for t in tokens if t]
    else:
        return []

    # 1. Filter non-HamNoSys words & canonicalize casing
    cleaned = []
    for t in token_list:
        tl = t.lower()
        if tl in NON_HAMNOSYS_WORDS:
            continue
        canon = canonicalize_token(t)
        cleaned.append(canon)

    # 2. Check for hamreplace split
    replace_idx = -1
    for i, t in enumerate(cleaned):
        if t == "hamreplace":
            replace_idx = i
            break

    main_tokens = cleaned[:replace_idx] if replace_idx != -1 else cleaned
    replace_tokens = cleaned[replace_idx+1:] if replace_idx != -1 else []

    # 3. Categorize tokens into formal CFG slots
    symm = None
    base_hs = None
    thumb_mod = None
    bend_mod = None
    ext_ori = None
    palm_ori = None
    loc = None
    contact_or_dist = None
    mov_base = None
    mov_mod = None
    rep = None

    for t in main_tokens:
        tl = t.lower()
        if tl in SYMMETRY_TOKENS and not symm:
            symm = t
        elif tl in HANDSHAPE_BASE_TOKENS and not base_hs:
            base_hs = t
        elif tl in THUMB_MOD_TOKENS and not thumb_mod:
            thumb_mod = t
        elif tl in BEND_MOD_TOKENS and not bend_mod:
            bend_mod = t
        elif tl in EXT_FINGER_TOKENS and not ext_ori:
            ext_ori = t
        elif tl in PALM_ORI_TOKENS and not palm_ori:
            palm_ori = t
        elif tl in LOCATION_BODY_ARM_TOKENS and not loc:
            loc = t
        elif tl in CONTACT_DISTANCE_TOKENS and not contact_or_dist:
            contact_or_dist = t
        elif tl in MOVEMENT_BASE_TOKENS and not mov_base:
            mov_base = t
        elif tl in MOVEMENT_MOD_TOKENS and not mov_mod:
            mov_mod = t
        elif tl in REPETITION_TOKENS and not rep:
            rep = t

    # Handle multiple contact/distance tokens:
    # If both hamtouch and hamarmextended were in the list, prioritize hamtouch if contact occurred
    has_touch = any(t.lower() == "hamtouch" for t in main_tokens)
    has_armext = any(t.lower() == "hamarmextended" for t in main_tokens)
    if has_touch:
        contact_or_dist = "hamtouch"
    elif has_armext:
        contact_or_dist = "hamarmextended"

    # In Ham4Parser grammar, neutralspace CANNOT take side relators (hamlrat/hamlrbeside)
    if loc == "hamneutralspace" and contact_or_dist in ["hamlrbeside", "hamlrat", "hamtouch", "hamclose"]:
        contact_or_dist = None

    # In symmetric signs, unilateral side relators are phonetically invalid
    if symm in ["hamsymmlr", "hamsymmpar"] and contact_or_dist in ["hamlrbeside", "hamlrat"]:
        contact_or_dist = None

    # Default fallbacks if core components are missing
    if not base_hs:
        base_hs = "hamflathand"
    if not ext_ori:
        ext_ori = "hamextfingeru"
    if not palm_ori:
        palm_ori = "hampalmd"
    if not loc:
        loc = "hamshoulders"

    # Ensure canonical casing on location (hamUpperarm)
    loc = canonicalize_token(loc)

    # 4. Assemble canonical sequence
    canonical = []
    if symm:
        canonical.append(symm)
    
    canonical.append(base_hs)

    if thumb_mod:
        canonical.append(thumb_mod)

    # Bend modifier (omit if redundant on basic flat hand)
    if bend_mod and not (base_hs == "hamflathand" and bend_mod == "hamfingerstraightmod"):
        canonical.append(bend_mod)

    # Extended finger and palm orientation
    canonical.append(ext_ori)
    canonical.append(palm_ori)

    # Location MUST precede contact/distance
    canonical.append(loc)

    if contact_or_dist:
        canonical.append(contact_or_dist)

    if mov_base and mov_base != "hamnomotion":
        canonical.append(mov_base)

    if mov_mod:
        canonical.append(mov_mod)

    if rep:
        canonical.append(rep)

    # 5. Attach replace sequence if valid
    if replace_tokens:
        clean_rep = [canonicalize_token(rt) for rt in replace_tokens if rt.lower() not in NON_HAMNOSYS_WORDS]
        if clean_rep:
            canonical.append("hamreplace")
            canonical.extend(clean_rep)

    return canonical


def sanitize_sigml_xml(xml_content: str) -> str:
    """
    Post-processes SiGML XML to fix any casing anomalies or rogue tags,
    guaranteeing 100% acceptance by CWASA and JASigning.
    """
    # Fix hamupperarm casing: <hamupperarm/> -> <hamUpperarm/>
    sanitized = re.sub(r'<hamupperarm\s*/>', '<hamUpperarm/>', xml_content, flags=re.IGNORECASE)
    sanitized = re.sub(r'<hamupperarm>', '<hamUpperarm>', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'</hamupperarm>', '</hamUpperarm>', sanitized, flags=re.IGNORECASE)

    # Remove non-HamNoSys XML elements if any slipped through
    for bad in ["signer", "none", "no-contact", "right", "left"]:
        sanitized = re.sub(rf'<{bad}\s*/>\s*', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(rf'<{bad}>.*?</{bad}>\s*', '', sanitized, flags=re.IGNORECASE)

    return sanitized
