"""Locate a built variant: robot/<variant>/haetburi_scene.xml and its variant.json."""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = "v6_split_round"


def available():
    base = os.path.join(ROOT, "robot")
    return sorted(d for d in os.listdir(base)
                  if os.path.isfile(os.path.join(base, d, "haetburi_scene.xml")))


def scene(variant):
    path = os.path.join(ROOT, "robot", variant, "haetburi_scene.xml")
    if not os.path.isfile(path):
        raise SystemExit(f"no built model for {variant!r}; built: {available()}")
    return path


def info(variant):
    with open(os.path.join(ROOT, "robot", variant, "variant.json")) as fh:
        return json.load(fh)


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--variant", default=DEFAULT,
                    help=f"model to load (default {DEFAULT}); see robot/")
    return ap
