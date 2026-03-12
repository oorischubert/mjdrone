"""Static landmark assets used to build randomized outdoor scenes."""

from __future__ import annotations

import math

import mujoco

from mjlab.entity import EntityCfg


ROAD_TILE_RADIUS = 2
ROAD_TILE_SIZE = 3.6
ROAD_HALF_WIDTH = 0.62
ROAD_BRANCH_HALF_LENGTH = (ROAD_TILE_SIZE * 0.5 - ROAD_HALF_WIDTH) * 0.5
GROUND_HALF_EXTENT = (ROAD_TILE_RADIUS + 1.2) * ROAD_TILE_SIZE


def get_ground_decor_cfg() -> EntityCfg:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ground_decor")
  body.add_geom(
    name="grass_main",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.0015),
    size=(GROUND_HALF_EXTENT, GROUND_HALF_EXTENT, 0.0015),
    rgba=(0.24, 0.45, 0.2, 1.0),
    contype=0,
    conaffinity=0,
  )
  body.add_geom(
    name="grass_inner",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.0028),
    size=(GROUND_HALF_EXTENT - 1.8, GROUND_HALF_EXTENT - 1.8, 0.0011),
    rgba=(0.28, 0.5, 0.23, 1.0),
    contype=0,
    conaffinity=0,
  )

  road_z = 0.004
  lane_z = 0.0054
  for row in range(-ROAD_TILE_RADIUS, ROAD_TILE_RADIUS + 1):
    for col in range(-ROAD_TILE_RADIUS, ROAD_TILE_RADIUS + 1):
      x = col * ROAD_TILE_SIZE
      y = row * ROAD_TILE_SIZE
      row_idx = row + ROAD_TILE_RADIUS
      col_idx = col + ROAD_TILE_RADIUS

      body.add_geom(
        name=f"road_center_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y, road_z),
        size=(ROAD_HALF_WIDTH, ROAD_HALF_WIDTH, 0.0012),
        rgba=(0.2, 0.2, 0.22, 1.0),
        contype=0,
        conaffinity=0,
      )
      body.add_geom(
        name=f"road_north_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y + ROAD_HALF_WIDTH + ROAD_BRANCH_HALF_LENGTH, road_z),
        size=(ROAD_HALF_WIDTH, ROAD_BRANCH_HALF_LENGTH, 0.0012),
        rgba=(0.2, 0.2, 0.22, 1.0),
        contype=0,
        conaffinity=0,
      )
      body.add_geom(
        name=f"road_south_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y - ROAD_HALF_WIDTH - ROAD_BRANCH_HALF_LENGTH, road_z),
        size=(ROAD_HALF_WIDTH, ROAD_BRANCH_HALF_LENGTH, 0.0012),
        rgba=(0.2, 0.2, 0.22, 1.0),
        contype=0,
        conaffinity=0,
      )
      body.add_geom(
        name=f"road_east_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x + ROAD_HALF_WIDTH + ROAD_BRANCH_HALF_LENGTH, y, road_z),
        size=(ROAD_BRANCH_HALF_LENGTH, ROAD_HALF_WIDTH, 0.0012),
        rgba=(0.2, 0.2, 0.22, 1.0),
        contype=0,
        conaffinity=0,
      )
      body.add_geom(
        name=f"road_west_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x - ROAD_HALF_WIDTH - ROAD_BRANCH_HALF_LENGTH, y, road_z),
        size=(ROAD_BRANCH_HALF_LENGTH, ROAD_HALF_WIDTH, 0.0012),
        rgba=(0.2, 0.2, 0.22, 1.0),
        contype=0,
        conaffinity=0,
      )

      lane_half_length = ROAD_BRANCH_HALF_LENGTH * 0.72
      body.add_geom(
        name=f"lane_north_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y + ROAD_HALF_WIDTH + ROAD_BRANCH_HALF_LENGTH, lane_z),
        size=(0.045, lane_half_length, 0.0007),
        rgba=(0.95, 0.94, 0.86, 1.0),
        contype=0,
        conaffinity=0,
      )
      body.add_geom(
        name=f"lane_south_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y - ROAD_HALF_WIDTH - ROAD_BRANCH_HALF_LENGTH, lane_z),
        size=(0.045, lane_half_length, 0.0007),
        rgba=(0.95, 0.94, 0.86, 1.0),
        contype=0,
        conaffinity=0,
      )
      body.add_geom(
        name=f"lane_east_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x + ROAD_HALF_WIDTH + ROAD_BRANCH_HALF_LENGTH, y, lane_z),
        size=(lane_half_length, 0.045, 0.0007),
        rgba=(0.95, 0.94, 0.86, 1.0),
        contype=0,
        conaffinity=0,
      )
      body.add_geom(
        name=f"lane_west_{row_idx}_{col_idx}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x - ROAD_HALF_WIDTH - ROAD_BRANCH_HALF_LENGTH, y, lane_z),
        size=(lane_half_length, 0.045, 0.0007),
        rgba=(0.95, 0.94, 0.86, 1.0),
        contype=0,
        conaffinity=0,
      )

  return EntityCfg(spec_fn=lambda: spec)


def get_tree_cfg() -> EntityCfg:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="tree")
  body.add_geom(
    name="trunk",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.0, 0.0, 0.0, 0.0, 0.0, 1.25),
    size=(0.08,),
    rgba=(0.42, 0.29, 0.16, 1.0),
  )
  body.add_geom(
    name="canopy_low",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=(0.0, 0.0, 1.48),
    size=(0.45,),
    rgba=(0.24, 0.58, 0.24, 1.0),
  )
  body.add_geom(
    name="canopy_high",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=(0.09, -0.05, 1.82),
    size=(0.32,),
    rgba=(0.28, 0.62, 0.3, 1.0),
  )
  return EntityCfg(spec_fn=lambda: spec)


def get_car_cfg() -> EntityCfg:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="car")
  body.add_geom(
    name="body",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.22),
    size=(0.48, 0.24, 0.12),
    rgba=(0.78, 0.18, 0.16, 1.0),
  )
  body.add_geom(
    name="roof",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(-0.02, 0.0, 0.38),
    size=(0.24, 0.2, 0.09),
    rgba=(0.78, 0.18, 0.16, 1.0),
  )
  body.add_geom(
    name="windshield",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.18, 0.0, 0.34),
    size=(0.06, 0.18, 0.08),
    rgba=(0.56, 0.78, 0.9, 0.55),
    contype=0,
    conaffinity=0,
  )
  for suffix, wx, wy in (
    ("fl", 0.3, 0.2),
    ("fr", 0.3, -0.2),
    ("rl", -0.3, 0.2),
    ("rr", -0.3, -0.2),
  ):
    body.add_geom(
      name=f"wheel_{suffix}",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      pos=(wx, wy, 0.1),
      quat=(0.70710678, 0.0, 0.70710678, 0.0),
      size=(0.09, 0.05),
      rgba=(0.08, 0.08, 0.09, 1.0),
      contype=0,
      conaffinity=0,
    )
  return EntityCfg(spec_fn=lambda: spec)


def get_target_tank_cfg() -> EntityCfg:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="tank")
  body.add_geom(
    name="hull",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.3),
    size=(0.84, 0.4, 0.18),
    rgba=(0.9, 0.76, 0.34, 1.0),
  )
  body.add_geom(
    name="glacis",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.5, 0.0, 0.39),
    size=(0.28, 0.34, 0.09),
    euler=(0.0, -0.28, 0.0),
    rgba=(0.96, 0.82, 0.38, 1.0),
  )
  body.add_geom(
    name="turret",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.05, 0.0, 0.56),
    size=(0.34, 0.28, 0.13),
    rgba=(0.86, 0.7, 0.3, 1.0),
  )
  body.add_geom(
    name="turret_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.72),
    size=(0.2, 0.2, 0.06),
    rgba=(0.98, 0.88, 0.44, 1.0),
  )
  body.add_geom(
    name="barrel",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.26, 0.0, 0.62, 1.32, 0.0, 0.62),
    size=(0.06,),
    rgba=(0.22, 0.21, 0.17, 1.0),
  )
  body.add_geom(
    name="barrel_tip",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=(1.35, 0.0, 0.62),
    quat=(0.70710678, 0.0, 0.70710678, 0.0),
    size=(0.075, 0.03),
    rgba=(0.93, 0.5, 0.16, 1.0),
    contype=0,
    conaffinity=0,
  )
  for suffix, y in (("left", 0.39), ("right", -0.39)):
    body.add_geom(
      name=f"track_{suffix}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(0.0, y * 1.05, 0.2),
      size=(0.86, 0.09, 0.2),
      rgba=(0.15, 0.15, 0.14, 1.0),
    )
    body.add_geom(
      name=f"side_skirt_{suffix}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(0.0, y * 0.92, 0.4),
      size=(0.78, 0.04, 0.09),
      rgba=(0.78, 0.62, 0.24, 1.0),
    )
  body.add_geom(
    name="sensor_pod",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=(-0.22, 0.0, 0.82),
    size=(0.08,),
    rgba=(0.14, 0.16, 0.18, 1.0),
    contype=0,
    conaffinity=0,
  )
  return EntityCfg(spec_fn=lambda: spec)


def get_billboard_cfg() -> EntityCfg:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="billboard")
  body.add_geom(
    name="post",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.0, 0.0, 0.0, 0.0, 0.0, 1.65),
    size=(0.06,),
    rgba=(0.35, 0.35, 0.38, 1.0),
  )
  body.add_geom(
    name="panel",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.12, 0.0, 2.0),
    size=(0.05, 0.95, 0.44),
    rgba=(0.92, 0.82, 0.24, 1.0),
  )
  body.add_geom(
    name="stripe",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.08, 0.0, 2.02),
    size=(0.02, 0.9, 0.08),
    rgba=(0.18, 0.28, 0.72, 1.0),
    contype=0,
    conaffinity=0,
  )
  return EntityCfg(spec_fn=lambda: spec)
