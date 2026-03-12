"""Quadcopter asset definition."""

from __future__ import annotations

import textwrap

import mujoco

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


def _build_quadcopter_spec() -> mujoco.MjSpec:
  xml = textwrap.dedent(
    """
    <mujoco model="mjdrone_quadcopter">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>

      <default>
        <geom rgba="0.35 0.38 0.42 1"/>
        <site size="0.012" rgba="0.95 0.2 0.2 1"/>
      </default>

      <worldbody>
        <body name="base" pos="0 0 0.105">
          <freejoint name="root"/>
          <inertial pos="0 0 0" mass="1.2" diaginertia="0.015 0.015 0.03"/>

          <geom name="body_lower" type="box" size="0.085 0.055 0.022" rgba="0.18 0.2 0.24 1" group="1"/>
          <geom name="body_upper" type="box" pos="0 0 0.022" size="0.07 0.045 0.015" rgba="0.28 0.31 0.37 1" group="1"/>
          <geom name="battery_pack" type="box" pos="0 0 -0.028" size="0.06 0.035 0.014" rgba="0.1 0.11 0.13 1" group="1"/>
          <geom name="mast" type="capsule" fromto="0 0 0.01 0 0 0.07" size="0.01" rgba="0.32 0.34 0.38 1" group="1"/>
          <geom name="gps_hat" type="sphere" pos="0 0 0.078" size="0.018" rgba="0.88 0.9 0.94 1" contype="0" conaffinity="0" group="1"/>

          <geom name="arm_x" type="capsule" fromto="-0.2 0 0.005 0.2 0 0.005" size="0.012" rgba="0.16 0.18 0.2 1" group="1"/>
          <geom name="arm_y" type="capsule" fromto="0 -0.2 0.005 0 0.2 0.005" size="0.012" rgba="0.16 0.18 0.2 1" group="1"/>

          <geom name="motor_fl" type="cylinder" pos="0.2 0.2 0.005" size="0.026 0.012" rgba="0.18 0.18 0.2 1" group="1"/>
          <geom name="motor_fr" type="cylinder" pos="0.2 -0.2 0.005" size="0.026 0.012" rgba="0.18 0.18 0.2 1" group="1"/>
          <geom name="motor_rl" type="cylinder" pos="-0.2 0.2 0.005" size="0.026 0.012" rgba="0.18 0.18 0.2 1" group="1"/>
          <geom name="motor_rr" type="cylinder" pos="-0.2 -0.2 0.005" size="0.026 0.012" rgba="0.18 0.18 0.2 1" group="1"/>

          <geom name="camera_mount" type="box" pos="0.097 0 0.002" size="0.03 0.02 0.016" rgba="0.82 0.42 0.14 1" group="1"/>
          <geom name="camera_lens" type="cylinder" pos="0.124 0 0.002" quat="0.70710678 0 0.70710678 0" size="0.012 0.016" rgba="0.08 0.08 0.1 1" group="1"/>
          <geom name="front_marker" type="capsule" fromto="0.02 0 0.03 0.085 0 0.03" size="0.008" rgba="0.95 0.45 0.12 1" contype="0" conaffinity="0" group="1"/>

          <geom name="skid_left" type="capsule" fromto="-0.1 0.11 -0.085 0.1 0.11 -0.085" size="0.009" rgba="0.55 0.58 0.62 1" group="1"/>
          <geom name="skid_right" type="capsule" fromto="-0.1 -0.11 -0.085 0.1 -0.11 -0.085" size="0.009" rgba="0.55 0.58 0.62 1" group="1"/>
          <geom name="skid_link_fl" type="capsule" fromto="0.085 0.11 -0.01 0.085 0.11 -0.085" size="0.006" rgba="0.55 0.58 0.62 1" group="1"/>
          <geom name="skid_link_fr" type="capsule" fromto="0.085 -0.11 -0.01 0.085 -0.11 -0.085" size="0.006" rgba="0.55 0.58 0.62 1" group="1"/>
          <geom name="skid_link_rl" type="capsule" fromto="-0.085 0.11 -0.01 -0.085 0.11 -0.085" size="0.006" rgba="0.55 0.58 0.62 1" group="1"/>
          <geom name="skid_link_rr" type="capsule" fromto="-0.085 -0.11 -0.01 -0.085 -0.11 -0.085" size="0.006" rgba="0.55 0.58 0.62 1" group="1"/>

          <geom name="guard_fl" type="cylinder" pos="0.2 0.2 0.005" size="0.072 0.0025" rgba="0.8 0.8 0.82 0.35" contype="0" conaffinity="0" group="1"/>
          <geom name="guard_fr" type="cylinder" pos="0.2 -0.2 0.005" size="0.072 0.0025" rgba="0.8 0.8 0.82 0.35" contype="0" conaffinity="0" group="1"/>
          <geom name="guard_rl" type="cylinder" pos="-0.2 0.2 0.005" size="0.072 0.0025" rgba="0.8 0.8 0.82 0.35" contype="0" conaffinity="0" group="1"/>
          <geom name="guard_rr" type="cylinder" pos="-0.2 -0.2 0.005" size="0.072 0.0025" rgba="0.8 0.8 0.82 0.35" contype="0" conaffinity="0" group="1"/>

          <site name="imu_site" pos="0 0 0"/>
          <site name="rotor_fl" pos="0.2 0.2 0.005"/>
          <site name="rotor_fr" pos="0.2 -0.2 0.005"/>
          <site name="rotor_rl" pos="-0.2 0.2 0.005"/>
          <site name="rotor_rr" pos="-0.2 -0.2 0.005"/>

          <geom name="rotor_fl_vis" type="cylinder" pos="0.2 0.2 0.022" size="0.06 0.003" contype="0" conaffinity="0" rgba="0.92 0.2 0.2 0.7" group="1"/>
          <geom name="rotor_fr_vis" type="cylinder" pos="0.2 -0.2 0.022" size="0.06 0.003" contype="0" conaffinity="0" rgba="0.92 0.52 0.16 0.7" group="1"/>
          <geom name="rotor_rl_vis" type="cylinder" pos="-0.2 0.2 0.022" size="0.06 0.003" contype="0" conaffinity="0" rgba="0.18 0.76 0.3 0.7" group="1"/>
          <geom name="rotor_rr_vis" type="cylinder" pos="-0.2 -0.2 0.022" size="0.06 0.003" contype="0" conaffinity="0" rgba="0.12 0.52 0.98 0.7" group="1"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="thrust_fl" site="rotor_fl" gear="0 0 1 0 0 0" ctrlrange="0 7"/>
        <motor name="thrust_fr" site="rotor_fr" gear="0 0 1 0 0 0" ctrlrange="0 7"/>
        <motor name="thrust_rl" site="rotor_rl" gear="0 0 1 0 0 0" ctrlrange="0 7"/>
        <motor name="thrust_rr" site="rotor_rr" gear="0 0 1 0 0 0" ctrlrange="0 7"/>
      </actuator>
    </mujoco>
    """
  ).strip()
  return mujoco.MjSpec.from_string(xml)


def get_quadcopter_cfg() -> EntityCfg:
  """Create the quadcopter entity configuration."""
  return EntityCfg(
    spec_fn=_build_quadcopter_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.105),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
      joint_pos={},
      joint_vel={},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlMotorActuatorCfg(
          target_names_expr=("rotor_fl", "rotor_fr", "rotor_rl", "rotor_rr"),
          transmission_type=TransmissionType.SITE,
        ),
      )
    ),
  )
