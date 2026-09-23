import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval.run_full_benchmark import (
    load_iovnbd_session,
    load_two_wheeler_session,
    preprocess_session,
    latlon_to_enu,
    ProductionMobileFusionEngine
)
from engine.calibration.calibrator import CalibrationEngine
from engine.map_matching.hmm_matcher import HMMMapMatcher, RoadNetwork, RoadSegment

def build_gt_road_network(e_gt, n_gt):
    rn = RoadNetwork(lat0=0.0, lon0=0.0)
    pts = np.column_stack([e_gt, n_gt])
    step = 5
    seg_id = 1
    for i in range(0, len(pts) - step, step):
        p1 = pts[i]
        p2 = pts[i+step]
        rn.segments.append(RoadSegment(seg_id, 1000 + seg_id, p1, p2))
        seg_id += 1
    rn._build_spatial_index()
    return rn

def run_continuous_closed_loop(session_config):
    category = session_config["category"]
    driver = session_config.get("driver")
    session_name = session_config["session"]
    raw_root = "data/raw"
    dt = 0.1

    print(f"\n=======================================================")
    print(f"Testing CONTINUOUS Map-Matching Feedback on {category.upper()}: {session_name}")
    print(f"=======================================================")

    if category == "car":
        s_df, v_df = load_iovnbd_session(raw_root, driver, session_name)
        synced = preprocess_session(s_df, v_df, target_dt=dt)
        veh_type = "car"
    elif category == "two_wheeler":
        synced = load_two_wheeler_session(os.path.join(raw_root, "two_wheeler"), session_name)
        veh_type = "two_wheeler"

    N = len(synced)
    acc = synced[["acc_x", "acc_y", "acc_z"]].values
    gyro = synced[["gyro_x", "gyro_y", "gyro_z"]].values
    speed = np.nan_to_num(synced["gt_speed"].values, nan=0.0)

    calib = CalibrationEngine()
    calib.calibrate_from_session(acc[:min(1200, N)], gyro[:min(1200, N)], speed[:min(1200, N)], dt=dt)

    lat0 = synced["gt_lat"].iloc[0]
    lon0 = synced["gt_lon"].iloc[0]
    alt0 = synced["gt_alt"].iloc[0]
    e_gt, n_gt, u_gt = latlon_to_enu(synced["gt_lat"].values, synced["gt_lon"].values, synced["gt_alt"].values, lat0, lon0, alt0)

    vx_gt = np.gradient(e_gt, dt)
    vy_gt = np.gradient(n_gt, dt)
    vz_gt = np.gradient(u_gt, dt)

    gt_heading = np.degrees(np.arctan2(vx_gt, vy_gt)) % 360.0

    rn = build_gt_road_network(e_gt, n_gt)
    matcher = HMMMapMatcher(rn, vehicle_type=veh_type)

    # Pick an active outage
    moving_mask = speed > 3.0
    moving_indices = np.where(moving_mask)[0]
    outage_len = int(60.0 / dt)
    start_idx = moving_indices[len(moving_indices) // 4]
    end_idx = start_idx + outage_len

    engine = ProductionMobileFusionEngine(k=100.0, dt=dt)
    engine.calib = calib
    engine.current_vehicle_type = veh_type

    p0 = np.array([e_gt[0], n_gt[0], u_gt[0]])
    v0 = np.array([vx_gt[0], vy_gt[0], vz_gt[0]])
    h0 = gt_heading[0]
    engine.initialize_state(p0, v0, h0, acc[0])

    dr_pos = []
    gt_pos_outage = []
    snap_success = []

    snapping_lost_at = None

    for i in range(min(N, end_idx + 10)):
        in_outage = (start_idx <= i <= end_idx)
        
        gps_p = np.array([e_gt[i], n_gt[i], u_gt[i]]) if not in_outage else None
        gps_v = np.array([vx_gt[i], vy_gt[i], vz_gt[i]]) if not in_outage else None
        
        res = engine.step(
            acc_raw=acc[i],
            gyro_raw=gyro[i],
            gnss_pos_enu=gps_p,
            gnss_vel_enu=gps_v,
            is_gnss_available=not in_outage,
            timestamp=i * dt,
            gnss_acc_m=2.0 if not in_outage else 99.0,
            gnss_sat_count=16 if not in_outage else 0,
            gnss_avg_cn0=38.0 if not in_outage else 0.0
        )

        if in_outage:
            current_pos = res['pos']
            current_heading_rad = engine.ekf.get_euler_angles_deg()[2] # Not rad, actually deg, wait method says get_euler_angles_deg()
            
            # Apply Continuous MM Correction (Every Epoch: 10Hz)
            match_res = matcher.match_point(current_pos, current_heading_rad)
            
            if match_res.snapped:
                # 1. Update position
                snapped_p = np.array([match_res.snapped_pos[0], match_res.snapped_pos[1], current_pos[2]])
                engine.ekf.update_gnss_position(snapped_p, sigma_pos=2.0)
                
                # 2. Update heading based on road bearing
                seg = matcher.last_matched_seg
                if seg is not None:
                    # Update heading
                    engine.ekf.update_heading(np.radians(seg.bearing_deg), sigma_heading=np.radians(5.0), source="MAP_HEADING")
                snap_success.append(True)
            else:
                snap_success.append(False)
                if snapping_lost_at is None:
                    snapping_lost_at = (i - start_idx) * dt

            dr_pos.append(engine.ekf.p[:2].copy())
            gt_pos_outage.append(np.array([e_gt[i], n_gt[i]]))

    if snapping_lost_at is None:
        print("Snap maintained continuously for the entire 60s outage!")
    else:
        print(f"Snap LOST after {snapping_lost_at:.1f}s of continuous outage.")

    if len(dr_pos) > 0:
        dr_pos = np.array(dr_pos)
        gt_pos_outage = np.array(gt_pos_outage)

        dist_traveled = np.sum(np.linalg.norm(np.diff(gt_pos_outage, axis=0), axis=1))
        final_error = np.linalg.norm(dr_pos[-1] - gt_pos_outage[-1])
        drift_pct = (final_error / dist_traveled) * 100.0 if dist_traveled > 0 else 0
        snap_rate = np.mean(snap_success) * 100.0 if snap_success else 0

        print(f"Outage Duration: 60.0s")
        print(f"Distance Traveled: {dist_traveled:.2f} m")
        print(f"Final Position Error: {final_error:.2f} m")
        print(f"Drift %: {drift_pct:.2f}% (Target: <= 10.0%)")
        print(f"Map Matching Active Snapping Success: {snap_rate:.1f}%")

if __name__ == "__main__":
    run_continuous_closed_loop({"category": "car", "driver": "S (Driver A)", "session": "S4"})
    run_continuous_closed_loop({"category": "two_wheeler", "session": "session1"})
