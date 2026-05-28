import pandas as pd
import sys
import numpy as np
import time
from IPython.display import display
import dataframe_image as dfi

# Combined Import
from TES_Availability_Engine import (
    calculate_thermal_potentials as calc_combined, 
    PV_SETTINGS, 
    HP_HEATING_SETTINGS,
    HP_COOLING_SETTINGS
)

def run_tes_sizing_engine(
    thermal_data_file='Input - Results_Yearly_Zone_Demands_Combined.csv', # This is a placeholder file that is overwritten by the Master Engines (FIXED & DYANMIC)
    materials_file='Input - Multi zone simplified thermal mass - case study.txt',
    T_START_LIST=[21.0, 22.0, 21.0, 21.0, 21.0],
    T_SET_LIST=[21.0, 22.0, 21.0, 21.0, 21.0],
    T_SWING_LIST=[2.0, 0.0, 2.0, 2.0, 2.0],
    HP_MAX_CAPACITY_H=115.0,
    HP_MAX_CAPACITY_C=65.0,
    REHEAT_STRATEGY_H="proportional",
    REHEAT_PRIORITY_H=[1, 2, 3, 4, 5],
    DISCHARGE_STRATEGY_H="proportional",
    DISCHARGE_PRIORITY_H=[1, 2, 3, 4, 5],
    ZONE_TES_FIRST_H=[False, False, False, False, False],
    delta_t_H=10.0,
    MANUAL_TANK_KWH_H=None,
    REHEAT_STRATEGY_C="proportional",
    REHEAT_PRIORITY_C=[1, 2, 3, 4, 5],
    DISCHARGE_STRATEGY_C="proportional",
    DISCHARGE_PRIORITY_C=[1, 2, 3, 4, 5],
    ZONE_TES_FIRST_C=[False, False, False, False, False],
    delta_t_C=8.0,
    MANUAL_TANK_KWH_C=None,
    ZONES_DELTA_T_H=[10.0, 10.0, 10.0, 10.0, 10.0],
    ZONES_DELTA_T_C=[8.0, 8.0, 8.0, 8.0, 8.0],
    S=1.2, rho=1000, cp=4.18, eta_str=0.95,
    DEAD_ZONE=0.20,
    TAU_STABILITY=0.25,
    UTILIZATION_FACTOR=1.2,
    KVA_SPLIT_MODE='proportional',
    HEATING_SHARE_INPUT=0.5,
    HP_MIN_MOD_KVA=1.5,
    DRIFT_RATE=0.0001,
    SHOW_FLOW_RATES=True,
    SILENT_MODE=False  
):
    """
    Exposes your final TES Sizing Engine as a callable function.
    All variables are exposed with their default original values.
    """
    MAX_COP_LIMIT_H = HP_HEATING_SETTINGS['MAX_COP']
    MAX_COP_LIMIT_C = HP_COOLING_SETTINGS['MAX_COP']
    TOTAL_EFFICIENCY = eta_str * (1 - DEAD_ZONE)
    
    MIN_MODULATION_KVA_H = HP_MIN_MOD_KVA
    MIN_MODULATION_KVA_C = HP_MIN_MOD_KVA

    # =================================================================
    # 2. DATA LOADING & PRE-PROCESSING (COMBINED)
    # =================================================================
    try:
        df_mats = pd.read_csv(materials_file) 
        df_mats['kWh_per_K'] = (df_mats['FloorArea'] * df_mats['SpecCapacity'] * 1000) / 3.6 / 1_000_000
        zone_masses = df_mats['kWh_per_K'].tolist()
        
        df_input = pd.read_csv(thermal_data_file)
        df_input = df_input.fillna(0)

        df_input['Hour'] = df_input.index + 1 
        demand_cols = [col for col in df_input.columns if 'Demand' in col]
        num_zones = len(demand_cols)

        if len(T_START_LIST) != num_zones and not SILENT_MODE:
            print(f"Warning: T_START_LIST has {len(T_START_LIST)} items but file has {num_zones} zones. Adjusting...")

        df_solar = pd.read_csv('Input PV - NEN5060-B2 1%.txt')
        df_temp  = pd.read_csv('Input Temp - NEN5060-B2 1%.txt')
        df_grid  = pd.read_csv('Output - total kVA - Hour, Grid_kVA.csv')

        sim_length = len(df_input)
        grid_indices = np.arange(sim_length) % len(df_grid)
        df_grid_aligned = df_grid.iloc[grid_indices].reset_index(drop=True)

        solar_indices = np.arange(sim_length) % len(df_solar)
        df_weather = df_solar.iloc[solar_indices].reset_index(drop=True)

        temp_indices = np.arange(sim_length) % len(df_temp)  
        df_weather['T'] = df_temp['T'].values[temp_indices]

        h_dem_series = df_input[demand_cols].apply(lambda x: x[x > 0].sum(), axis=1)
        c_dem_series = df_input[demand_cols].apply(lambda x: abs(x[x < 0].sum()), axis=1)

        df_potentials = calc_combined(
            df_weather, 
            df_grid_aligned, 
            split_mode=KVA_SPLIT_MODE, 
            h_demand=h_dem_series, 
            c_demand=c_dem_series,
            manual_share=HEATING_SHARE_INPUT,
            hp_max_h=HP_MAX_CAPACITY_H,
            hp_max_c=HP_MAX_CAPACITY_C
        )

        df_input['Available_kVA_H'] = df_potentials['Elec_kVA_Heating'].values
        df_input['COP_H']           = df_potentials['COP_Heating'].values
        df_input['Max_Supply_kW_H'] = df_potentials['Potential_Heating_kW'].clip(upper=HP_MAX_CAPACITY_H).values

        df_input['Available_kVA_C'] = df_potentials['Elec_kVA_Cooling'].values
        df_input['COP_C']           = df_potentials['COP_Cooling'].values
        df_input['Max_Supply_kW_C'] = df_potentials['Potential_Cooling_kW'].clip(upper=HP_MAX_CAPACITY_C).values
        
        if not SILENT_MODE:
            print(f"Data Loaded: {len(df_input)} timesteps processed.")

    except Exception as e:
        sys.exit(f"File Error: {e}")

    # =================================================================
    # 3. INTERNAL SIMULATION ENGINE (PARALLEL DUAL-HP MODE)
    # =================================================================
    def simulate_merged_multi_zone(
        df_input, demand_cols, zone_masses, T_starts, T_sets, T_swings, 
        tank_cap_h, tank_cap_c,
        reh_strat_h, dis_strat_h, reh_prio_h, dis_prio_h, tes_h, zone_dt_list_h,
        reh_strat_c, dis_strat_c, reh_prio_c, dis_prio_c, tes_c, zone_dt_list_c,
        min_mod_kva_h, min_mod_kva_c, drift_rate
    ):
        T_mins = [s - sw for s, sw in zip(T_sets, T_swings)]
        T_maxs = [s + sw for s, sw in zip(T_sets, T_swings)]
        current_temps = list(T_starts)
        current_tank_kwh_h = tank_cap_h if MANUAL_TANK_KWH_H is None else MANUAL_TANK_KWH_H
        current_tank_kwh_c = tank_cap_c if MANUAL_TANK_KWH_C is None else MANUAL_TANK_KWH_C

        EPSILON = 1e-7 
        num_steps = len(df_input)
        num_zones = len(demand_cols)

        hour_arr = df_input['Hour'].values
        demand_arr = df_input[demand_cols].values

        max_supply_arr_h = df_input['Max_Supply_kW_H'].values
        cop_arr_h = df_input['COP_H'].values
        available_kva_arr_h = df_input['Available_kVA_H'].values
        cong_indices_h = np.where(max_supply_arr_h < (np.where(demand_arr > 0, demand_arr, 0).sum(axis=1) - EPSILON))[0]
        next_cong_map_h = np.searchsorted(cong_indices_h, np.arange(num_steps), side='right')

        max_supply_arr_c = df_input['Max_Supply_kW_C'].values
        cop_arr_c = df_input['COP_C'].values
        available_kva_arr_c = df_input['Available_kVA_C'].values
        cong_indices_c = np.where(max_supply_arr_c < (np.where(demand_arr < 0, abs(demand_arr), 0).sum(axis=1) - EPSILON))[0]
        next_cong_map_c = np.searchsorted(cong_indices_c, np.arange(num_steps), side='right')

        num_base_cols = 9
        num_per_zone = 4 if SHOW_FLOW_RATES else 3
        num_individual_cols = num_base_cols + (num_zones * num_per_zone)

        results_matrix_H = np.zeros((num_steps, num_individual_cols))
        results_matrix_C = np.zeros((num_steps, num_individual_cols))
        strat_h_hist, strat_c_hist = [], []

        for idx in range(num_steps):
            current_tank_kwh_h *= (1 - drift_rate)
            current_tank_kwh_c *= (1 - drift_rate)

            demands = demand_arr[idx]
            current_time_val = hour_arr[idx]
            
            h_demands = np.maximum(0, demands)
            c_demands = np.abs(np.minimum(0, demands))

            total_demand_h = np.sum(h_demands)
            total_demand_c = np.sum(c_demands)

            available_power_h = max_supply_arr_h[idx]
            current_cop_h = cop_arr_h[idx]
            current_cop_c = cop_arr_c[idx]
            
            delta_t_window_h, tsat_tes_h = 0.0, 0.0
            delta_t_window_c, tsat_tes_c = 0.0, 0.0
            tsats_per_zone_h = [0.0] * num_zones
            tsats_per_zone_c = [0.0] * num_zones
            active_strat_h = "Discharge"
            active_strat_c = "Discharge"
            current_zone_flows_h = [0.0] * num_zones
            current_zone_flows_c = [0.0] * num_zones

            e_to_zones_h, e_to_tank_h = 0.0, 0.0
            e_to_zones_c, e_to_tank_c = 0.0, 0.0
            actual_hp_out_h = 0.0
            actual_hp_out_c = 0.0
            deficit_to_track_h = 0.0
            deficit_to_track_c = 0.0

            results_matrix_H[idx, 0] = current_time_val
            results_matrix_H[idx, 1] = total_demand_h
            results_matrix_C[idx, 1] = total_demand_c

            # --- HEATING HP BRANCH ---
            if available_power_h >= (total_demand_h - EPSILON):
                current_zone_flows_h = [(d * 3600) / (rho * cp * zone_dt_list_h[i]) if d > EPSILON else 0.0 for i, d in enumerate(h_demands)]
                surplus_h = max(0.0, available_power_h - total_demand_h)
                actual_hp_out_h = total_demand_h + surplus_h
                
                map_idx_h = next_cong_map_h[idx]
                delta_t_window_h = float(hour_arr[cong_indices_h[map_idx_h]] - current_time_val) if map_idx_h < len(cong_indices_h) else float(hour_arr[-1] - current_time_val)
                
                hunger_h = [max(0, T_sets[i] - current_temps[i]) * zone_masses[i] for i in range(num_zones)]
                total_hunger_h = sum(hunger_h)

                tsat_struct_h = (total_hunger_h / surplus_h * UTILIZATION_FACTOR) if surplus_h > EPSILON else 0.0
                tsats_per_zone_h = [(h / surplus_h * UTILIZATION_FACTOR) if surplus_h > EPSILON else 0.0 for h in hunger_h]
                tank_gap_h = max(0, tank_cap_h - current_tank_kwh_h)
                tsat_tes_h = (tank_gap_h / surplus_h) if surplus_h > EPSILON else 0.0

                if delta_t_window_h >= (tsat_struct_h - EPSILON):
                    active_strat_h = "STRUCT"
                    e_to_zones_h = min(surplus_h, total_hunger_h)
                    e_to_tank_h = min(surplus_h - e_to_zones_h, tank_gap_h)
                else:
                    active_strat_h = "TES"
                    e_to_tank_h = min(surplus_h, tank_gap_h)
                    e_to_zones_h = min(surplus_h - e_to_tank_h, total_hunger_h)
                
                if e_to_zones_h > EPSILON:
                    if reh_strat_h == "proportional":
                        for i in range(num_zones):
                            if hunger_h[i] > EPSILON:
                                share_surplus_h = hunger_h[i] / total_hunger_h
                                zone_pwr_share = share_surplus_h * e_to_zones_h
                                tsats_per_zone_h[i] = (hunger_h[i] / zone_pwr_share * UTILIZATION_FACTOR)
                                current_temps[i] += (zone_pwr_share / zone_masses[i])
                                current_zone_flows_h[i] = ((h_demands[i] + zone_pwr_share) * 3600) / (rho * cp * zone_dt_list_h[i])
                    elif reh_strat_h == "sequential":
                        remaining_e_h = e_to_zones_h
                        for p_idx in reh_prio_h:
                            i = p_idx - 1
                            if hunger_h[i] > EPSILON and remaining_e_h > EPSILON:
                                give_h = min(remaining_e_h, hunger_h[i])
                                tsats_per_zone_h[i] = (hunger_h[i] / give_h * UTILIZATION_FACTOR)
                                current_temps[i] += (give_h / zone_masses[i])
                                current_zone_flows_h[i] = ((h_demands[i] + give_h) * 3600) / (rho * cp * zone_dt_list_h[i])
                                remaining_e_h -= give_h
                current_tank_kwh_h += e_to_tank_h
            else:
                actual_hp_out_h = max(min_mod_kva_h * current_cop_h, available_power_h) if total_demand_h > 0 else 0.0
                hp_shares_kw_h = [0.0] * num_zones
                if total_demand_h > EPSILON:
                    if dis_strat_h == "proportional":
                        for i in range(num_zones):
                            hp_shares_kw_h[i] = actual_hp_out_h * (h_demands[i] / total_demand_h)
                    elif dis_strat_h == "sequential":
                        rem_hp = actual_hp_out_h
                        for p_idx in dis_prio_h:
                            i = p_idx - 1
                            give_h = min(rem_hp, h_demands[i])
                            hp_shares_kw_h[i] = give_h
                            rem_hp -= give_h
                for i in range(num_zones):
                    if h_demands[i] > EPSILON:
                        deficit = h_demands[i] - hp_shares_kw_h[i]
                        actual_from_tank = 0.0
                        if tes_h[i]:
                            actual_from_tank = min(deficit, current_tank_kwh_h)
                            current_tank_kwh_h -= actual_from_tank
                            unmet_energy_h = deficit - actual_from_tank
                            if unmet_energy_h > EPSILON: deficit_to_track_h += unmet_energy_h
                            current_temps[i] -= unmet_energy_h / zone_masses[i]
                        else:
                            t_pred = current_temps[i] - (h_demands[i] / zone_masses[i]) + (hp_shares_kw_h[i] / zone_masses[i])
                            if t_pred < (T_mins[i] - EPSILON):
                                needed = (T_mins[i] - t_pred) * zone_masses[i]
                                actual_from_tank = min(needed, current_tank_kwh_h)
                                current_tank_kwh_h -= actual_from_tank
                                current_temps[i] = t_pred + (actual_from_tank / zone_masses[i])
                                if current_temps[i] < (T_mins[i] - EPSILON):
                                    deficit_to_track_h += (T_mins[i] - current_temps[i]) * zone_masses[i]
                            else:
                                current_temps[i] = t_pred
                        current_zone_flows_h[i] = ((hp_shares_kw_h[i] + actual_from_tank) * 3600) / (rho * cp * zone_dt_list_h[i])

            # --- COOLING HP BRANCH ---
            available_power_c = max_supply_arr_c[idx]
            if available_power_c >= (total_demand_c - EPSILON):
                current_zone_flows_c = [(d * 3600) / (rho * cp * zone_dt_list_c[i]) if d > EPSILON else 0.0 for i, d in enumerate(c_demands)]
                surplus_c = max(0.0, available_power_c - total_demand_c)
                actual_hp_out_c = total_demand_c + surplus_c
                
                map_idx_c = next_cong_map_c[idx]
                delta_t_window_c = float(hour_arr[cong_indices_c[map_idx_c]] - current_time_val) if map_idx_c < len(cong_indices_c) else float(hour_arr[-1] - current_time_val)
                
                hunger_c = [max(0, current_temps[i] - T_sets[i]) * zone_masses[i] for i in range(num_zones)]
                total_hunger_c = sum(hunger_c)

                tsat_struct_c = (total_hunger_c / surplus_c * UTILIZATION_FACTOR) if surplus_c > EPSILON else 0.0
                tsats_per_zone_c = [(h / surplus_c * UTILIZATION_FACTOR) if surplus_c > EPSILON else 0.0 for h in hunger_c]
                tank_gap_c = max(0, tank_cap_c - current_tank_kwh_c)
                tsat_tes_c = (tank_gap_c / surplus_c) if surplus_c > EPSILON else 0.0

                if delta_t_window_c >= (tsat_struct_c - EPSILON):
                    active_strat_c = "STRUCT"
                    e_to_zones_c = min(surplus_c, total_hunger_c)
                    e_to_tank_c = min(surplus_c - e_to_zones_c, tank_gap_c)
                else:
                    active_strat_c = "TES"
                    e_to_tank_c = min(surplus_c, tank_gap_c)
                    e_to_zones_c = min(surplus_c - e_to_tank_c, total_hunger_c)
                
                if e_to_zones_c > EPSILON:
                    if reh_strat_c == "proportional":
                        for i in range(num_zones):
                            if hunger_c[i] > EPSILON:
                                share_surplus_c = hunger_c[i] / total_hunger_c
                                zone_pwr_share_c = share_surplus_c * e_to_zones_c
                                current_temps[i] -= (zone_pwr_share_c / zone_masses[i])
                                current_zone_flows_c[i] = ((c_demands[i] + zone_pwr_share_c) * 3600) / (rho * cp * zone_dt_list_c[i])
                    elif reh_strat_c == "sequential":
                        remaining_e_c = e_to_zones_c
                        for p_idx in reh_prio_c:
                            i = p_idx - 1
                            if hunger_c[i] > EPSILON and remaining_e_c > EPSILON:
                                give_c = min(remaining_e_c, hunger_c[i])
                                current_temps[i] -= (give_c / zone_masses[i])
                                current_zone_flows_c[i] = ((c_demands[i] + give_c) * 3600) / (rho * cp * zone_dt_list_c[i])
                                remaining_e_c -= give_c
                current_tank_kwh_c += e_to_tank_c
            else:
                actual_hp_out_c = max(min_mod_kva_c * current_cop_c, available_power_c) if total_demand_c > 0 else 0.0
                hp_shares_kw_c = [0.0] * num_zones
                if total_demand_c > EPSILON:
                    if dis_strat_c == "proportional":
                        for i in range(num_zones):
                            hp_shares_kw_c[i] = actual_hp_out_c * (c_demands[i] / total_demand_c)
                    elif dis_strat_c == "sequential":
                        rem_hp_c = actual_hp_out_c
                        for p_idx in dis_prio_c:
                            i = p_idx - 1
                            give_c = min(rem_hp_c, c_demands[i])
                            hp_shares_kw_c[i] = give_c
                            rem_hp_c -= give_c
                for i in range(num_zones):
                    if c_demands[i] > EPSILON:
                        deficit = c_demands[i] - hp_shares_kw_c[i]
                        actual_from_tank = 0.0
                        if tes_c[i]:
                            actual_from_tank = min(deficit, current_tank_kwh_c)
                            current_tank_kwh_c -= actual_from_tank
                            unmet_energy_c = deficit - actual_from_tank
                            if unmet_energy_c > EPSILON: deficit_to_track_c += unmet_energy_c
                            current_temps[i] += unmet_energy_c / zone_masses[i]
                        else:
                            t_pred = current_temps[i] + (c_demands[i] / zone_masses[i]) - (hp_shares_kw_c[i] / zone_masses[i])
                            if t_pred > (T_maxs[i] + EPSILON):
                                needed = (t_pred - T_maxs[i]) * zone_masses[i]
                                actual_from_tank = min(needed, current_tank_kwh_c)
                                current_tank_kwh_c -= actual_from_tank
                                current_temps[i] = t_pred - (actual_from_tank / zone_masses[i])
                                if current_temps[i] > (T_maxs[i] + EPSILON):
                                    deficit_to_track_c += (current_temps[i] - T_maxs[i]) * zone_masses[i]
                            else:
                                current_temps[i] = t_pred
                        current_zone_flows_c[i] = ((hp_shares_kw_c[i] + actual_from_tank) * 3600) / (rho * cp * zone_dt_list_c[i])

            # --- STORE MATRICES ---
            results_matrix_H[idx, 0:9] = [current_time_val, total_demand_h, total_demand_c, available_power_h, actual_hp_out_h, current_tank_kwh_h, deficit_to_track_h, delta_t_window_h, tsat_tes_h]
            results_matrix_C[idx, 0:9] = [current_time_val, total_demand_h, total_demand_c, available_power_c, actual_hp_out_c, current_tank_kwh_c, deficit_to_track_c, delta_t_window_c, tsat_tes_c]
            strat_h_hist.append(active_strat_h)
            strat_c_hist.append(active_strat_c)
            
            col_ptr_h = 9
            col_ptr_c = 9
            for i in range(num_zones):
                results_matrix_H[idx, col_ptr_h:col_ptr_h+3] = [h_demands[i], current_temps[i], tsats_per_zone_h[i]]
                col_ptr_h += 3
                if SHOW_FLOW_RATES:
                    results_matrix_H[idx, col_ptr_h] = current_zone_flows_h[i]
                    col_ptr_h += 1
                    
                results_matrix_C[idx, col_ptr_c:col_ptr_c+3] = [c_demands[i], current_temps[i], tsats_per_zone_c[i]]
                col_ptr_c += 3
                if SHOW_FLOW_RATES:
                    results_matrix_C[idx, col_ptr_c] = current_zone_flows_c[i]
                    col_ptr_c += 1

        cols = ["Sim_Timestep", "Total_Demand_H_kW", "Total_Demand_C_kW", "Max_Supply_kW", "HP_Out_kW", "Tank_kWh", "Deficit_kW", "Window_h", "Tsat_TES"]
        for i in range(num_zones):
            cols.extend([f"Demand_Z{i+1}_kW", f"Temp_Z{i+1}", f"Tsat_Z{i+1}_h"])
            if SHOW_FLOW_RATES: cols.append(f"Flow_Z{i+1}_m3h")

        df_h = pd.DataFrame(results_matrix_H, columns=cols)
        df_h["Strategy"] = strat_h_hist
        df_c = pd.DataFrame(results_matrix_C, columns=cols)
        df_c["Strategy"] = strat_c_hist

        return df_h, df_c, T_mins, T_maxs

    # =================================================================
    # 4. RUN & OPTIMIZE (BISECTION SEARCH)
    # =================================================================
    def check_violations(df, T_mins, T_maxs, T_sets, tes_first_list):
        for i, (t_min, t_max, t_set, tes_first) in enumerate(zip(T_mins, T_maxs, T_sets, tes_first_list)):
            floor = t_set if tes_first else t_min
            if (df[f"Temp_Z{i+1}"] < (floor - 1e-6)).any(): return True
            ceiling = t_set if tes_first else t_max
            if (df[f"Temp_Z{i+1}"] > (ceiling + 1e-6)).any(): return True
        return False

    start_total = time.perf_counter()
    run_count = 0

    # --- 4.1 HOT TANK OPTIMIZATION ---
    if MANUAL_TANK_KWH_H is None:
        if not SILENT_MODE: print("Optimizing Hot Tank Size (Bisection Method)...")
        low_h, high_h = 0.0, 5000.0
        best_tank_h = high_h
        for i in range(15):
            mid_h = (low_h + high_h) / 2
            df_h, _, t_mins, t_maxs = simulate_merged_multi_zone(
                df_input, demand_cols, zone_masses, T_START_LIST, T_SET_LIST, T_SWING_LIST,
                mid_h, 5000.0,
                REHEAT_STRATEGY_H, DISCHARGE_STRATEGY_H, REHEAT_PRIORITY_H, DISCHARGE_PRIORITY_H, ZONE_TES_FIRST_H, ZONES_DELTA_T_H,
                REHEAT_STRATEGY_C, DISCHARGE_STRATEGY_C, REHEAT_PRIORITY_C, DISCHARGE_PRIORITY_C, ZONE_TES_FIRST_C, ZONES_DELTA_T_C,
                HP_MIN_MOD_KVA, HP_MIN_MOD_KVA, DRIFT_RATE
            )
            if check_violations(df_h, t_mins, t_maxs, T_SET_LIST, ZONE_TES_FIRST_H): low_h = mid_h
            else: high_h = mid_h; best_tank_h = mid_h
            if not SILENT_MODE: print(f"  Iter {i+1}: Testing {mid_h:.2f} kWh -> {'Violations' if low_h == mid_h else 'Pass'}")
        TANK_KWH_H_FINAL = best_tank_h
    else:
        TANK_KWH_H_FINAL = MANUAL_TANK_KWH_H
        if not SILENT_MODE: print(f"Using Manual Hot Tank Size: {TANK_KWH_H_FINAL} kWh")

    # --- 4.2 COLD TANK OPTIMIZATION ---
    if MANUAL_TANK_KWH_C is None:
        if not SILENT_MODE: print("\nOptimizing Cold Tank Size (Bisection Method)...")
        low_c, high_c = 0.0, 5000.0
        best_tank_c = high_c
        for i in range(15):
            mid_c = (low_c + high_c) / 2
            _, df_c, t_mins, t_maxs = simulate_merged_multi_zone(
                df_input, demand_cols, zone_masses, T_START_LIST, T_SET_LIST, T_SWING_LIST,
                TANK_KWH_H_FINAL, mid_c, 
                REHEAT_STRATEGY_H, DISCHARGE_STRATEGY_H, REHEAT_PRIORITY_H, DISCHARGE_PRIORITY_H, ZONE_TES_FIRST_H, ZONES_DELTA_T_H,
                REHEAT_STRATEGY_C, DISCHARGE_STRATEGY_C, REHEAT_PRIORITY_C, DISCHARGE_PRIORITY_C, ZONE_TES_FIRST_C, ZONES_DELTA_T_C,
                HP_MIN_MOD_KVA, HP_MIN_MOD_KVA, DRIFT_RATE
            )
            if check_violations(df_c, t_mins, t_maxs, T_SET_LIST, ZONE_TES_FIRST_C): low_c = mid_c
            else: high_c = mid_c; best_tank_c = mid_c
            if not SILENT_MODE: print(f"  Iter {i+1}: Testing {mid_c:.2f} kWh -> {'Violations' if low_c == mid_c else 'Pass'}")
        TANK_KWH_C_FINAL = best_tank_c
    else:
        TANK_KWH_C_FINAL = MANUAL_TANK_KWH_C
        if not SILENT_MODE: print(f"Using Manual Cold Tank Size: {TANK_KWH_C_FINAL} kWh")

    # --- 4.3 FINAL RUN ---
    run_count += 1
    df_res_h, df_res_c, final_t_mins, final_t_maxs = simulate_merged_multi_zone(
        df_input, demand_cols, zone_masses, T_START_LIST, T_SET_LIST, T_SWING_LIST,
        TANK_KWH_H_FINAL, TANK_KWH_C_FINAL,
        REHEAT_STRATEGY_H, DISCHARGE_STRATEGY_H, REHEAT_PRIORITY_H, DISCHARGE_PRIORITY_H, ZONE_TES_FIRST_H, ZONES_DELTA_T_H,
        REHEAT_STRATEGY_C, DISCHARGE_STRATEGY_C, REHEAT_PRIORITY_C, DISCHARGE_PRIORITY_C, ZONE_TES_FIRST_C, ZONES_DELTA_T_C,
        HP_MIN_MOD_KVA, HP_MIN_MOD_KVA, DRIFT_RATE
    )

    df_res_h['Net_Power_Deficit_kW'] = (df_res_h['Total_Demand_H_kW'] - df_res_h['HP_Out_kW']).clip(lower=0)
    df_res_h['Net_Flow_m3h'] = (df_res_h['Net_Power_Deficit_kW'] * 3600) / (rho * cp * delta_t_H)
    has_deficit_h = df_res_h['Net_Power_Deficit_kW'] > 1e-6
    peak_net_flow_h = df_res_h.loc[has_deficit_h, 'Net_Flow_m3h'].max() if has_deficit_h.any() else 0.0
    Vh_volume_m3_h = (peak_net_flow_h * TAU_STABILITY) / (1 - DEAD_ZONE)
    worst_idx_Vh_h = df_res_h.loc[has_deficit_h, 'Net_Flow_m3h'].idxmax() if has_deficit_h.any() else 0

    df_res_c['Net_Power_Deficit_kW'] = (df_res_c['Total_Demand_C_kW'] - df_res_c['HP_Out_kW']).clip(lower=0)
    df_res_c['Net_Flow_m3h'] = (df_res_c['Net_Power_Deficit_kW'] * 3600) / (rho * cp * delta_t_C)
    has_deficit_c = df_res_c['Net_Power_Deficit_kW'] > 1e-6
    peak_net_flow_c = df_res_c.loc[has_deficit_c, 'Net_Flow_m3h'].max() if has_deficit_c.any() else 0.0
    Vh_volume_m3_c = (peak_net_flow_c * TAU_STABILITY) / (1 - DEAD_ZONE)
    worst_idx_Vh_c = df_res_c.loc[has_deficit_c, 'Net_Flow_m3h'].idxmax() if has_deficit_c.any() else 0

    end_total = time.perf_counter()
    total_ms = (end_total - start_total) * 1000

    # =================================================================
    # 5. STYLED DISPLAY LOGIC (PRESERVED)
    # =================================================================
    def apply_styles(row):
        h_dem = row.get('Total_Demand_H_kW', 0)
        c_dem = row.get('Total_Demand_C_kW', 0)
        active_demand = max(h_dem, c_dem)
        if active_demand > row['Max_Supply_kW'] + 1e-4:
            return ['background-color: #002b5e; color: white'] * len(row)
        return [''] * len(row)

    def clean_format(x):
        if isinstance(x, (int, float)):
            if abs(x) < 1e-6: return "0"
            if x % 1 == 0: return f"{int(x)}"
            return f"{x:.2f}"
        return str(x)

    delta_u_h = cp * delta_t_H 
    v_c_h = (TANK_KWH_H_FINAL * 3600 * S) / (rho * delta_u_h * TOTAL_EFFICIENCY)
    delta_u_c = cp * delta_t_C 
    v_c_c = (TANK_KWH_C_FINAL * 3600 * S) / (rho * delta_u_c * TOTAL_EFFICIENCY)

    if not SILENT_MODE:
        print(f"\nOptimization Complete.")
        print(f"Total Simulation Runs: {run_count}")
        print(f"Total Time Taken: {end_total - start_total:.2f} seconds")
        print("-" * 33)
        print(f"HEATING SIDE:")
        print(f"Optimized Tank (Vc): {TANK_KWH_H_FINAL:.2f} kWh")
        print(f"Charging: {REHEAT_STRATEGY_H.title()} | Discharge: {DISCHARGE_STRATEGY_H.title()}")
        print(f"Final Volume (Vc): {v_c_h:.2f} m³")
        print(f"Final Volume (Vh): {Vh_volume_m3_h:.2f} m³")
        print("-" * 33)
        print("ZONE DISCHARGE STRATEGIES (H):")
        for i in range(num_zones):
            status = f"MAINTAIN T_SET at {T_SET_LIST[i]}°C" if ZONE_TES_FIRST_H[i] else f"ALLOW DRIFT from {T_SET_LIST[i]}°C down to {final_t_mins[i]:.2f}°C"
            print(f"Zone {i+1}: {status}")
        print("-" * 33)
        print("\n" + "="*40 + "\n")
        
        print("\n" + "="*30 + " TABLE 1: ENERGY CAPACITY (Vc_H) " + "="*30)
        min_tank_h = df_res_h['Tank_kWh'].min()
        worst_idx_vc_h = df_res_h[df_res_h['Tank_kWh'] <= min_tank_h + 1e-6].index[0]
        is_full_h = (df_res_h['Tank_kWh'] >= TANK_KWH_H_FINAL - 1e-6)
        start_idx_vc_h = df_res_h.index[(df_res_h.index < worst_idx_vc_h) & is_full_h][-1] if not df_res_h.index[(df_res_h.index < worst_idx_vc_h) & is_full_h].empty else 0
        end_idx_vc_h = df_res_h.index[(df_res_h.index > worst_idx_vc_h) & is_full_h][0] if not df_res_h.index[(df_res_h.index > worst_idx_vc_h) & is_full_h].empty else len(df_res_h) - 1
        df_zoom_vc_h = df_res_h.iloc[start_idx_vc_h : end_idx_vc_h + 1].copy()
        numeric_cols_h = df_zoom_vc_h.select_dtypes(include=['number']).columns
        display(df_zoom_vc_h.style.format({col: clean_format for col in numeric_cols_h}).hide().apply(apply_styles, axis=1))

        
        print("\n" + "="*30 + " TABLE 2: HYDRAULIC STABILITY (Vh_H) " + "="*30)
        df_zoom_Vh_h = df_res_h.iloc[max(0, worst_idx_Vh_h - 2) : min(len(df_res_h)-1, worst_idx_Vh_h + 2) + 1].copy()
        display(df_zoom_Vh_h.style.format({col: clean_format for col in numeric_cols_h}).hide().apply(lambda x: ['background-color: #5e0000; color: white' if x.name == worst_idx_Vh_h else '' for i in x], axis=1).set_caption("Peak Net Flow Heating Highlighted"))

       
        print(f"COOLING SIDE:")
        print(f"Optimized Tank (Vc): {TANK_KWH_C_FINAL:.2f} kWh")
        print(f"Charging: {REHEAT_STRATEGY_C.title()} | Discharge: {DISCHARGE_STRATEGY_C.title()}")
        print(f"Final Volume (Vc): {v_c_c:.2f} m³")
        print(f"Final Volume (Vh): {Vh_volume_m3_c:.2f} m³")
        print("-" * 33)
        print("ZONE DISCHARGE STRATEGIES (C):")
        for i in range(num_zones):
            status = f"MAINTAIN T_SET at {T_SET_LIST[i]}°C" if ZONE_TES_FIRST_C[i] else f"ALLOW DRIFT from {T_SET_LIST[i]}°C up to {final_t_maxs[i]:.2f}°C"
            print(f"Zone {i+1}: {status}")
        print("-" * 33)

        print("\n" + "="*30 + " TABLE 1: ENERGY CAPACITY (Vc_C) " + "="*30)
        min_tank_c = df_res_c['Tank_kWh'].min()
        worst_idx_vc_c = df_res_c[df_res_c['Tank_kWh'] <= min_tank_c + 1e-6].index[0]
        is_full_c = (df_res_c['Tank_kWh'] >= TANK_KWH_C_FINAL - 1e-6)
        start_idx_vc_c = df_res_c.index[(df_res_c.index < worst_idx_vc_c) & is_full_c][-1] if not df_res_c.index[(df_res_c.index < worst_idx_vc_c) & is_full_c].empty else 0
        end_idx_vc_c = df_res_c.index[(df_res_c.index > worst_idx_vc_c) & is_full_c][0] if not df_res_c.index[(df_res_c.index > worst_idx_vc_c) & is_full_c].empty else len(df_res_c) - 1
        df_zoom_vc_c = df_res_c.iloc[start_idx_vc_c : end_idx_vc_c + 1].copy()
        numeric_cols_c = df_zoom_vc_c.select_dtypes(include=['number']).columns
        display(df_zoom_vc_c.style.format({col: clean_format for col in numeric_cols_c}).hide().apply(apply_styles, axis=1))
        
        
        print("\n" + "="*30 + " TABLE 2: HYDRAULIC STABILITY (Vh_C) " + "="*30)
        df_zoom_Vh_c = df_res_c.iloc[max(0, worst_idx_Vh_c - 2) : min(len(df_res_c)-1, worst_idx_Vh_c + 2) + 1].copy()
        display(df_zoom_Vh_c.style.format({col: clean_format for col in numeric_cols_c}).hide().apply(lambda x: ['background-color: #5e0000; color: white' if x.name == worst_idx_Vh_c else '' for i in x], axis=1).set_caption("Peak Net Flow Cooling Highlighted"))
        
        
        print(f"Performance: {total_ms:.2f} ms (1 runs total)")

    return {
        "Hot_Tank_Capacity_kWh": TANK_KWH_H_FINAL,
        "Cold_Tank_Capacity_kWh": TANK_KWH_C_FINAL,
        "Hot_Tank_Volume_Vc_m3": v_c_h,
        "Hot_Tank_Volume_Vh_m3": Vh_volume_m3_h,
        "Cold_Tank_Volume_Vc_m3": v_c_c,
        "Cold_Tank_Volume_Vh_m3": Vh_volume_m3_c,
        "Execution_Time_ms": total_ms,
        "Heating_Result_DF": df_res_h,
        "Cooling_Result_DF": df_res_c
    }