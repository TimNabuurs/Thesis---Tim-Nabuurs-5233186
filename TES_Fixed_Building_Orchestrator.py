import pandas as pd
import numpy as np
from TES_ZoneModel_Class import ZoneModel

def run_building_simulation(custom_floor_profile=None, silent_mode=False, export_excel=False):
    """
    Executes the multi-zone thermodynamic building loop.
    Accepts an underfloor heating profile array (length 168) for optimization loops.
    """
    h_day = [21.0] * 24
    c_day = [21.0] * 24
    heating_profile = (h_day * 7) 
    cooling_profile = (c_day * 7) 

    h_atrium = [16.0] * 24
    c_atrium = [28.0] * 24
    atrium_h_profile = h_atrium * 7
    atrium_c_profile = c_atrium * 7

    if custom_floor_profile is None:
        # Default baseline floor schedule fallback (ON 02:00 to 09:00)
        custom_floor_profile = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] + [0.0]*15
        custom_floor_profile = custom_floor_profile * 7

    base_inputs = {
        "t_ground": 15.0,
        "heating_setpoint_profile": heating_profile,
        "cooling_setpoint_profile": cooling_profile,
        "vent_comfort_limit": 21.0, 
        "initial_temp": 21.0,
        "heating_setpoint": 21.0,
        "cooling_setpoint": 21.0,
        "rc_roof": 7.84,
        "rc_ground_floor": 4.7,
        "u_value_windows": 1.84,
        "alfai": 7.5,
        "alfao": 15.0,
        "thermal_mass_factor": 450000,
        "solar_absorption_coefficient": 0.5,
        "solar_heat_coefficient_shading": 0.20,
        "solar_heat_coefficient_glazing": 0.35,
        "air_density": 1.2,
        "air_heat_capacity": 1003.0,
        "vent_flow_per_person": 36.0,
        "infiltration_ach": 0.2,
        "natural_vent_rate": 3.0,
        "heat_per_person": 65.0,
        "appliances_w_m2": 6.0,
        "lighting_w_m2": 6.0,
        "heating_power_max": 1000000.0,     
        "cooling_power_max": -1000000.0,    
        "system_pressure_drop": 200.0,
        "efficiency_fan_and_motor": 0.6,
        "eta": 0.0, 
        "u_floor_heating": 11.0,                
        "floor_heating_area": 200,              
        "t_floor_water_heating": 35.0,          
        "t_floor_water_cooling": 18.0,          
        "t_floor_limit_heating": 23.5,          
        "t_floor_limit_cooling": 19.0,          
        "system_profile": [1.0]*168
    }

    internal_wall_connections = [
        {"zones": ("Zone 1", "Zone 2"), "area": 505.0, "R": 0.33},  
        {"zones": ("Zone 1", "Zone 4"), "area": 122.0, "R": 0.60},  
        {"zones": ("Zone 1", "Zone 6"), "area": 35.0, "R": 1.00},   
        {"zones": ("Zone 3", "Zone 2"), "area": 355.0, "R": 0.33},  
        {"zones": ("Zone 3", "Zone 4"), "area": 295.0, "R": 0.33}, 
        {"zones": ("Zone 3", "Zone 5"), "area": 525.0, "R": 0.50},  
        {"zones": ("Zone 3", "Zone 6"), "area": 177.0, "R": 1.00}, 
        {"zones": ("Zone 4", "Zone 2"), "area": 208.0, "R": 0.50},
        {"zones": ("Zone 4", "Zone 5"), "area": 20.0, "R": 0.50},  
        {"zones": ("Zone 4", "Zone 6"), "area": 218.0, "R": 1.00}, 
        {"zones": ("Zone 5", "Zone 6"), "area": 266.0, "R": 1.00}, 
        {"zones": ("Zone 6", "Zone 2"), "area": 359.0, "R": 1.00}, 
    ]

    adjacencies = []
    for connection in internal_wall_connections:
        adjacencies.append({"zones": connection["zones"], "UA": connection["area"] / connection["R"]})

    zone_definitions = []
    
    # Zone 1
    z1 = base_inputs.copy()
    z1.update({"floor_area": 455.0, "area_roof": 455.0, "area_ground": 455.0, "room_volume": 4077.0, "rc_facade": 4.86, "max_people_per_m2": 0.55, "appliances_w_m2": 12.0, "lighting_w_m2": 15.0, "floor_heating_activated": False, "total_facade_areas": {'N': 255.0, 'NE': 0.0, 'E': 0.0, 'SE': 0.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.0}, "window_percentages": {'N': 0.0, 'NE': 0.0, 'E': 0.0, 'SE': 0.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.0}, "glazing_percentages": {'N': 0.0, 'NE': 0.0, 'E': 0.0, 'SE': 0.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.0}})
    z1["occ_profile"] = [0.0]*48 + [0.0]*15 + [0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 0.4] + [0.0]*24 + [0.0]*15 + [0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 0.4] + [0.0]*15 + [0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 0.4] + [0.0]*15 + [0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 0.4]
    z1["equip_profile"] = z1["occ_profile"]
    z1["vent_profile"] = ([0.10]*8 + [1.00]*16)*7
    zone_definitions.append(("Zone 1", z1))

    # Zone 2
    z2 = base_inputs.copy()
    z2.update({"floor_area": 565.0, "area_roof": 565.0, "area_ground": 565.0, "room_volume": 7370.0, "rc_roof": 13.66, "rc_facade": 4.7, "max_people_per_m2": 0.28, "appliances_w_m2": 10.0, "lighting_w_m2": 6.0, "floor_heating_activated": True, "heating_setpoint_profile": atrium_h_profile, "cooling_setpoint_profile": atrium_c_profile, "heating_setpoint": 16.0, "cooling_setpoint": 28.0, "vent_cooling_setpoint": 27.0, "total_facade_areas": {'N': 0.0, 'NE': 80.0, 'E': 0.0, 'SE': 205.0, 'S': 0.0, 'SW': 315.0, 'W': 0.0, 'NW': 230.0}, "window_percentages": {'N': 0.0, 'NE': 0.0, 'E': 0.00, 'SE': 0.293, 'S': 0.0, 'SW': 0.143, 'W': 0.0, 'NW': 0.187}, "glazing_percentages": {'N': 0.0, 'NE': 0.0, 'E': 0.0, 'SE': 0.80, 'S': 0.0, 'SW': 0.80, 'W': 0.0, 'NW': 0.80}})
    z2["occ_profile"] = ([0.0]*8 + [0.2]*10 + [0.3]*6)*2 + ([0.0]*8 + [0.2]*7 + [0.5]*4 + [0.3]*4 + [0.5]*1) + ([0.0]*8 + [0.2]*7 + [0.3]*10 + [0.3]*1) + ([0.0]*8 + [0.2]*7 + [0.5]*4 + [0.3]*4 + [0.3]*1) + ([0.0]*8 + [0.2]*7 + [0.5]*4 + [0.3]*4 + [0.5]*1)*2
    z2["equip_profile"] = z2["occ_profile"]
    z2["vent_profile"] = ([0.10]*8 + [1.00]*16)*7
    zone_definitions.append(("Zone 2", z2))

    # Zone 3
    z3 = base_inputs.copy()
    z3.update({"floor_area": 685.0, "area_roof": 160.0, "area_ground": 525.0, "room_volume": 4315.0, "rc_facade": 4.86, "max_people_per_m2": 0.26, "appliances_w_m2": 10.0, "lighting_w_m2": 12.0, "floor_heating_activated": False, "vent_flow_per_person": 25.2, "total_facade_areas": {'N': 0.0, 'NE': 93.0, 'E': 0.0, 'SE': 265.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 64.0}, "window_percentages": {'N': 0.0, 'NE': 0.032, 'E': 0.0, 'SE': 0.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.0}, "glazing_percentages": {'N': 0.0, 'NE': 0.80, 'E': 0.0, 'SE': 0.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.0}})
    z3["occ_profile"] = ([0.0]*12 + [0.4]*7 + [1.0]*4 + [0.0]*1)*7
    z3["equip_profile"] = z3["occ_profile"]
    z3["vent_profile"] = ([0.10]*8 + [1.00]*16)*7
    zone_definitions.append(("Zone 3", z3))

    # Zone 4
    z4 = base_inputs.copy()
    z4.update({"floor_area": 730.0, "area_roof": 295.0, "area_ground": 218.0, "room_volume": 2407.0, "rc_facade": 4.86, "max_people_per_m2": 0.06, "appliances_w_m2": 7.5, "lighting_w_m2": 6.0, "floor_heating_activated": False, "vent_flow_per_person": 25.2, "total_facade_areas": {'N': 0.0, 'NE': 13.0, 'E': 0.0, 'SE': 80.0, 'S': 0.0, 'SW': 150.0, 'W': 0.0, 'NW': 95.0}, "window_percentages": {'N': 0.0, 'NE': 0.0, 'E': 0.0, 'SE': 0.475, 'S': 0.0, 'SW': 0.00, 'W': 0.00, 'NW': 0.642}, "glazing_percentages": {'N': 0.0, 'NE': 0.0, 'E': 0.0, 'SE': 0.80, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.80}})
    z4["occ_profile"] = ([0.0]*8 + [1.0]*4 + [0.5]*1 + [1.0]*4 + [0.0]*7)*5 + ([0.0]*24)*2
    z4["equip_profile"] = z4["occ_profile"]
    z4["vent_profile"] = ([0.10]*8 + [1.00]*16)*7
    zone_definitions.append(("Zone 4", z4))

    # Zone 5
    z5 = base_inputs.copy()
    z5.update({"floor_area": 480.0, "area_roof": 319.0, "area_ground": 160.0, "room_volume": 2495.0, "rc_facade": 4.86, "max_people_per_m2": 0.09, "appliances_w_m2": 5.0, "lighting_w_m2": 6.0, "floor_heating_activated": False, "vent_flow_per_person": 25.2, "total_facade_areas": {'N': 0.0, 'NE': 176.0, 'E': 0.0, 'SE': 110.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 74.0}, "window_percentages": {'N': 0.0, 'NE': 0.034, 'E': 0.0, 'SE': 0.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.0}, "glazing_percentages": {'N': 0.0, 'NE': 0.80, 'E': 0.0, 'SE': 0.0, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.80}})
    z5["occ_profile"] = z4["occ_profile"]
    z5["equip_profile"] = z5["occ_profile"]
    z5["vent_profile"] = ([0.10]*8 + [1.00]*16)*7
    zone_definitions.append(("Zone 5", z5))

    # Zone 6
    z6 = base_inputs.copy()
    z6.update({"floor_area": 420.0, "area_roof": 360.0, "area_ground": 283.0, "room_volume": 3456.0, "rc_facade": 4.86, "max_people_per_m2": 0.03, "appliances_w_m2": 0.0, "lighting_w_m2": 0.0, "floor_heating_activated": False, "vent_flow_per_person": 25.2, "heating_setpoint_profile": [16.0]*168, "cooling_setpoint_profile": [28.0]*168, "vent_cooling_setpoint": 27.0, "total_facade_areas": {'N': 0.0, 'NE': 130.0, 'E': 0.0, 'SE': 133.0, 'S': 0.0, 'SW': 53.0, 'W': 0.0, 'NW': 75.0}, "window_percentages": {'N': 0.0, 'NE': 0.0, 'E': 0.0, 'SE': 0.083, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.0}, "glazing_percentages": {'N': 0.0, 'NE': 0.0, 'E': 0.0, 'SE': 0.80, 'S': 0.0, 'SW': 0.0, 'W': 0.0, 'NW': 0.0}})
    z6["occ_profile"] = ([0.0]*8 + [0.3]*14 + [0.1]*2)*2 + ([0.0]*8 + [0.3]*5 + [0.7]*2 + [1.0]*9) + ([0.0]*8 + [0.3]*14 + [0.1]*2) + ([0.0]*8 + [0.3]*5 + [0.7]*2 + [1.0]*9)*3
    z6["equip_profile"] = z6["occ_profile"]
    z6["vent_profile"] = ([0.10]*8 + [1.00]*16)*7
    zone_definitions.append(("Zone 6", z6))

    file_path = 'Input Multi Zone Model - NEN5060-B2 1% solar orient - Extreme Year.csv'
    weather_df = pd.read_csv(file_path, sep=';')
    weather_df.columns = weather_df.columns.str.strip()
    if 'temp_ext' in weather_df.columns and 'T' not in weather_df.columns:
        weather_df = weather_df.rename(columns={'temp_ext': 'T'})
    
    zones = {}
    for name, params in zone_definitions:
        params['hourly_df'] = weather_df 
        zones[name] = ZoneModel(name, **params)

    for t in range(len(weather_df)):
        zone_external_flows = {name: 0.0 for name in zones.keys()}
        for adj in adjacencies:
            z_a_name, z_b_name = adj["zones"]
            ua_value = adj["UA"]
            flow_b_to_a = ua_value * (zones[z_b_name].current_temp - zones[z_a_name].current_temp)
            zone_external_flows[z_a_name] += flow_b_to_a
            zone_external_flows[z_b_name] -= flow_b_to_a 

        forecast_end = min(t + 24, len(weather_df))
        upcoming_avg_temp = weather_df.iloc[t:forecast_end]['T'].mean()
        is_winter = True if upcoming_avg_temp < 15.5 else False
        hour_of_week = t % 168
        atrium_air_temp = zones['Zone 2'].current_temp

        for name, zone_obj in zones.items():
            current_source = atrium_air_temp if name != 'Zone 2' else None
            if name == 'Zone 2':
                schedule_val = custom_floor_profile[hour_of_week]
                is_installed = zone_definitions[1][1].get('floor_heating_activated', False)
                zone_obj.floor_heating_activated = bool(is_installed and schedule_val > 0)
        
            zone_obj.calculate_hour_step(t, external_q_flow=zone_external_flows[name], is_winter=is_winter, t_source=current_source)

    hourly_export_data = []
    for t in range(len(weather_df)):
        row = {"Hour": t + 1}
        for name, zone_obj in zones.items():
            if name == "Zone 6": continue
            net_demand = zone_obj.annual_heating_results[t] + zone_obj.annual_cooling_results[t]
            row[f"Demand_{name}"] = round(net_demand, 1)
        hourly_export_data.append(row)

    df_demands_combined = pd.DataFrame(hourly_export_data)
    df_demands_combined.to_csv("Output - Results_Yearly_Zone_Demands_Combined - FIXED.csv", index=False)
    
    TOTAL_GRID_LIMIT = 61.25    
    PUMP_POWER_W_M2  = 0.5     
    grid_kva_history = []

    for t in range(8760):
        total_building_load_w = 0
        for name, zone_obj in zones.items():
            idx_168 = t % 168
            total_building_load_w += (zone_obj.lighting_w_m2 * zone_obj.floor_area * zone_obj.equip_profile[idx_168])
            total_building_load_w += (zone_obj.appliances_w_m2 * zone_obj.floor_area * zone_obj.equip_profile[idx_168])
            total_building_load_w += (zone_obj.fans_power * zone_obj.vent_profile[idx_168])
            total_building_load_w += (PUMP_POWER_W_M2 * zone_obj.floor_area) if zone_obj.system_profile[idx_168] > 0 else 0
        
        building_kva = total_building_load_w / 1000.0
        grid_kva_history.append(max(0, TOTAL_GRID_LIMIT - building_kva))

    df_grid_final = pd.DataFrame({'Hour': range(1, 8761), 'Grid_kVA': grid_kva_history})
    df_grid_final.to_csv('Output - total kVA - Hour, Grid_kVA.csv', index=False)

    if not silent_mode:
        print("Simulation Completed successfully. Energy tables and Dynamic Grid Profiles exported.")

    # =================================================================
    # TOGGLEABLE SECTION: EXPORT TO EXCEL
    # =================================================================
    if export_excel:
        export_filename = 'Building_Thermal_Performance_Detailed_FIXED.xlsx'
        print(f"\nExporting detailed results to {export_filename}...")
        print("Please wait, calculating column widths for all zones...")

        # Form the standard summary output frame
        final_results = []
        for name, zone_obj in zones.items():
            final_results.append({
                "Zone": name, 
                "Annual Heating [kWh]": round(sum(zone_obj.annual_heating_results), 2), 
                "Annual Cooling [kWh]": round(sum(zone_obj.annual_cooling_results), 2)
            })
        results_df = pd.DataFrame(final_results)

        with pd.ExcelWriter(export_filename, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Annual_Summary', index=False)
            
            summary_ws = writer.sheets['Annual_Summary']
            for column_cells in summary_ws.columns:
                content_length = max(len(str(cell.value)) for cell in column_cells)
                summary_ws.column_dimensions[column_cells[0].column_letter].width = min(max(content_length, 10), 25)
            
            for name, zone_obj in zones.items():
                zone_detailed_df = zone_obj.get_hourly_dataframe()
                zone_detailed_df['Q_cool (kWh)'] = zone_obj.annual_cooling_results
                zone_detailed_df.to_excel(writer, sheet_name=name, index=False)
                
                worksheet = writer.sheets[name]
                for column_cells in worksheet.columns:
                    content_length = max(len(str(cell.value)) for cell in column_cells)
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(content_length, 10), 25)

        print(f"Export Complete. Detailed results saved in: {export_filename}")

    return df_demands_combined