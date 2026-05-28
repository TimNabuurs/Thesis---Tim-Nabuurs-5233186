import pandas as pd
import numpy as np

class ZoneModel:
    def __init__(self, zone_name, **kwargs):
        self.zone_name = zone_name
        
        # --- Unpack all inputs from the orchestrator ---
        self.hourly_df = kwargs.get('hourly_df') 
        self.occ_profile = kwargs.get('occ_profile')
        self.equip_profile = kwargs.get('equip_profile')
        self.vent_profile = kwargs.get('vent_profile')
        self.system_profile = kwargs.get('system_profile')

        # Geometry
        self.floor_area = kwargs.get('floor_area')
        self.room_volume = kwargs.get('room_volume')
        self.area_roof = kwargs.get('area_roof')
        self.area_ground = kwargs.get('area_ground')
        self.total_facade_areas = kwargs.get('total_facade_areas')
        self.window_percentages = kwargs.get('window_percentages')
        self.glazing_percentages = kwargs.get('glazing_percentages')

        # Thermal Properties
        self.rc_facade = kwargs.get('rc_facade')
        self.rc_roof = kwargs.get('rc_roof')
        self.rc_ground_floor = kwargs.get('rc_ground_floor')
        self.u_value_windows = kwargs.get('u_value_windows')

        # Solar & Ventilation Constants
        self.alfai = kwargs.get('alfai')
        self.alfao = kwargs.get('alfao')
        self.solar_absorption_coefficient = kwargs.get('solar_absorption_coefficient')
        self.solar_heat_coefficient_shading = kwargs.get('solar_heat_coefficient_shading')
        self.solar_heat_coefficient_glazing = kwargs.get('solar_heat_coefficient_glazing')

        self.air_density = kwargs.get('air_density')
        self.air_heat_capacity = kwargs.get('air_heat_capacity')
        self.vent_flow_per_person = kwargs.get('vent_flow_per_person')
        self.infiltration_ach = kwargs.get('infiltration_ach')
        self.natural_vent_rate = kwargs.get('natural_vent_rate')

        # Internal Gains & System Constants
        self.max_people_per_m2 = kwargs.get('max_people_per_m2')
        self.heat_per_person = kwargs.get('heat_per_person')
        self.appliances_w_m2 = kwargs.get('appliances_w_m2')
        self.lighting_w_m2 = kwargs.get('lighting_w_m2')

        # HVAC & Ventilation System
        self.system_pressure_drop = kwargs.get('system_pressure_drop')
        self.efficiency_fan_and_motor = kwargs.get('efficiency_fan_and_motor')
        self.eta = kwargs.get('eta', 0.0) 

        # Simulation Constants
        self.heating_setpoint = kwargs.get('heating_setpoint')
        self.cooling_setpoint = kwargs.get('cooling_setpoint')
        self.vent_cooling_setpoint = kwargs.get('vent_cooling_setpoint')

        self.h_setpoint_profile = kwargs.get('heating_setpoint_profile')
        self.c_setpoint_profile = kwargs.get('cooling_setpoint_profile')
        self.vent_comfort_limit = kwargs.get('vent_comfort_limit', 23.5)

        self.thermal_mass_factor = kwargs.get('thermal_mass_factor', 165000) 
        self.thermal_capacity = self.thermal_mass_factor * self.floor_area
        self.dt = 3600
        self.t_ground = kwargs.get('t_ground')

        self.heating_power_max = kwargs.get('heating_power_max')
        self.cooling_power_max = kwargs.get('cooling_power_max')
        self.current_temp = kwargs.get('initial_temp', 21.0) 

        # Floor heating inputs
        self.u_floor_heating = kwargs.get('u_floor_heating', 11.0)
        self.floor_heating_area = kwargs.get('floor_heating_area', 0.0)
        self.t_floor_water_heating = kwargs.get('t_floor_water_heating', 35.0)
        self.t_floor_water_cooling = kwargs.get('t_floor_water_cooling', 18.0)
        self.floor_heating_activated = kwargs.get('floor_heating_activated', False)
        
        # Comfort limits for floor activation
        self.t_floor_limit_heating = kwargs.get('t_floor_limit_heating', 22.0)
        self.t_floor_limit_cooling = kwargs.get('t_floor_limit_cooling', 20.0)

        # Pre-calculate surface areas
        self.solar_orientations = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        self._initialize_surfaces()
        
    def _initialize_surfaces(self):
        self.area_windows = 0
        self.area_walls = 0
        self.glass_areas = {}
        self.opaque_wall_areas = {}

        for orient in self.solar_orientations:
            facade = self.total_facade_areas.get(orient, 0.0)
            win_perc = self.window_percentages.get(orient, 0.0)
            glaz_perc = self.glazing_percentages.get(orient, 1.0)
    
            window_area = facade * win_perc
            self.glass_areas[orient] = window_area * glaz_perc
            self.opaque_wall_areas[orient] = facade - window_area
            
            self.area_windows += window_area
            self.area_walls += self.opaque_wall_areas[orient]
            
        self.max_people = self.floor_area * self.max_people_per_m2
        self.flow_rate_vent = self.max_people * self.vent_flow_per_person 
        self.total_flow_rate_vent = self.vent_flow_per_person * self.max_people_per_m2 * self.floor_area 
        self.fans_power = (self.total_flow_rate_vent * self.system_pressure_drop) / (3600 * self.efficiency_fan_and_motor)
        
        self.annual_heating_results = []
        self.annual_cooling_results = []
        self.annual_temp_results = []

        self.detailed_results = {
            "Hour": [], "T_ext": [],
            "Trans window": [], "Trans wall": [], "Trans roof": [], "Transmission tot (W/K)": [],
            "Infiltration (W/K)": [], "Ventilation heat mode (W/K)": [], "Ventilation cool mode (W/K)": [],
            "Heating mode tot coupling (W/K)": [], "Cooling mode tot coupling (W/K)": [], "Ground coupling (W/K)": [],
            "Solar glazing (Wh)": [], "Solar opaque (Wh)": [],
            "People (Wh)": [], "Lighting (Wh)": [], "Appliances (Wh)": [], "Fans (Wh)": [], "Internal heat gains (Wh)": [],
            "T_i Max heating": [], "T_i free floating with H_heat": [], "T_i Max cooling": [], "T_i free floating with H_cool": [],
            "T_i indoor (oC)": [], "Q_heat (kWh)": [], "Q_cool (kWh)": [], "UFH_kWh": [],
            "Sch_Occupancy": [], "Sch_Equipment": [], "Sch_SystemActive": [], "Sch_Ventilation": []
        }

        for orient in self.solar_orientations:
            self.detailed_results[f"G_{orient}_(Wh)"] = []
            self.detailed_results[f"O_{orient}_(Wh)"] = []

    def calculate_nta8800_u_value(self, rc_value):
        if rc_value <= 0: return 0
        delta_u = 0.15 
        return (1 / (rc_value + 0.17)) + delta_u

    def calculate_hour_step(self, t, external_q_flow=0, t_source=None, is_winter=True):
        
        # DEBUG
        # if self.zone_name == 'Zone 2':
        #     print(f"DEBUG: ZoneModel Instance (ID: {id(self)}) | Flag is: {self.floor_heating_activated}")
        # DEBUG
        
        idx_168 = t % 168
        if self.h_setpoint_profile is not None:
            self.heating_setpoint = self.h_setpoint_profile[idx_168]
        if self.c_setpoint_profile is not None:
            self.cooling_setpoint = self.c_setpoint_profile[idx_168]

        if self.zone_name == "Zone 2":
            self.vent_cooling_setpoint = 27.0 
        else:
            self.vent_cooling_setpoint = max(self.vent_comfort_limit, self.cooling_setpoint - 0.1)

        u_value_walls = self.calculate_nta8800_u_value(self.rc_facade)        
        u_value_roof  = self.calculate_nta8800_u_value(self.rc_roof) 
        u_value_ground = self.calculate_nta8800_u_value(self.rc_ground_floor)       

        trans_e_windows = self.area_windows * self.u_value_windows
        trans_e_walls = self.area_walls * u_value_walls
        trans_e_roof = self.area_roof * u_value_roof
        H_e_transmission = trans_e_walls + trans_e_roof + trans_e_windows

        row = self.hourly_df.iloc[t]
        current_occ_val = self.occ_profile[idx_168]
        current_light_val = self.equip_profile[idx_168]
        current_vent_val = self.vent_profile[idx_168]
        current_sys = self.system_profile[idx_168]
        current_temp = self.current_temp

        t_ext = row['T'] 
        if t_source is None:
            t_source = t_ext  

        glazing_gains_breakdown = {}
        opaque_gains_breakdown = {}

        for orient in self.solar_orientations:
            if current_temp >= self.vent_cooling_setpoint and row[orient] > 0:
                active_shading = self.solar_heat_coefficient_shading
            else:
                active_shading = 1.0

            glazing_val = (row[orient] * self.glass_areas[orient] * self.solar_heat_coefficient_glazing * active_shading)
            glazing_gains_breakdown[orient] = glazing_val

            opaque_val = (row[orient] * self.opaque_wall_areas[orient] * self.solar_absorption_coefficient * (u_value_walls / self.alfao))
            opaque_gains_breakdown[orient] = opaque_val

        sun_glazing = sum(glazing_gains_breakdown.values())
        sun_opaque_walls = sum(opaque_gains_breakdown.values())
        sun_opaque_roof = row['Horizontal'] * self.solar_absorption_coefficient * (u_value_roof / self.alfao) * self.area_roof

        people_heat = self.max_people * current_occ_val * self.heat_per_person
        lighting_heat = self.lighting_w_m2 * self.floor_area * current_light_val
        equipment_heat = self.appliances_w_m2 * self.floor_area * current_light_val
        fans_heat = 0.5 * self.fans_power * current_vent_val                                        

        total_internal_gains = people_heat + equipment_heat + lighting_heat + (fans_heat * 0.5)     
        
        # DEBUG
        # if self.zone_name == 'Zone 2':
        #    print(f"DEBUG: Zone 2 floor_heating_activated is currently: {self.floor_heating_activated}")
        # DEBUG

        # Initialize floor heating/cooling contributions
        floor_heat_kwh = 0.0
        floor_cool_kwh = 0.0
        q_floor_gain = 0.0

        # --- 1. UFH Block ---
        q_floor_gain = 0.0
        if self.floor_heating_activated:
            u_eff = self.u_floor_heating if is_winter else 7.0
            t_water = self.t_floor_water_heating if is_winter else self.t_floor_water_cooling
            max_ufh_capacity_kw = self.floor_heating_area * 0.100 

            # DEBUG
            # print(f"DEBUG: Flux Calc | t_water: {t_water} | t_room: {self.current_temp} | Result: {u_eff * self.floor_heating_area * (t_water - self.current_temp)}")
            # DEBUG
            
            q_floor_gain_watts = u_eff * self.floor_heating_area * (t_water - self.current_temp)
            q_floor_gain_kw = q_floor_gain_watts / 1000.0
            q_floor_gain = min(q_floor_gain_kw, max_ufh_capacity_kw)

            floor_heat_kwh = max(q_floor_gain, 0)
            floor_cool_kwh = min(q_floor_gain, 0)

            # 2. Safety Override 
            if is_winter and self.current_temp > self.t_floor_limit_heating:
                q_floor_gain = 0.0
            elif not is_winter and self.current_temp < self.t_floor_limit_cooling:
                q_floor_gain = 0.0

        # --- 2. Virtual FCU Gain (New Thesis-Aligned Logic) ---
        q_atrium_effect = 0.0
        self.coil_UA = 0.5
        # Only zones other than the Atrium (Zone 2) benefit from the Atrium air
        if self.zone_name != "Zone 2" and t_source is not None:
            # Formula: Change in demand = Coil_UA * Delta_T
            delta_t_atrium = t_source - 21.0 # This it the fixed temperature setpoint for the other zones.
            q_atrium_effect = self.coil_UA * delta_t_atrium 
            
        # --- 3. Final Energy Balance ---
        # Note: We subtract q_atrium_effect because it's a reduction in required load
        total_free_gains = sun_glazing + sun_opaque_walls + sun_opaque_roof + \
                           total_internal_gains + external_q_flow + q_floor_gain + q_atrium_effect

        flow_rate_inf = self.room_volume * self.infiltration_ach
        h_inf = (flow_rate_inf * self.air_density * self.air_heat_capacity) / 3600
        h_ground = self.area_ground * u_value_ground

        current_flow_m3h = self.flow_rate_vent * current_occ_val * current_vent_val 

        h_vent_heat = (1 - self.eta) * (current_flow_m3h * self.air_density * self.air_heat_capacity) / 3600    
        h_vent_cool = (current_flow_m3h + (self.natural_vent_rate * self.room_volume)) * self.air_density * self.air_heat_capacity / 3600 

        H_total_heat = H_e_transmission + h_vent_heat + h_inf        
        H_total_cool = H_e_transmission + h_vent_cool + h_inf  

        Total_exterior_heat = ((H_e_transmission + h_inf) * t_ext) + (h_ground * self.t_ground) + (h_vent_heat * t_source)
        Total_exterior_cool = ((H_e_transmission + h_inf) * t_ext) + (h_ground * self.t_ground) + (h_vent_cool * t_source)

        H_total_with_ground_heat = H_total_heat + h_ground
        H_total_with_ground_cool = H_total_cool + h_ground
        
        den_test = 1 + (self.dt / self.thermal_capacity) * H_total_with_ground_heat
        t_check = (current_temp + (self.dt / self.thermal_capacity) * (total_free_gains + Total_exterior_heat)) / den_test

        t_eq_heat = (total_free_gains + Total_exterior_heat) / H_total_with_ground_heat
        t_eq_cool = (total_free_gains + Total_exterior_cool) / H_total_with_ground_cool

        if t_check > self.vent_cooling_setpoint and t_ext < current_temp:
            H_active = H_total_with_ground_cool
            self.current_ach = (current_flow_m3h / self.room_volume) + self.natural_vent_rate 
        else:
            H_active = H_total_with_ground_heat
            self.current_ach = (current_flow_m3h / self.room_volume) 

        eps = 1e-9 
        exp_factor_heat_raw = 1 - np.exp(-(H_total_with_ground_heat) / self.thermal_capacity * self.dt)        
        exp_factor_cool_raw = 1 - np.exp(-(H_total_with_ground_cool) / self.thermal_capacity * self.dt)        

        exp_factor_heat = max(exp_factor_heat_raw, eps)
        exp_factor_cool = max(exp_factor_cool_raw, eps)
        
        h_limit = 0.01  
        if H_total_with_ground_heat > h_limit:
            t_room_free_heating = current_temp + (t_eq_heat - current_temp) * exp_factor_heat
            t_room_max_heating = t_room_free_heating + (self.heating_power_max / H_total_with_ground_heat) * exp_factor_heat
        else:
            t_room_free_heating = current_temp + ((total_free_gains + Total_exterior_heat) / self.thermal_capacity) * self.dt
            t_room_max_heating = t_room_free_heating + (self.heating_power_max / self.thermal_capacity) * self.dt

        if H_total_with_ground_cool > h_limit:
            t_room_free_cooling = current_temp + (t_eq_cool - current_temp) * exp_factor_cool
            t_room_max_cooling = t_room_free_cooling + (self.cooling_power_max / H_total_with_ground_cool) * exp_factor_cool
        else:
            t_room_free_cooling = current_temp + ((total_free_gains + Total_exterior_cool) / self.thermal_capacity) * self.dt
            t_room_max_cooling = t_room_free_cooling + (self.cooling_power_max / self.thermal_capacity) * self.dt

        q_heat_kwh = 0.0
        q_cool_kwh = 0.0
        system_active = current_sys

        if system_active > 0:
            if t_room_max_heating < self.heating_setpoint:
                current_temp = t_room_max_heating
                q_heat_kwh = self.heating_power_max / 1000.0
            elif t_room_max_heating >= self.heating_setpoint and t_room_free_heating < self.heating_setpoint:
                current_temp = self.heating_setpoint
                p_needed = H_total_with_ground_heat * (self.heating_setpoint - t_room_free_heating) / exp_factor_heat
                q_heat_kwh = p_needed / 1000.0
            elif t_room_free_heating >= self.heating_setpoint and t_room_free_heating <= self.vent_cooling_setpoint:
                current_temp = t_room_free_heating
            elif t_room_free_heating > self.vent_cooling_setpoint and t_room_free_cooling < self.vent_cooling_setpoint:
                current_temp = self.vent_cooling_setpoint
            elif t_room_free_cooling >= self.vent_cooling_setpoint and t_room_free_cooling <= self.cooling_setpoint:
                current_temp = self.vent_cooling_setpoint
            elif t_room_max_cooling <= self.cooling_setpoint and t_room_free_cooling > self.cooling_setpoint:
                current_temp = self.cooling_setpoint
                if t_ext < self.cooling_setpoint:
                    p_needed = H_total_with_ground_cool * (self.cooling_setpoint - t_room_free_cooling) / exp_factor_cool
                else:
                    p_needed = H_total_with_ground_heat * (self.cooling_setpoint - t_room_free_heating) / exp_factor_heat
                q_cool_kwh = p_needed / 1000.0
            elif t_room_max_cooling > self.cooling_setpoint:
                current_temp = t_room_max_cooling
                q_cool_kwh = self.cooling_power_max / 1000.0
        else:
            current_temp = t_room_free_heating

        q_heat_total = q_heat_kwh + floor_heat_kwh 
        q_cool_total = q_cool_kwh + floor_cool_kwh

        self.current_temp = current_temp
        self.annual_heating_results.append(q_heat_total)
        self.annual_cooling_results.append(q_cool_total)
        self.annual_temp_results.append(current_temp)

        # Temperature update per timestep for the MPC controller
        self.temps = self.annual_temp_results

        d = self.detailed_results
        d["Hour"].append(t + 1)
        d["T_ext"].append(t_ext)
        d["Trans window"].append(trans_e_windows)
        d["Trans wall"].append(trans_e_walls)
        d["Trans roof"].append(trans_e_roof)
        d["Transmission tot (W/K)"].append(H_e_transmission)
        d["Infiltration (W/K)"].append(h_inf)
        d["Ventilation heat mode (W/K)"].append(h_vent_heat)
        d["Ventilation cool mode (W/K)"].append(h_vent_cool)
        d["Heating mode tot coupling (W/K)"].append(H_total_with_ground_heat)
        d["Cooling mode tot coupling (W/K)"].append(H_total_with_ground_cool)
        d["Ground coupling (W/K)"].append(h_ground)
        d["Solar glazing (Wh)"].append(sun_glazing)
        d["Solar opaque (Wh)"].append(sun_opaque_walls + sun_opaque_roof)
        d["People (Wh)"].append(people_heat)
        d["Lighting (Wh)"].append(lighting_heat)
        d["Appliances (Wh)"].append(equipment_heat)
        d["Fans (Wh)"].append(fans_heat)
        d["Internal heat gains (Wh)"].append(total_internal_gains)
        d["T_i Max heating"].append(t_room_max_heating)
        d["T_i free floating with H_heat"].append(t_room_free_heating)
        d["T_i Max cooling"].append(t_room_max_cooling)
        d["T_i free floating with H_cool"].append(t_room_free_cooling)
        d["T_i indoor (oC)"].append(current_temp)
        d["Sch_Occupancy"].append(current_occ_val)
        d["Sch_Equipment"].append(current_light_val)
        d["Sch_SystemActive"].append(current_sys)
        d["Sch_Ventilation"].append(current_vent_val)
        d["UFH_kWh"].append(q_floor_gain )
        
        for orient in self.solar_orientations:
            d[f"G_{orient}_(Wh)"].append(glazing_gains_breakdown.get(orient, 0))
            d[f"O_{orient}_(Wh)"].append(opaque_gains_breakdown.get(orient, 0))


        # DEBUG
        # if self.zone_name != "Zone 2":
        #    print(f"DEBUG: {self.zone_name} | Atrium Effect: {q_atrium_effect:.2f} | "
        #          f"Calculated Heating Demand: {q_heat_kwh:.2f} | "
        #          f"Calculated Cooling Demand: {q_cool_kwh:.2f}")
        # DEBUG

        return current_temp

    def run_simulation(self):
        self.annual_heating_results = []
        self.annual_cooling_results = []
        self.annual_temp_results = []
        for t in range(len(self.hourly_df)):
            self.calculate_hour_step(t, external_q_flow=0)
        return sum(self.annual_heating_results), sum(self.annual_cooling_results)
    
    def get_hourly_dataframe(self):
        return pd.DataFrame(self.detailed_results)