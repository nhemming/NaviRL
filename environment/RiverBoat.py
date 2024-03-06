"""
A river boat that uses only a turning propeller to manuever. The throttle is fixed during operation currenty, but
can be changed to not be fixed.
"""

# native modules

# 3rd party modules
import copy
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np

# own modules
from environment.Entity import CollideEntity, CollisionCircle, CollisionRectangle


class RiverBoatEntity(CollideEntity):

    def __init__(self, area_air, area_water, bsfc, collision_shape, delta, delta_max, density_air, density_water,
                 fom, fuel, fuel_capacity, hull_len, hull_width, id, mass, moi, power, power_max, phi,
                 prop_diam, propeller_model, name):
        super(RiverBoatEntity, self).__init__( collision_shape, id, name)

        self.init_state_dict = dict()
        self.init_state_dict['time'] = 0.0
        self.init_state_dict['name'] = name
        self.init_state_dict['alpha'] = 0.0
        self.init_state_dict['area_air'] = area_air
        self.init_state_dict['area_water'] = area_water
        self.init_state_dict['bsfc'] = bsfc
        self.init_state_dict['delta'] = delta
        self.init_state_dict['delta_max'] = delta_max
        self.init_state_dict['density_air'] = density_air
        self.init_state_dict['density_water'] = density_water
        self.init_state_dict['fom'] = fom
        self.init_state_dict['fuel'] = fuel
        self.init_state_dict['fuel_capacity'] = fuel_capacity
        self.init_state_dict['hull_length'] = hull_len
        self.init_state_dict['hull_width'] = hull_width
        self.init_state_dict['mass'] = mass
        self.init_state_dict['moi'] = moi
        self.init_state_dict['power'] = power
        self.init_state_dict['power_max'] = power_max
        self.init_state_dict['phi'] = phi
        self.init_state_dict['prop_area'] = np.pi * prop_diam * prop_diam

        self.initialize_in_state_dict()

        # get parameters for simplified model
        if propeller_model == 'simple':
            self.populate_simplified_thrust_model()
        elif propeller_model == 'table_lookup':
            self.build_thrust_map()
        self.state_dict['propeller_model'] = propeller_model

    def initialize_in_state_dict(self):
        """
        Initialize the state of the boat to a default state
        :return:
        """
        # --------------------------------------------------------------------------------------------------------------
        # position data
        # --------------------------------------------------------------------------------------------------------------
        # x position in the global reference frame
        self.state_dict['x_pos'] = 0.0
        # y position in the global reference frame
        self.state_dict['y_pos'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # geometric data
        # --------------------------------------------------------------------------------------------------------------
        # relative angle of attack of the propeller disk [deg]
        self.state_dict['alpha'] = 0.0
        # angle [rad] of the propeller relative to the longitudinal axis of the boat
        self.state_dict['delta'] = 0.0
        # angle of the boat hull in the global reference frame where positive x is where the angle is measured too
        self.state_dict['phi'] = 0.0
        # the total mass of the boat [kg]
        self.state_dict['mass'] = 0.0
        # air projected area [m^2]
        self.state_dict['area_air'] = 0.0
        # water projected area [m^2]
        self.state_dict['area_water'] = 0.0
        # moment of interia of the boat []
        self.state_dict['moi'] = 0.0
        # lenght of the hull of the boat
        self.state_dict['hull_length'] = 0.0
        # the widest width of the hull of the boat
        self.state_dict['hull_width'] = 0.0
        # the disk area of the propeller excluding the aread of the spinner
        self.state_dict['prop_area'] = 0.0
        # effective angle of incidence of the air
        self.state_dict['phi_eff_air'] = 0.0
        # effective angle of incidence of the water
        self.state_dict['phi_eff_water'] = 0.0
        # angle between the boat and the destination [rad]
        self.state_dict['theta'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # velocity data
        # --------------------------------------------------------------------------------------------------------------
        # x (longitudinal/surge) velocity [m/s] in the boats local frame
        self.state_dict['v_xp'] = 0.0
        # y (lateral/sway) velocity [m/s] in the boats local frame
        self.state_dict['v_yp'] = 0.0
        # x velocity [m/s] in the global frame
        self.state_dict['v_x'] = 0.0
        # y velocity [m/s] in the global frame
        self.state_dict['v_y'] = 0.0
        # magnitude of the velocity [m/s]
        self.state_dict['v_mag'] = 0.0
        # rotational velocity
        self.state_dict['phi_dot'] = 0.0
        # effective longitudinal velocity of the air
        self.state_dict['v_x_eff_air'] = 0.0
        # effective lateral velocity of the air
        self.state_dict['v_y_eff_air'] = 0.0
        # effective longitudinal velocity of the water
        self.state_dict['v_x_eff_water'] = 0.0
        # effective lateral velocity of the water
        self.state_dict['v_y_eff_water'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # acceleration data
        # --------------------------------------------------------------------------------------------------------------
        # x (longitudinal/surge) acceleration [m/s^2] in the boats local frame
        self.state_dict['acc_xp'] = 0.0
        # y (lateral/sway) acceleration [m/s^2] in the boats local frame
        self.state_dict['acc_yp'] = 0.0
        # x acceleration [m/s^2] in the global reference frame
        self.state_dict['acc_x'] = 0.0
        # y acceleration [m/s^2] in the global reference frane
        self.state_dict['acc_y'] = 0.0
        # rotational acceleration
        self.state_dict['phi_double_dot'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # other data
        # --------------------------------------------------------------------------------------------------------------
        # the power [watt] that is deliverd to the propeller
        self.state_dict['power'] = 0.0
        # the thrust [N] that is produce by the propeller at a given power level
        self.state_dict['thrust'] = 0.0
        # the figure of merit for the propeller
        self.state_dict['fom'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # simulation data
        # --------------------------------------------------------------------------------------------------------------
        # the name of this boat
        self.state_dict['name'] = ''
        # wind velocity [m/s]
        self.state_dict['v_wind'] = np.array([0.0, 0.0])
        # current velocity [m/s]
        self.state_dict['v_current'] = np.array([0.0, 0.0])

        # --------------------------------------------------------------------------------------------------------------
        # forces and moments
        # --------------------------------------------------------------------------------------------------------------
        # axial air force
        self.state_dict['f_d_air'] = 0.0
        # lateral air force
        self.state_dict['f_s_air'] = 0.0
        # lateral water force
        self.state_dict['f_s_water'] = 0.0
        # axial water force
        self.state_dict['f_d_water'] = 0.0
        # longitudinal force of propeller in boats reference frame
        self.state_dict['fx_p'] = 0.0
        # lateral force of propeller in boats reference frame
        self.state_dict['fy_p'] = 0.0
        # moment induced by the air
        self.state_dict['m_air'] = 0.0
        # moment indueced by the water
        self.state_dict['m_water'] = 0.0
        # moment at 90 [deg] (pure moment?) need to describe this better
        self.state_dict['mr'] = 0.0
        # moment induce by the propeller
        self.state_dict['my_p'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # set the values of the boat
        # name of the boat
        self.state_dict['name'] = self.init_state_dict['name']
        # angle of attack of the propeller disk [rad]
        self.state_dict['alpha'] = self.init_state_dict['alpha']
        # cross sectional area of the part of the boat in the air [m^2]
        self.state_dict['area_air'] = self.init_state_dict['area_air']
        # cross sectional area of the part of the boat in the water [m^2]
        self.state_dict['area_water'] = self.init_state_dict['area_water']
        # sets the brake specific fuel consumption [kg/w-s]
        self.state_dict['bsfc'] = self.init_state_dict['bsfc']
        # propeller angle [rad]
        self.state_dict['delta'] = self.init_state_dict['delta']
        # set bounds for the propeller angle
        self.state_dict['delta_max'] = self.init_state_dict['delta_max']
        # air density [kg/m^3]
        self.state_dict['density_air'] = self.init_state_dict['density_air']
        # water density [kg/m^3]
        self.state_dict['density_water'] = self.init_state_dict['density_water']
        # figure of merit of the propeller
        self.state_dict['fom'] = self.init_state_dict['fom']
        # sets the current level of fuel on the boat [kg]
        self.state_dict['fuel'] = self.init_state_dict['fuel']
        # sets the maximum amount of fuel the boat can have [kg]
        self.state_dict['fuel_capacity'] = self.init_state_dict['fuel_capacity']
        # the length of the hull [m]
        self.state_dict['hull_length'] = self.init_state_dict['hull_length']
        # the length of the hull [m]
        self.state_dict['hull_width'] = self.init_state_dict['hull_width']
        # the total mass of the boat
        self.state_dict['mass'] = self.init_state_dict['mass']
        # set the moment of inertia of the boat [kg m^2]
        self.state_dict['moi'] = self.init_state_dict['moi']
        # current power level of the propeller [watt]
        self.state_dict['power'] = self.init_state_dict['power']
        # maximum power level of the propeller [watt]
        self.state_dict['power_max'] = self.init_state_dict['power_max']
        # angle of the hull to the positive x axis in the global frame [rad]
        self.state_dict['phi'] = self.init_state_dict['phi']
        # sets the disk area of the propeller counting the area of the spinner in the disk area [m]
        self.state_dict['prop_area'] = self.init_state_dict['prop_area']

    def add_step_history(self, sim_time):
        """
        save the state information for the current time steo
        :return:
        """
        self.state_dict['sim_time'] = sim_time

        # don't save entire state dict

        save_data = {
            'sim_time' :self.state_dict['sim_time'],
            'x_pos' :self.state_dict['x_pos'],
            'y_pos' : self.state_dict['y_pos'],
            'alpha': self.state_dict['alpha'],
            'delta' : self.state_dict['delta'],
            'is_collided' : self.state_dict['is_collided'],
            'phi' : self.state_dict['phi'],
            'theta' : self.state_dict['theta'],
            'v_xp':self.state_dict['v_xp'],
            'v_yp' : self.state_dict['v_yp'],
            'v_x': self.state_dict['v_x'],
            'v_y' : self.state_dict['v_y'],
            'v_mag' : self.state_dict['v_mag'],
            'phi_dot':self.state_dict['phi_dot'],
            #'v_x_eff_air':self.state_dict['v_x_eff_air'],
            #'v_y_eff_airself': self.state_dict['v_y_eff_air'],
            #'v_x_eff_water':self.state_dict['v_x_eff_water'],
            #'v_y_eff_water':self.state_dict['v_y_eff_water']
            'acc_xp':self.state_dict['acc_xp'],
            'acc_yp':self.state_dict['acc_yp'],
            'acc_x':self.state_dict['acc_x'],
            'acc_y':self.state_dict['acc_y'],
            'phi_double_dot':self.state_dict['phi_double_dot'],
            'thrust':self.state_dict['thrust'],
            #'v_wind':self.state_dict['v_wind'],
            #'v_current':self.state_dict['v_current'],
            'f_d_air':self.state_dict['f_d_air'],
            'f_s_air':self.state_dict['f_s_air'],
            'f_s_water':self.state_dict['f_s_water'],
            'f_d_water':self.state_dict['f_d_water'],
            'fx_p':self.state_dict['fx_p'],
            'fy_p':self.state_dict['fy_p'],
            'm_air':self.state_dict['m_air'],
            'm_water':self.state_dict['m_water'],
            'mr':self.state_dict['mr'],
            'my_p':self.state_dict['my_p'],
            'hull_length':self.state_dict['hull_length'],
            'hull_width': self.state_dict['hull_width']
        }

        self.history.append(save_data)

    def step(self, delta_t):
        """
        Use Euler stepping to step the entity one step in time.
        :param delta_t: Time step [s] to step over.
        :return:
        """

        # correct power if there is no fuel
        if self.state_dict['fuel'] <= 0.0:
            self.state_dict['power'] = 0.0
            self.state_dict['thrust'] = 0.0
        else:
            # get the thrust the propeller is currently outputing
            self.state_dict['thrust'] = self.calc_thrust([self.state_dict['v_xp'], self.state_dict['v_yp']],
                                                         self.state_dict['phi_dot'])

        # update v_x and v_y. Needed for first step. Look to place this somewhere else
        self.state_dict['v_x'] = self.state_dict['v_xp'] * np.cos(-self.state_dict['phi']) + self.state_dict[
            'v_yp'] * np.sin(-self.state_dict['phi'])
        self.state_dict['v_y'] = -self.state_dict['v_xp'] * np.sin(-self.state_dict['phi']) + self.state_dict[
            'v_yp'] * np.cos(-self.state_dict['phi'])

        # get the forces and moments of the boat. save them for telemetry later
        self.calc_forces_and_moments(self.state_dict['thrust'])

        fx_p = self.state_dict['f_d_air'] + self.state_dict['f_d_water'] + self.state_dict['fx_p']
        delta_xp = self.state_dict['v_xp'] * delta_t + 0.5 * fx_p / self.state_dict['mass'] * \
                   delta_t * delta_t

        fy_p = self.state_dict['f_s_air'] + self.state_dict['f_s_water'] + self.state_dict['fy_p']
        delta_yp = self.state_dict['v_yp'] * delta_t + 0.5 * fy_p / self.state_dict['mass'] * \
                   delta_t * delta_t

        mom = self.state_dict['m_air'] + self.state_dict['m_water'] + self.state_dict['my_p'] + self.state_dict['mr']
        delta_phi = self.state_dict['phi_dot'] * delta_t + 0.5 * mom * (
                self.state_dict['hull_length'] / 2.0) / self.state_dict['moi'] * delta_t * \
                    delta_t

        # update heading and bound to domain
        self.state_dict['phi'] = self.state_dict['phi'] + delta_phi
        if self.state_dict['phi'] > 2.0 * np.pi:
            self.state_dict['phi'] -= 2.0 * np.pi
        elif self.state_dict['phi'] < 0.0:
            self.state_dict['phi'] += 2.0 * np.pi

        # convert change in position to global frame
        delta_x = delta_xp * np.cos(-self.state_dict['phi']) + delta_yp * np.sin(-self.state_dict['phi'])
        delta_y = -delta_xp * np.sin(-self.state_dict['phi']) + delta_yp * np.cos(-self.state_dict['phi'])

        self.state_dict['x_pos'] = self.state_dict['x_pos'] + delta_x
        self.state_dict['y_pos'] = self.state_dict['y_pos'] + delta_y

        self.state_dict['v_xp'] = self.state_dict['v_xp'] + fx_p / self.state_dict['mass'] * delta_t
        self.state_dict['v_yp'] = self.state_dict['v_yp'] + fy_p / self.state_dict['mass'] * delta_t
        self.state_dict['phi_dot'] = self.state_dict['phi_dot'] + mom / self.state_dict['moi'] * delta_t

        self.state_dict['v_x'] = self.state_dict['v_xp'] * np.cos(-self.state_dict['phi']) + self.state_dict[
            'v_yp'] * np.sin(-self.state_dict['phi'])
        self.state_dict['v_y'] = -self.state_dict['v_xp'] * np.sin(-self.state_dict['phi']) + self.state_dict[
            'v_yp'] * np.cos(-self.state_dict['phi'])

        self.state_dict['v_mag'] = np.sqrt( self.state_dict['v_xp']*self.state_dict['v_xp'] + self.state_dict['v_yp']*self.state_dict['v_yp'] )

        self.state_dict['acc_xp'] = fx_p / self.state_dict['mass']
        self.state_dict['acc_yp'] = fy_p / self.state_dict['mass']
        self.state_dict['phi_double_dot'] = mom / self.state_dict['moi']

        # convert acceleration to global reference plane
        self.state_dict['acc_x'] = self.state_dict['acc_xp'] * np.cos(-self.state_dict['phi']) + self.state_dict[
            'acc_yp'] * np.sin(-self.state_dict['phi'])
        self.state_dict['acc_y'] = -self.state_dict['acc_xp'] * np.sin(-self.state_dict['phi']) + self.state_dict[
            'acc_yp'] * np.cos(-self.state_dict['phi'])

        # calculate the fuel used in the simulation
        fuel_used = self.state_dict['power'] * self.state_dict['bsfc'] * delta_t  # [kg of fuel]
        self.state_dict['fuel'] -= fuel_used

        if self.state_dict['fuel'] < 0:
            self.state_dict['fuel'] = 0.0

    def calc_forces_and_moments(self, thrust):
        """
        given the state of the boat and the current thrust level, determine the forces abd moments from the air, water,
        and the propeller. axial, transverse, and moments are found for both air and water. An additional moment is
        found from the rotational component of the boat. Forces and moments induced on the boat from the propeller
        are also found.

        :param state: a vector of x position, local x velocity, y position, local y velocity, angle, angular velocity
        :param thrust: the amount of thrust [N]. This is also a function of state
        :return:
            f_d_air - axial force from the air [N]
            f_s_air - lateral force from the air [N]
            m_air - moment induced from the air [N-m]
            f_d_hydro - axial force from the water [N]
            f_s_hydro - lateral force from the water [N]
            m_hydro - moment induced from the water [N-m]
            fx_p - axial force from the propeller [N]
            fy_p - lateral force from the propeller [N]
            my_p - moment induced from the propeller [N-m]
            mr - moment induced by the boat rotating in the water [N-m]
        """

        rho_air = self.state_dict['density_air']
        rho_water = self.state_dict['density_water']

        # in global coordinates
        v_eff_air = self.state_dict['v_wind'] - [self.state_dict['v_x'], self.state_dict['v_y']]
        self.state_dict['v_x_eff_air'] = v_eff_air[0]
        self.state_dict['v_y_eff_air'] = v_eff_air[1]
        rot_angle = -self.state_dict['phi']

        rot_mat = [[np.cos(rot_angle), -np.sin(rot_angle)],
                   [np.sin(rot_angle), np.cos(rot_angle)]]
        rot_mat = np.reshape(rot_mat, (2, 2))
        v_eff_air_local = np.matmul(rot_mat, v_eff_air)

        # air forces and moments
        phi_eff_air_local = np.arctan2(v_eff_air_local[1], v_eff_air_local[0])
        self.state_dict['phi_eff_air'] = phi_eff_air_local
        v_eff_air_local_mag = np.linalg.norm(v_eff_air_local)

        cd_aero, cs_aero, cy_aero, cr_aero = self.get_aero_coeffs(phi_eff_air_local)

        f_d_air = 0.5 * rho_air * v_eff_air_local_mag * v_eff_air_local_mag * self.state_dict['area_air'] * cd_aero
        f_s_air = 0.5 * rho_air * v_eff_air_local_mag * v_eff_air_local_mag * self.state_dict['area_air'] * cs_aero

        m_air = -0.5 * cy_aero * self.state_dict['area_air'] * self.state_dict[
            'hull_length'] * rho_air * v_eff_air_local_mag * v_eff_air_local_mag

        # --------------------------------------------------------------------------------------------------------------
        # water forces
        v_eff_water = self.state_dict['v_current'] - [self.state_dict['v_x'], self.state_dict['v_y']]
        self.state_dict['v_x_eff_water'] = v_eff_water[0]
        self.state_dict['v_y_eff_water'] = v_eff_water[1]
        rot_angle = -self.state_dict['phi']
        rot_mat = [[np.cos(rot_angle), -np.sin(rot_angle)],
                   [np.sin(rot_angle), np.cos(rot_angle)]]
        rot_mat = np.reshape(rot_mat, (2, 2))
        v_eff_water_local = np.matmul(rot_mat, v_eff_water)
        v_eff_water_local_mag = np.linalg.norm(v_eff_water_local)

        phi_eff_water_local = np.arctan2(v_eff_water_local[1], v_eff_water_local[0])
        self.state_dict['phi_eff_water'] = phi_eff_water_local
        cd_hydro, cs_hydro, cy_hydro, cr_hydro = self.get_hydro_coeffs(phi_eff_water_local)

        f_d_hydro = 0.5 * rho_water * v_eff_water_local_mag * v_eff_water_local_mag * \
                    self.state_dict['area_water'] * cd_hydro
        f_s_hydro = 0.5 * rho_water * v_eff_water_local_mag * v_eff_water_local_mag * \
                    self.state_dict['area_water'] * cs_hydro

        m_hydro = -0.5 * cy_hydro * self.state_dict['area_water'] * self.state_dict[
            'hull_length'] * rho_air * v_eff_water_local_mag * v_eff_water_local_mag

        mr = self.get_moment_hull(cr_hydro, self.state_dict['v_yp'])

        # propulsion forces
        fx_p = thrust * np.cos(self.state_dict['delta'])
        fy_p = thrust * np.sin(self.state_dict['delta'])
        my_p = -fy_p * self.state_dict['hull_length'] / 2.0

        # save all of the forces and moments
        self.state_dict['f_d_air'] = f_d_air
        self.state_dict['f_s_air'] = f_s_air
        self.state_dict['m_air'] = m_air
        self.state_dict['f_d_water'] = f_d_hydro
        self.state_dict['f_s_water'] = f_s_hydro
        self.state_dict['m_water'] = m_hydro
        self.state_dict['fx_p'] = fx_p
        self.state_dict['fy_p'] = fy_p
        self.state_dict['my_p'] = my_p
        self.state_dict['mr'] = mr

    def get_alpha(self, v_local, phi_dot):
        """
        determines the angle of attack or incidence of the flow travelling across the propeller disk

        :param v_local: velocity of the boat in its local reference frame [m/s]
        :param phi_dot: the rotational velocity of the boat [rad/s]
        :return: the angle of attack of the flow to the propeller disk
        """

        # lateral velocity induced at the propeller from the boat yawing
        v_rot = self.state_dict['hull_length'] / 2.0 * phi_dot
        v_local = np.array(v_local) * -1.0

        propeller_axial = np.array(
            [np.cos(self.state_dict['delta'] + np.pi / 2.0), np.sin(self.state_dict['delta'] + np.pi / 2.0)])
        alpha = self.get_angle_between_vectors(propeller_axial, [v_local[0], v_rot + v_local[1]], True)

        self.state_dict['alpha'] = alpha

        return alpha

    def get_angle_between_vectors(self, v1, v2, keep_sign):
        """
        uses the dot product or cross product to get the angle between two vectors. 2D vectors are assumed for this

        :param v1: first vector
        :param v2: second vector
        :param keep_sign: boolean for if the sign of the angle should be maintained or ignored
        :return: angle between vectors 1 and 2 [rad]
        """
        if keep_sign:
            # the sign the angle matters
            angle = np.arctan2(v2[1] * v1[0] - v2[0] * v1[1], v1[0] * v2[0] + v1[1] * v2[1])
        else:
            # the sign should be ignored
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        return angle

    def calc_thrust(self, v_local, phi_dot):
        """
        calculates the thrust delivered to the propeller based on the controlled power when using the non-simple model.
        The non-simple model needs to solve an implicit equation which is slower but more accurate. The simplified
        model approximates the response with out ----- effects.

        :param v_local: the velocity of the boat in its local reference frame
        :param phi_dot: the angular velocity of boat
        :return: the current thrust output of the propeller given a fixed power level
        """
        thrust = 0.0
        if self.state_dict['propeller_model'] == 'simple':
            thrust = self.simplified_thrust_model(v_local, phi_dot)
        elif self.state_dict['propeller_model'] == 'table_lookup':
            thrust = self.thrust_from_table(v_local,phi_dot)
        elif self.state_dict['propeller_model'] == 'solver':
            thrust = self.non_axial_momentum(v_local, phi_dot)

        return thrust

    def non_axial_momentum(self, v_local, phi_dot, alpha_d = None):
        v_mag = np.linalg.norm(v_local)

        if alpha_d is None:
            alpha_d = self.get_alpha(v_local, phi_dot)

        # Line search for inflecion point
        eps = 1e-3

        step = 0.1
        v_guess = 0.0
        f_eval_old = self.thrust_helper(v_guess, v_mag, alpha_d, self.state_dict['power'],
                                        self.state_dict['density_water'], self.state_dict['prop_area'])
        early_break = False
        while np.abs(step) > eps:

            v_guess += step
            f_eval_new = self.thrust_helper(v_guess, v_mag, alpha_d, self.state_dict['power'],
                                            self.state_dict['density_water'], self.state_dict['prop_area'])

            if f_eval_new < f_eval_old:
                # inflection point has been passed
                v_guess -= 2.0 * step
                step /= 10.0

            if np.sign(f_eval_new) != np.sign(f_eval_old):
                # a root has been crossed, and the loop can prematureing be broken
                early_break = True
                break

            f_eval_old = f_eval_new

        if early_break:
            # use bisect to solve for root as it has cross the x-axis

            a = 0
            fa = self.thrust_helper(a, v_mag, alpha_d, self.state_dict['power'], self.state_dict['density_water'],
                                    self.state_dict['prop_area'])
            b = v_guess
            error = 1.0
            eps = 1e-6
            while error > eps:

                c = (a + b) / 2.0
                fc = self.thrust_helper(c, v_mag, alpha_d, self.state_dict['power'], self.state_dict['density_water'],
                                        self.state_dict['prop_area'])

                if np.sign(fa) == np.sign(fc):
                    a = c
                    fa = fc
                else:
                    b = c

                error = (b - a) / 2.0

            v_induced = c

        else:
            # inflection point has not reached x axis

            # golden section search for root bounds
            # line search for bounds
            v_tmp = 1e-6  # initial guess
            k = 1
            GOLDEN_RATIO = 1.61803
            guess = [v_tmp]
            error = self.thrust_helper(v_tmp, v_mag, alpha_d, self.state_dict['power'],
                                       self.state_dict['density_water'], self.state_dict['prop_area'])
            errors = [error]
            delta = 0.1
            while True:

                v_tmp = v_tmp + delta * np.power(GOLDEN_RATIO, k)
                guess.append(v_tmp)
                error = self.thrust_helper(v_tmp, v_mag, alpha_d, self.state_dict['power'],
                                           self.state_dict['density_water'], self.state_dict['prop_area'])
                errors.append(error)

                if np.sign(errors[k - 1]) != np.sign(errors[k]):
                    break

                if v_tmp > 100.0:
                    # the search has failed. Assume induced velocity is zero
                    break

                k += 1

            # bisect search for root
            error = 1.0
            eps = 1e-3
            a = guess[k - 1]
            b = guess[k]
            fa = self.thrust_helper(a, v_mag, alpha_d, self.state_dict['power'], self.state_dict['density_water'],
                                    self.state_dict['prop_area'])
            while error > eps:

                c = (a + b) / 2.0
                fc = self.thrust_helper(c, v_mag, alpha_d, self.state_dict['power'], self.state_dict['density_water'],
                                        self.state_dict['prop_area'])

                if np.sign(fc) == np.sign(fa):
                    a = c
                    fa = fc
                else:
                    b = c

                error = (b - a) / 2.0

            v_induced = c

        thrust = self.state_dict['power'] / (
                v_mag + v_induced)  # 2.0*self.state_dict['density_water']*self.state_dict['prop_area']*v_induced*v_induced*np.sign(v_induced)
        return thrust

    def simplified_thrust_model(self, v_local, phi_dot):

        # get vel into the disk
        v_local = np.array(v_local) * -1.0
        v_rot = self.state_dict['hull_length'] / 2.0 * phi_dot
        propeller_axial = np.array(
            [np.cos(self.state_dict['delta'] + np.pi / 2.0), np.sin(self.state_dict['delta'] + np.pi / 2.0)])
        inflow_vel = np.dot(propeller_axial,[v_local[0], v_rot + v_local[1]])
        vel_reduce_factor = self.state_dict['thrust_loss_vel']*inflow_vel

        alpha_d = self.get_alpha(v_local, phi_dot)
        alpha_reduce_factor = self.state_dict['thrust_loss_alpha']*np.abs(alpha_d)
        if alpha_reduce_factor > 1:
            # should reach this correction
            alpha_reduce_factor = 1.0
        elif alpha_reduce_factor < 0.0:
            # should reach this correction
            alpha_reduce_factor = 0.0

        thrust = self.state_dict['thrust_max']*vel_reduce_factor*alpha_reduce_factor

        return thrust

    def thrust_from_table(self, v_local, phi_dot):
        v_local = np.array(v_local) * -1.0
        v_rot = self.state_dict['hull_length'] / 2.0 * phi_dot
        vel_mag = np.linalg.norm([v_local[0],v_local[1]+v_rot])
        alpha_d = np.abs(self.get_alpha(v_local, phi_dot)) # use absolute value because the map is symmetric

        # get the alphas that are below and above
        alpha_low = None
        alpha_high = None
        alpha_low_idx = None
        alpha_high_idx = None
        for i, tmp_a in enumerate(self.alpha_map):
            if i > 0:
                if tmp_a > alpha_d and self.alpha_map[i-1] < alpha_d:
                    alpha_low = self.alpha_map[i-1]
                    alpha_high = tmp_a
                    alpha_low_idx = i-1
                    alpha_high_idx = i
                    break

        # get the velocities that are below and above
        vel_low = None
        vel_high = None
        vel_low_idx = None
        vel_high_idx = None
        for i, tmp_v in enumerate(self.v_mag_map):
            if i > 0:
                if tmp_v > vel_mag and self.v_mag_map[i-1] < vel_mag:
                    vel_low = self.v_mag_map[i-1]
                    vel_high = tmp_v
                    vel_low_idx = i - 1
                    vel_high_idx = i
                    break

        if vel_low is None or vel_high is None or alpha_low is None or alpha_high is None:
            return 0.0 # something went wrong return zero

        # interpolation
        thrust_alow_v_low = self.thrust_map[alpha_low_idx,vel_low_idx]
        thrust_alow_v_high = self.thrust_map[alpha_low_idx, vel_high_idx]
        thrust_ahigh_v_low = self.thrust_map[alpha_high_idx, vel_low_idx]
        thrust_ahigh_v_high = self.thrust_map[alpha_high_idx, vel_high_idx]

        f = np.array([[thrust_alow_v_low, thrust_alow_v_high],
                      [thrust_ahigh_v_low, thrust_ahigh_v_high]])
        y = np.array([vel_high-vel_mag],[vel_mag-vel_low])
        x = np.array([alpha_high-alpha_d,alpha_d-alpha_low])
        thrust = 1.0/((alpha_high-alpha_low)*(vel_high-vel_low))*np.matmul(x,np.matmul(f,y))
        return thrust

    def thrust_helper(self, v, v0, alpha_d, power, rho, area):
        """
        calculates the error of the thrust equation given the current guess for total induced velocity

        :param v: current guess for total induced velocity [m/s]
        :param v0: magnitude of the velocity of the air flowing into the rotor disk not due to the rotor [m/s]
        :param alpha_d: the effective angle of incidence of the fluid flowing into the rotor disk [rad]
        :param power: the power applied to the rotor [watt]
        :param rho: the density of the fluid the rotor is in [kg/m^3]
        :param area: the disk area of the rotor [m^2]
        :return: error
        """
        vh = self.vh_calc(power, rho, area)

        p1 = np.power(v / vh, 4.0) + 2.0 * (v0 / vh) * np.power(v / vh, 3.0) * np.sin(alpha_d) + np.power(v0 / vh,
                                                                                                          2.0) * np.power(
            v / vh, 2.0)
        p2 = np.power((v0 * np.sin(alpha_d) + v) / vh, 2.0)

        return p1 * p2 - 1

    def vh_calc(self, power, rho, area):
        """
        calculates the induced velocity in hover based on the current operating conditions of rotor/propeller

        :param power: power delivered to the rotor [watt]
        :param rho: density of the fluid in the rotor [kg/m^3]
        :param area: disk area of the rotor [m^2]
        :return:
        """
        return np.power(power / (2.0 * rho * area), 1.0 / 3.0)

    def populate_simplified_thrust_model(self):
        # set propeller angle to 0.0
        self.state_dict['delta'] = 0.0

        # get thrust at alpha = 0, vel = 0
        thrust_a0_v0 = self.non_axial_momentum([0.0,0.0], 0.0)

        self.state_dict['thrust_max'] = thrust_a0_v0

        # get thrust at alpha = 0, vel = 0.5
        thrust_a0_v05 = self.non_axial_momentum([0.5, 0.0], 0.0) # TODO check if alpha is 0 deg here. It should be

        # get thrust decrement from finitie difference
        self.state_dict['thrust_loss_vel'] = (thrust_a0_v0-thrust_a0_v05)/0.5

        # get thrust at alpha = 90 [deg], vel = 0
        thrust_a90_v05 = self.non_axial_momentum([0.0, 0.5], 0.0)

        # get thrust decrement from alpha from finite difference
        self.state_dict['thrust_loss_alpha'] = (thrust_a0_v05 - thrust_a90_v05) / np.pi/2.0

    def build_thrust_map(self):

        self.state_dict['delta'] = 0.0

        # with alpha = 90 get when the thrust is 25% of max thrust. This defines the max velocity. build a 2d map to be interpolated later
        # secant search for when inflow flow velocity produces thrust that is 25% of max thrust
        alpha_secant = np.rad2deg(90)
        x0 = 0.0 # [m/s]
        x1 = 1.0 # [m/s]
        x2 = 0.0 # [m/s]
        fx0 = self.non_axial_momentum([x0,0.0],0.0, alpha_secant)
        v_max_thrust = copy.deepcopy(fx0)
        fx0 = fx0-v_max_thrust*0.25
        fx1 = self.non_axial_momentum([x1, 0.0], 0.0, alpha_secant)-v_max_thrust*0.25
        tol = 1e-3 # milli-Newton of thrust error
        err =1
        iterations = 0
        max_iterations = 50
        while err > tol and iterations < max_iterations:

            x2 = x1 - fx1*(x1-x0)/(fx1-fx0)
            fx2 = self.non_axial_momentum([x2, 0.0], 0.0, alpha_secant)-v_max_thrust*0.25

            err = np.abs(fx2)

            x0, x1 = x1, x2
            fx0, fx1 = fx1, fx2

            iterations += 1
        v_ten_max_thrust = x2

        v_delta = 0.25 # [m/s]
        v_test_x = np.arange(0,v_ten_max_thrust,v_delta)
        v_test_y = 0.0

        alpha_test = np.arange(0,np.pi/2.0,np.deg2rad(10.0))

        self.alpha_map = alpha_test
        self.v_mag_map = v_test_x
        self.thrust_map = np.zeros((len(alpha_test),len(v_test_x)))
        for i, alpha in enumerate(alpha_test):
            for j, vx in enumerate(v_test_x):
                self.thrust_map[i,j] = self.non_axial_momentum([vx,v_test_y],0.0, alpha)

    def set_control(self, power, propeller_angle_change):
        """
        Changes the heading (phi) of the entity by the passed in amount.
        :param phi_adj: The amount [rad] to change the heading.
        :return:
        """

        if isinstance(power,np.ndarray):
            power = power[0]
        if isinstance(propeller_angle_change,np.ndarray):
            propeller_angle_change = propeller_angle_change[0]

        # check for bounds
        if power > self.state_dict['power_max']:
            self.state_dict['power'] = self.state_dict['power_max']
        elif power < 0.0:
            self.state_dict['power'] = 0.0
        else:
            self.state_dict['power'] = power

        # check for bounds of the propeller angle
        propeller_angle = self.state_dict['delta'] - propeller_angle_change
        if propeller_angle < self.state_dict['delta_max'][0]:
            self.state_dict['delta'] = self.state_dict['delta_max'][0]
        elif propeller_angle > self.state_dict['delta_max'][1]:
            self.state_dict['delta'] = self.state_dict['delta_max'][1]
        else:
            self.state_dict['delta'] = propeller_angle

    def get_aero_coeffs(self, x):
        """
        get the aerodynamic coefficients that act on the boat from the wind and the relative wind induced by motion

        :param x: relative flow angle [deg]
        :return:
            axial - coefficient of axial flow
            side - coefficient of lateratl flow
            moment - coefficent for induced moment
            normal side - coefficient for when flow is directly perpendicular
        """

        # drag coefficient
        cd = 0.195738 + 0.518615 * np.abs(x) - 0.496029 * x * x + 0.0941925 * np.abs(x) ** 3 + \
             1.86427 * np.sin(2.0 * np.pi * np.power(np.abs(x) / np.pi, 1.05)) * np.exp(
            -2.17281 * np.power(np.abs(x) - np.pi / 2.0, 2.0))

        # side force coefficient
        cs = np.sign(x) * (12.3722 - 15.453 * np.abs(x) + 6.0261 * np.abs(x * x) - 0.532325 * np.abs(x) ** 3) * \
             np.sin(np.abs(x)) * np.exp(-1.68668 * np.power(np.abs(x) - np.pi / 2.0, 2.0))  # - np.sign(x)*0.1

        # yaw coefficient
        cy = np.sign(x) * (0.710204 - 0.297196 * np.abs(x) + 0.0857296 * np.abs(x * x)) * np.sin(
            np.pi * 2.0 * np.power(np.abs(x) / np.pi, 1.05))

        # perpendicular side force coefficient
        cr = 0.904313

        return cd, cs, cy, cr

    def get_hydro_coeffs(self, x):
        """
        get the hydrodynamic coefficients that act on the boat from the water and the relative current induced by motion

        :param x: relative flow angle [deg]
        :return:
            cd (axial) - coefficient of axial flow
            cs (side) - coefficient of lateratl flow
            cy (moment) - coefficent for induced moment
            cr (normal side) - coefficient for when flow is directly perpendicular
        """

        # drag coefficient
        cd = 0.245219 - 0.93044 * np.abs(x) + 0.745752 * np.abs(x * x) - 0.15915 * np.power(np.abs(x), 3.0) + \
             2.79188 * np.sin(2.0 * np.abs(x)) * np.exp(-1.05667 * np.power(np.abs(x) - np.pi / 2.0, 2.0))

        # side force coefficient
        cs = np.sign(x) * (0.115554 + 3.09423 * np.abs(x) - 0.984923 * x * x) * np.sin(np.abs(x))

        # yaw coefficient
        cy = np.sign(x) * (0.322986 + 0.317964 * np.abs(x) - 0.1021844 * x * x) * np.sin(2.0 * np.abs(x))

        # perpendicular side force coefficient
        cr = 2.545759

        return cd, cs, cy, cr

    def get_moment_hull(self, cr, vy):
        """
        get the moment induced on the hull of the boat by the water due to the boat rotating around is vertical axis

        :param cr: side force at phi_eff at 90[deg]. The flow velocity in the normal direction of the hull while spinning
        :param vy: effective transverse velocity
        :return:
        """

        l = self.state_dict['hull_length']
        omega = self.state_dict['phi_dot']
        # if np.abs(omega) < 1e-3:
        #    omega = 0.0
        alpha = cr * self.state_dict['area_water'] * self.state_dict['density_water'] / (l / 2.0)

        # forward porition
        mrf = l * l * alpha / 192.0 * (3 * l * l * omega * omega + 16.0 * l * omega * vy + 24.0 * vy * vy)

        if np.abs(vy) >= np.abs(omega * l / 2.0):
            mrb = -l * l * alpha / 192.0 * (3.0 * l * l * omega * omega - 16.0 * l * omega * vy + 24.0 * vy * vy)
        else:
            mrb = alpha / (192.0 * omega * omega) * (
                    np.power(l * omega - 2.0 * vy, 3.0) * (3.0 * l * omega + 2 * vy) - 16.0 * np.power(vy, 4.0))

        mr = mrf + mrb

        # adjust the direction of the moment based on the rate of rotation
        if self.state_dict['phi_dot'] < 0:
            mr = np.abs(mr)
        else:
            mr = -np.abs(mr)

        return mr

    def reset(self):

        self.initialize_in_state_dict()
        self.state_dict['v_mag'] = np.sqrt(
            self.state_dict['v_xp'] * self.state_dict['v_xp'] + self.state_dict['v_yp'] * self.state_dict['v_yp'])

    def reset_random(self):
        # reset the heading to a random vector
        self.initialize_in_state_dict()

        self.state_dict['phi'] = np.random.uniform(low=0, high=2.0 * np.pi)
        self.state_dict['phi_dot'] = np.random.uniform(low=-0.2, high=0.2)
        self.state_dict['delta'] = np.random.uniform(low=self.state_dict['delta_max'][0], high=self.state_dict['delta_max'][1])
        self.state_dict['v_xp'] = np.random.uniform(low=0.0,high=2.0)
        self.state_dict['v_yp'] = np.random.uniform(low=-0.5, high=0.5)
        self.state_dict['v_mag'] = np.sqrt( self.state_dict['v_xp']*self.state_dict['v_xp'] + self.state_dict['v_yp']*self.state_dict['v_yp'] )

    def apply_action(self, action_vec):
        """
        Accepts a vector produced by a neural network. It may not be the direct output, as the actions may have been
        mutated. This function dispatches the vector to the appropriate model changes.
        :param action_vec: Vector describing the value of change for the actuators.
        :return:
        """

        # adjust the heading of the entity
        self.set_control(self.state_dict['power'], action_vec)

    def draw_trajectory(self, ax, data, sim_time):

        # draw trajectory
        ax.plot(data['x_pos'],data['y_pos'])

        # draw shape
        if isinstance(self.collision_shape,CollisionRectangle):
            row = data.loc[data['sim_time'] == sim_time]

            cx = row['x_pos'].iloc[0]
            cy = row['y_pos'].iloc[0]
            hl = row['hull_length'].iloc[0]
            hw = row['hull_width'].iloc[0]
            x = cx-hl/2.0
            y = cy-hw/2.0
            phi = row['phi'].iloc[0]
            corners = [[hl / 2.0, -hw / 2.0],
                       [hl / 2.0, hw / 2.0],
                       [-hl / 2.0, hw / 2.0],
                       [-hl / 2.0, -hw / 2.0]]

            for i, corn in enumerate(corners):
                corn_new = copy.deepcopy(corn)
                corn[0] = corn_new[0] * np.cos(phi) - corn_new[1] * np.sin(phi) + cx
                corn[1] = corn_new[0] * np.sin(phi) + corn_new[1] * np.cos(phi) + cy

            corners = np.reshape(corners, (len(corners), 2))
            polygon = Polygon(corners, True, label='Boat')
            bp = [polygon]

            p = PatchCollection(bp, alpha=0.4)
            p.set_color('tab:blue')
            ax.add_collection(p)

            # draw propeller
            delta = row['delta'].iloc[0]
            prop_end_x = -hl / 2.0
            prop_end_y = 0.0
            prop_end_x_new = prop_end_x * np.cos(phi) - prop_end_y * np.sin(phi) + cx
            prop_end_y_new = prop_end_x * np.sin(phi) + prop_end_y * np.cos(phi) + cy
            base_vec = [prop_end_x_new - cx, prop_end_y_new - cy]
            base_vec = base_vec / np.linalg.norm(base_vec) * 2.0
            rot_mat = [[np.cos(delta), -np.sin(delta)],
                       [np.sin(delta), np.cos(delta)]]
            rot_mat = np.reshape(rot_mat, (2, 2))
            base_vec = np.matmul(rot_mat, base_vec)

            ax.plot([prop_end_x_new, prop_end_x_new + base_vec[0]], [prop_end_y_new, prop_end_y_new + base_vec[1]],
                     color='tab:blue')

    def draw_telemetry_trajectory(self, ax, data, sim_time):
        ax.plot(data['sim_time'],data['x_pos'],label='X')
        ax.plot(data['sim_time'], data['y_pos'], label='Y')
        ax.legend()

    def draw_telemetry_heading(self, ax, data, sim_time):
        ax.plot(data['sim_time'],data['phi'],label='X')

    def draw_telemetry_velocity(self, ax, data, sim_time):
        ax.plot(data['sim_time'],data['v_mag'],label='X')

    @staticmethod
    def get_default(id, name):
        """
        a basic boat for use in training. Based on the Grm model
        :return: a built boat that has the properties already populated
        """

        area_air = 15  # [m^2]
        area_water = 2.5  # [m^2]
        delta = 0
        delta_max = [-np.pi / 4.0, np.pi / 4.0]
        density_air = 1.225
        density_water = 998
        fom = 0.75  # figure of merit
        fuel = 10.0  # [kg]
        fuel_capacity = 10.0  # [kg]
        hull_len = 10.0
        hull_width = 2.5
        mass = 5000  # [kg]
        moi = 23000  # [kg m^2]
        phi = 0
        power = 9500  # [watt]
        power_max = 9500  # [watt]
        prop_diam = 0.25  # [m]

        # bsfc = 5.0e-8  # [kg/w-s] this is the realistic value
        bsfc = 5.0e-7  # [kg/w-s] this is the inefficient value for use
        propeller_model = 'solver'

        # TODO is the name, the name of the mover?
        collision_shape = CollisionRectangle(phi,hull_len,name,hull_len)

        rb = RiverBoatEntity(area_air, area_water, bsfc, collision_shape, delta, delta_max, density_air, density_water,
                 fom, fuel, fuel_capacity, hull_len, hull_width, id, mass, moi, power, power_max, phi,
                 prop_diam, propeller_model, name)

        return rb