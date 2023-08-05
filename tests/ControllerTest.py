"""
Test the controller to understand path
"""

# native modules
from collections import namedtuple
from unittest import TestCase

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# own modules
from environment.ActionOperation import BSplineControl
from environment.Controller import PDController
from environment.Entity import CollisionCircle, CollisionRectangle, get_collision_status
from environment.MassFreeVectorEntity import MassFreeVectorEntity
from environment.StaticEntity import StaticEntity


class CollisionTest(TestCase):

    def testMassFreeVectorAgentPath(self):

        mfve = MassFreeVectorEntity(None,0,"test")
        entities = {mfve.name: mfve}
        sensors = dict()

        sns.set_theme()
        fig = plt.figure(0, figsize=(14, 8))
        ax = fig.add_subplot(111)
        d_vec = [0.01,0.06]

        for k, d_tmp in enumerate(d_vec):

            coeffs = namedtuple("coeffs", "p d")
            coeffs.p = 1.0
            coeffs.d = d_tmp
            controller = PDController(coeffs)
            delta_t = 0.1

            frequency = 2.0
            is_continuous = False
            name = "bao"
            number_controls = 1
            #action_options_dict = {'option0': '-0.392699082,0,0.392699082','option1': '-0.392699082,0,0.392699082'}
            action_options_dict = {'option0': '-0.392699082,0,0.392699082', 'option1': '-0.392699082,0,0.392699082',  'option2': '-0.392699082,0,0.392699082'}

            segment_length = 1.0
            target_entity = mfve.name
            bSplineAo = BSplineControl(action_options_dict, controller, frequency, is_continuous, name, number_controls,
                         None, segment_length, target_entity)

            # set initial conditions of ao
            bSplineAo.start_location = [0,0]
            bSplineAo.start_angle = 0.0
            action_vector = 5

            mfve.state_dict['x_pos'] = 0.0
            mfve.state_dict['y_pos'] = -0.25
            x = [mfve.state_dict['x_pos']]
            y = [mfve.state_dict['y_pos']]


            path_angles = bSplineAo.action_options[action_vector]

            # build the b spline curve. from the called out angles
            control_points = np.zeros((len(path_angles) + 1, 2))
            control_points[0, :] = np.array(bSplineAo.start_location)
            cp_angle = np.zeros(len(path_angles))
            cp_angle[0] = bSplineAo.start_angle + path_angles[0]
            for i in range(len(path_angles)):
                control_points[i + 1, 0] = control_points[i, 0] + bSplineAo.segment_length * np.cos(cp_angle[i])
                control_points[i + 1, 1] = control_points[i, 1] + bSplineAo.segment_length * np.sin(cp_angle[i])
                if i < len(path_angles) - 1:
                    cp_angle[i + 1] = cp_angle[i] + path_angles[i + 1]

            samples = bSplineAo.bezier_curve(control_points, bSplineAo.n_samples)
            if k == 0:
                ax.plot(control_points[:,0],control_points[:,1],'o--',label='cp')
                ax.plot(samples[:,0],samples[:,1],'o-',label='path')

            t = 0
            max_t = 3.0
            while t < max_t:

                command = bSplineAo.convert_action( action_vector, delta_t, entities, sensors)

                mfve.adj_heading(command)

                mfve.step(delta_t)
                t += delta_t

                x.append(mfve.state_dict['x_pos'])
                y.append(mfve.state_dict['y_pos'])


            ax.plot(x,y,'x-',label=str(d_tmp))

        ax.legend()
        plt.tight_layout()
        plt.show()
